"""Moebius (ECCV 2026) scene inpainting through its Places2 ONNX conversion."""

from __future__ import annotations

import hashlib
import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from wildlife_datasets.datasets import DatasetImage

_REVISION = "5bf1ef5d2861ec01a727183a3f95dc64f352120e"
_FILES = {"encoder": "vae_encoder.onnx", "denoiser": "unet.onnx", "decoder": "vae_decoder.onnx"}
_VAE_SCALE = np.float32(0.13025)


def _ddim_schedule(steps):
    betas = np.linspace(np.sqrt(0.00085), np.sqrt(0.012), 1000) ** 2
    alphas = np.cumprod(1 - betas)
    timesteps = (np.arange(steps, dtype=np.int64) * (1000 // steps))[::-1]
    return alphas, timesteps


def _ddim_step(sample, noise, alpha, previous_alpha):
    clean = (sample - np.sqrt(1 - alpha) * noise) / np.sqrt(alpha)
    return (np.sqrt(previous_alpha) * clean + np.sqrt(1 - previous_alpha) * noise).astype(np.float32)


class MoebiusBackgroundFill:
    """Reconstruct backgrounds with the 226M-parameter Moebius model (2026).

    Uses Simon Willison's FP32 ONNX conversion of the official Places2 scene
    checkpoint and its shared PixelHacker VAE. The three files total about
    1.24 GB and are cached after the first call. No text encoder or prompt is
    needed. Install ``wildlife-datasets[inpainting]`` for ONNX Runtime.

    Apple Silicon uses Core ML with GPU acceleration when available. The first
    call compiles the graphs; compiled models and sessions are reused. CPU is
    supported, but is substantially slower. Returned RGB PIL images retain the
    original dimensions and every pixel outside the expanded mask.

    Args:
        model_dir: Optional directory containing the three ONNX files for
            offline use. Otherwise download the pinned conversion from the Hub.
        num_inference_steps: Sampling steps, default 20.
        guidance_scale: Classifier-free guidance, default 2.0.
        seed: Seed reset for each fill; None uses fresh randomness.
        mask_dilation: Expansion in source-image pixels, default 8.
        num_threads: CPU worker threads, default 4.
        providers: Optional ONNX execution providers. Defaults to Core ML on
            supported Macs, CUDA when available, otherwise CPU.
        cache_dir: Optional Hugging Face and compiled-model cache directory.
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        *,
        num_inference_steps: int = 20,
        guidance_scale: float = 2.0,
        seed: int | None = 0,
        mask_dilation: int = 8,
        num_threads: int = 4,
        providers=None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if not isinstance(num_inference_steps, int) or not 2 <= num_inference_steps <= 1000:
            raise ValueError("num_inference_steps must be an integer between 2 and 1000.")
        if not np.isfinite(guidance_scale) or guidance_scale < 0:
            raise ValueError("guidance_scale must be finite and nonnegative.")
        if not isinstance(mask_dilation, int) or mask_dilation < 0:
            raise ValueError("mask_dilation must be a nonnegative integer.")
        if not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError("num_threads must be a positive integer.")
        self.model_dir = model_dir
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.mask_dilation = mask_dilation
        self.num_threads = num_threads
        self.providers = providers
        self.cache_dir = cache_dir
        self._sessions = None

    def _get_sessions(self):
        if self._sessions is not None:
            return self._sessions
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError('Moebius requires pip install "wildlife-datasets[inpainting]".') from error
        from huggingface_hub import constants, hf_hub_download

        paths = {
            name: str(Path(self.model_dir) / filename)
            if self.model_dir is not None
            else hf_hub_download("simonw/Moebius-ONNX", filename, revision=_REVISION, cache_dir=self.cache_dir)
            for name, filename in _FILES.items()
        }
        cache = Path(self.cache_dir or constants.HF_HUB_CACHE) / "moebius-coreml" / _REVISION
        providers = self.providers
        if providers is None:
            available = ort.get_available_providers()
            providers = ["CPUExecutionProvider"]
            if "CoreMLExecutionProvider" in available:
                providers.insert(
                    0,
                    (
                        "CoreMLExecutionProvider",
                        {
                            "ModelFormat": "MLProgram",
                            "MLComputeUnits": "CPUAndGPU",
                            "RequireStaticInputShapes": "1",
                            "ModelCacheDirectory": str(cache),
                        },
                    ),
                )
            elif "CUDAExecutionProvider" in available:
                providers.insert(0, "CUDAExecutionProvider")
        if any((p[0] if isinstance(p, tuple) else p) == "CoreMLExecutionProvider" for p in providers):
            paths["denoiser"] = _coreml_denoiser(paths["denoiser"], cache, self.num_threads)
        sessions = {}
        for name, path in paths.items():
            options = ort.SessionOptions()
            options.intra_op_num_threads = self.num_threads
            options.add_free_dimension_override_by_name("B", 2 if name == "denoiser" else 1)
            sessions[name] = ort.InferenceSession(path, sess_options=options, providers=providers)
        self._sessions = sessions
        return self._sessions

    @staticmethod
    def _run(session, inputs, shape):
        result = np.asarray(session.run(None, inputs)[0])
        if result.shape != shape or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError("Unexpected or nonfinite output from a Moebius ONNX graph.")
        return result

    def _generate(self, image, mask, seed):
        sessions = self._get_sessions()
        rng = np.random.default_rng(seed)
        pixels = image.astype(np.float32).transpose(2, 0, 1)[None] / np.float32(127.5) - 1
        hole = mask.astype(np.float32)[None, None]

        moments = self._run(sessions["encoder"], {"image": np.ascontiguousarray(pixels * (1 - hole))}, (1, 8, 64, 64))
        mean, logvar = np.split(moments, 2, axis=1)
        masked = (
            mean
            + np.exp(np.clip(logvar, -30, 20) * np.float32(0.5)) * rng.standard_normal(mean.shape, dtype=np.float32)
        ) * _VAE_SCALE
        latent_mask = hole[:, :, ::8, ::8]
        # Full-noise initialization keeps removed animal pixels out of the fill.
        latent = rng.standard_normal(masked.shape, dtype=np.float32)
        latent += np.float32(0.0357) * rng.standard_normal((1, 4, 1, 1), dtype=np.float32)
        alphas, timesteps = _ddim_schedule(self.num_inference_steps)
        ids = np.stack([np.arange(10, 20), np.arange(10)]).astype(np.int64)
        for index, t in enumerate(timesteps):
            sample = np.concatenate([latent, latent_mask, masked], axis=1)
            prediction = self._run(
                sessions["denoiser"],
                {
                    "latent": np.concatenate([sample, sample]),
                    "timesteps": np.array([t, t], dtype=np.int64),
                    "input_ids": ids,
                },
                (2, 4, 64, 64),
            )
            unconditional, conditional = prediction[:1], prediction[1:]
            prediction = unconditional + self.guidance_scale * (conditional - unconditional)
            previous = alphas[timesteps[index + 1]] if index + 1 < len(timesteps) else 1.0
            latent = _ddim_step(latent, prediction, alphas[t], previous)
        decoded = self._run(sessions["decoder"], {"latent": latent / _VAE_SCALE}, (1, 3, 512, 512))
        return np.rint(np.clip((decoded[0].transpose(1, 2, 0) + 1) * 127.5, 0, 255)).astype(np.uint8)

    def __call__(self, item: DatasetImage, *, seed: int | None = None) -> Image.Image:
        """Fill an item's mask; empty masks return a copy without loading models."""
        if item.mask is None:
            raise ValueError("This image has no foreground mask; background filling is unavailable.")
        image = item.image.convert("RGB")
        mask = np.asarray(item.mask, dtype=bool)
        if mask.shape != (image.height, image.width):
            raise ValueError("The foreground mask must have the same height and width as the image.")
        if not mask.any():
            return image.copy()
        import cv2

        expanded = mask.astype(np.uint8)
        if self.mask_dilation:
            size = 2 * self.mask_dilation + 1
            expanded = cv2.dilate(expanded, np.ones((size, size), dtype=np.uint8))
        x, y, width, height = cv2.boundingRect(expanded)
        left, top = max(0, x - 128), max(0, y - 128)
        right, bottom = min(image.width, x + width + 128), min(image.height, y + height + 128)
        crop = np.array(image)[top:bottom, left:right]
        crop_mask = expanded[top:bottom, left:right]
        # Remove hidden RGB before interpolation as well as before VAE encoding.
        crop[crop_mask > 0] = 128
        scale = 512 / max(crop.shape[:2])
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        working = cv2.resize(crop, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        hole = cv2.resize(crop_mask.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0
        # Fixed square graphs: letterbox with edge pixels, preserving aspect ratio.
        pad_y, pad_x = 512 - size[1], 512 - size[0]
        padding = ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2))
        working = np.pad(working, (*padding, (0, 0)), mode="edge")
        hole = np.pad(hole, padding, mode="constant")
        # Cover every touched VAE cell so small holes survive the 8x downsample.
        hole = hole.reshape(64, 8, 64, 8).any(axis=(1, 3)).repeat(8, axis=0).repeat(8, axis=1)
        generated = self._generate(working, hole, self.seed if seed is None else seed)
        generated = generated[padding[0][0] : padding[0][0] + size[1], padding[1][0] : padding[1][0] + size[0]]
        generated = cv2.resize(generated, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LINEAR)
        alpha = crop_mask * 255
        if self.mask_dilation:
            distance = cv2.distanceTransform(crop_mask, cv2.DIST_L2, 3)
            alpha = np.rint(np.minimum(distance / self.mask_dilation, 1) * 255).astype(np.uint8)
            alpha[mask[top:bottom, left:right]] = 255
        output = image.copy()
        output.paste(Image.fromarray(generated), (left, top), Image.fromarray(alpha))
        return output


# Core ML graph optimization, cached after the first use.
def _replace_einsums(model):
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.shape_inference.infer_shapes(model, data_prop=True)
    values = list(model.graph.input) + list(model.graph.value_info)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in values}
    shapes.update({v.name: list(v.dims) for v in model.graph.initializer})
    nodes = []
    for original in model.graph.node:
        if original.op_type != "Einsum":
            nodes.append(original)
            continue
        equation = next(
            helper.get_attribute_value(a).decode().replace(" ", "") for a in original.attribute if a.name == "equation"
        )
        left, rest = equation.split(",")
        right, output = rest.split("->")
        if any(len(set(labels)) != len(labels) or not labels.isalpha() for labels in (left, right, output)):
            raise ValueError("Unsupported Moebius attention equation.")
        dims = {}
        for labels, value in zip((left, right), original.input):
            for label, size in zip(labels, shapes[value]):
                if not size or (label in dims and dims[label] != size):
                    raise ValueError("Moebius attention requires fixed, matching dimensions.")
                dims[label] = size
        batch = [x for x in output if x in left and x in right]
        reduction = [x for x in left if x in right and x not in output]
        free_left = [x for x in left if x not in batch + reduction]
        free_right = [x for x in right if x not in batch + reduction]
        order = batch + free_left + free_right
        if sorted(order) != sorted(output):
            raise ValueError("Unsupported Moebius attention contraction.")
        prefix = original.name + "/matmul"

        def node(op, inputs, suffix, **kwargs):
            name = prefix + "/" + suffix
            nodes.append(helper.make_node(op, inputs, [name], name=name, **kwargs))
            return name

        def shape(value, suffix):
            name = prefix + "/" + suffix
            model.graph.initializer.append(numpy_helper.from_array(np.array(value, dtype=np.int64), name=name))
            return name

        def operand(value, labels, target, newshape, suffix):
            transposed = node("Transpose", [value], suffix + "-transpose", perm=[labels.index(x) for x in target])
            return node("Reshape", [transposed, shape(newshape, suffix + "-shape")], suffix + "-reshape")

        common = [dims[x] for x in batch]
        contract = math.prod(dims[x] for x in reduction)
        a = operand(
            original.input[0],
            left,
            batch + free_left + reduction,
            common + [math.prod(dims[x] for x in free_left), contract],
            "left",
        )
        b = operand(
            original.input[1],
            right,
            batch + reduction + free_right,
            common + [contract, math.prod(dims[x] for x in free_right)],
            "right",
        )
        product = node("MatMul", [a, b], "product")
        restored = node("Reshape", [product, shape([dims[x] for x in order], "output-shape")], "restore")
        nodes.append(
            helper.make_node(
                "Transpose",
                [restored],
                list(original.output),
                name=prefix + "/output",
                perm=[order.index(x) for x in output],
            )
        )
    model.graph.ClearField("node")
    model.graph.node.extend(nodes)
    return model


def _coreml_denoiser(path, cache, num_threads):
    source = Path(path).resolve()
    stat = source.stat()
    key = hashlib.sha256(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    cache = Path(cache)
    target = cache / f"denoiser-matmul-v1-{key}.onnx"
    if target.exists():
        return str(target)

    import onnxruntime as ort

    try:
        import onnx
    except ImportError as error:
        raise ImportError('Moebius on Core ML requires pip install "wildlife-datasets[inpainting]".') from error

    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache) as directory:
        temporary = Path(directory) / "denoiser.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = num_threads
        options.add_free_dimension_override_by_name("B", 2)
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        options.optimized_model_filepath = str(temporary)
        # Fold shape arithmetic before reshaping contractions into MatMul.
        session = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
        del session
        model = _replace_einsums(onnx.load(temporary))
        onnx.checker.check_model(model)
        onnx.save(model, temporary)
        temporary.replace(target)
    return str(target)
