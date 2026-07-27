import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from scripts.BrownBearHeads.predict_keypoints import (
    flatten_prediction,
    flatten_scores,
    import_pose_code,
    keypoints_out_of_bounds,
    resolve_checkpoint,
    resolve_image_path,
    write_head_keypoints_sidecar,
)
from scripts.BrownBearHeads.prepare_metadata import main as prepare_metadata
from scripts.BrownBearHeads.resize_and_restructure import main as resize_and_restructure
from wildlife_datasets.datasets import BrownBearHeads


class TestBrownBearHeads(unittest.TestCase):
    def test_create_catalogue_from_prepared_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = pd.DataFrame(
                [
                    {
                        "image_id": 10,
                        "identity": "bear-1",
                        "path": "BrownBearHeads/2017/bear-1/image_1.jpg",
                        "width": 720,
                        "height": 540,
                        "width_original": 1440,
                        "height_original": 1080,
                        "date": "2017-07-15 12:30:00",
                        "year": 2017,
                        "camera": "cam-1",
                        "split_2017": "test",
                        "split_2018": "train",
                        "split_2019": "train",
                        "split_2020": "train",
                        "split_2021": "train",
                        "split_2022": "train",
                        "split_ood": "train",
                        "split_iid": "train",
                    },
                    {
                        "image_id": 20,
                        "identity": "bear-2",
                        "path": "BrownBearHeads/2022/bear-2/image_2.jpg",
                        "width": 700,
                        "height": 700,
                        "width_original": 1200,
                        "height_original": 1200,
                        "date": "2022-08-01 08:00:00",
                        "year": 2022,
                        "camera": "cam-2",
                        "split_2017": pd.NA,
                        "split_2018": pd.NA,
                        "split_2019": pd.NA,
                        "split_2020": pd.NA,
                        "split_2021": "val",
                        "split_2022": "test",
                        "split_ood": "test",
                        "split_iid": "test",
                    },
                ]
            )
            metadata.to_csv(os.path.join(root, "metadata.csv"), index=False)

            for path in metadata["path"]:
                absolute_path = os.path.join(root, path)
                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                Image.new("RGB", (8, 8), color=(120, 80, 40)).save(absolute_path)

            dataset = BrownBearHeads(root)

            self.assertEqual(len(dataset.df), 2)
            self.assertEqual(dataset.df["identity"].nunique(), 2)
            self.assertEqual(set(dataset.df["species"]), {"brown bear"})
            self.assertTrue("image_id" in dataset.df.columns)
            self.assertTrue(dataset.df["path"].str.startswith("BrownBearHeads/").all())
            self.assertNotIn("keypoints", dataset.df.columns)

    def test_create_catalogue_loads_head_keypoint_sidecar(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = pd.DataFrame(
                [
                    {
                        "image_id": 10,
                        "identity": "bear-1",
                        "path": "BrownBearHeads/2017/bear-1/image_1.jpg",
                        "split_iid": "train",
                    },
                    {
                        "image_id": 20,
                        "identity": "bear-2",
                        "path": "BrownBearHeads/2022/bear-2/image_2.jpg",
                        "split_iid": "test",
                    },
                ]
            )
            metadata.to_csv(os.path.join(root, "metadata.csv"), index=False)
            pd.DataFrame(
                [
                    {
                        "image_id": 10,
                        "keypoint_00_x": 1,
                        "keypoint_00_y": 2,
                        "keypoint_00_score": 0.9,
                        "keypoint_01_x": 3,
                        "keypoint_01_y": 4,
                        "keypoint_01_score": 0.8,
                        "min_keypoint_score": 0.8,
                        "mean_keypoint_score": 0.85,
                        "n_out_of_bounds_keypoints": 0,
                        "path": "BrownBearHeads/2022/bear-2/image_2.jpg",
                    },
                    {
                        "image_id": 20,
                        "keypoint_00_x": 5,
                        "keypoint_00_y": 6,
                        "keypoint_00_score": 0.7,
                        "keypoint_01_x": pd.NA,
                        "keypoint_01_y": pd.NA,
                        "keypoint_01_score": pd.NA,
                        "min_keypoint_score": 0.7,
                        "mean_keypoint_score": 0.7,
                        "n_out_of_bounds_keypoints": 1,
                        "path": "BrownBearHeads/2017/bear-1/image_1.jpg",
                    },
                ]
            ).to_csv(os.path.join(root, "head_keypoints.csv"), index=False)

            for path in metadata["path"]:
                absolute_path = os.path.join(root, path)
                os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
                Image.new("RGB", (8, 8), color=(120, 80, 40)).save(absolute_path)

            dataset = BrownBearHeads(root, load_keypoints=True)

            self.assertEqual(dataset.df.loc[0, "keypoints"][:2], [5.0, 6.0])
            self.assertTrue(np.isnan(dataset.df.loc[0, "keypoints"][2]))
            self.assertEqual(dataset.df.loc[0, "n_out_of_bounds_keypoints"], 1)
            self.assertEqual(dataset.df.loc[1, "keypoints"], [1.0, 2.0, 3.0, 4.0])
            self.assertEqual(dataset.df.loc[1, "keypoint_scores"], [0.9, 0.8])
            self.assertNotIn("keypoint_00_x", dataset.df.columns)

    def test_build_clean_metadata_attaches_keypoints(self):
        with tempfile.TemporaryDirectory() as root:
            split_dir = os.path.join(root, "data", "reid_annotations", "test_on_2018")
            os.makedirs(split_dir, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "image": "2018_heads/images/img_1.jpg",
                        "id": "Fisher",
                        "year": 2018,
                        "timestamp": "2018-06-18 14:59:00",
                        "camera": "Canon EOS 7D Mark II",
                        "raw_image": "2018/2018_raw/Fisher/img_1.jpg",
                        "width": 200,
                        "height": 100,
                        "width_raw": 400,
                        "height_raw": 300,
                        "body_image": "2018/2018_bodies/images/img_1.jpg",
                    }
                ]
            ).to_csv(os.path.join(split_dir, "train_iid.csv"), index=False)

            coco_dir = os.path.join(root, "data", "preprocessing_annotations", "coco_bearface_new", "annotations")
            os.makedirs(coco_dir, exist_ok=True)
            pd.Series(
                {
                    "images": [{"id": 1, "height": 100, "width": 200, "file_name": "img_1.jpg"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "keypoints": [10, 20, 2, 30, 40, 2, 0, 0, 0, 50, 60, 2, 70, 80, 2],
                        }
                    ],
                    "categories": [{"id": 1, "name": "bear"}],
                }
            ).to_json(os.path.join(coco_dir, "train.json"))

            superface_dir = os.path.join(root, "data", "preprocessing_annotations", "superface_data", "annotations")
            os.makedirs(superface_dir, exist_ok=True)
            pd.Series(
                {
                    "images": [{"id": 1, "height": 100, "width": 200, "file_name": "/tmp/bear_img_1_1.jpg"}],
                    "annotations": [
                        {
                            "id": 1,
                            "image_id": 1,
                            "keypoints": [1, 2, 2] * 13,
                        }
                    ],
                    "categories": [{"id": 1, "name": "bear"}],
                }
            ).to_json(os.path.join(superface_dir, "train.json"))

            output_csv = os.path.join(root, "clean_metadata.csv")
            metadata = prepare_metadata(root, output_csv, include_keypoints=True)

            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata.loc[0, "identity"], "Fisher")
            self.assertEqual(metadata.loc[0, "path"], "2018_heads/images/img_1.jpg")
            self.assertEqual(metadata.loc[0, "path_original"], "2018_heads/images/img_1.jpg")
            self.assertEqual(metadata.loc[0, "keypoints"][:4], [10.0, 20.0, 30.0, 40.0])
            self.assertTrue(np.isnan(metadata.loc[0, "keypoints"][4]))
            self.assertNotIn("path_body_original", metadata.columns)
            self.assertNotIn("keypoints_5", metadata.columns)

    def test_resize_and_restructure_writes_final_paths(self):
        with tempfile.TemporaryDirectory() as root:
            source_root = os.path.join(root, "Public_release")
            prepared_root = os.path.join(root, "BrownBearHeads")
            image_path = os.path.join(source_root, "2018_heads", "images", "img_1.jpg")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.new("RGB", (120, 80), color=(120, 80, 40)).save(image_path)

            metadata_path = os.path.join(root, "clean_metadata.csv")
            pd.DataFrame(
                [
                    {
                        "image_id": 0,
                        "identity": "Fisher",
                        "path": "2018_heads/images/img_1.jpg",
                        "path_original": "2018_heads/images/img_1.jpg",
                        "date": "2018-06-18 14:59:00",
                        "year": 2018,
                        "camera": "Canon EOS 7D Mark II",
                        "split_iid": "train",
                        "keypoints": "[1, 2]",
                    }
                ]
            ).to_csv(metadata_path, index=False)

            prepared = resize_and_restructure(
                source_root=source_root,
                metadata_path=metadata_path,
                prepared_root=prepared_root,
                max_side=60,
                workers=1,
            )

            self.assertEqual(prepared.loc[0, "path"], "BrownBearHeads/2018/Fisher/img_1.jpg")
            self.assertNotIn("path_original", prepared.columns)
            self.assertNotIn("keypoints", prepared.columns)
            self.assertEqual(prepared.loc[0, "width"], 60)
            self.assertEqual(prepared.loc[0, "height"], 40)
            self.assertTrue(os.path.exists(os.path.join(prepared_root, prepared.loc[0, "path"])))

    def test_predict_keypoint_helpers(self):
        with tempfile.TemporaryDirectory() as root:
            checkpoint = os.path.join(
                root,
                "checkpoints",
                "preprocessing_ckpts",
                "pose",
                "hrnet_w48_balanced_n13_refined.pth",
            )
            os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
            with open(checkpoint, "w"):
                pass

            self.assertEqual(resolve_checkpoint(root, None), checkpoint)

            image_path = os.path.join(root, "data", "2018_heads", "images", "img_1.jpg")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            Image.new("RGB", (8, 8), color=(120, 80, 40)).save(image_path)

            self.assertEqual(resolve_image_path(root, "2018_heads/images/img_1.jpg"), image_path)

            prediction = np.asarray([[1, 2], [3, 4]], dtype=float)
            confidence = np.asarray([[0.9], [0.1]], dtype=float)
            flattened = flatten_prediction(prediction, confidence, confidence_threshold=0.5)

            self.assertEqual(flattened[:2], [1.0, 2.0])
            self.assertTrue(np.isnan(flattened[2]))
            self.assertTrue(np.isnan(flattened[3]))
            self.assertEqual(flatten_scores(confidence), [0.9, 0.1])
            self.assertEqual(keypoints_out_of_bounds(prediction, (4, 4, 3)).tolist(), [False, True])

    def test_write_head_keypoints_sidecar_uses_metadata_path(self):
        with tempfile.TemporaryDirectory() as root:
            output_csv = os.path.join(root, "head_keypoints.csv")
            metadata = pd.DataFrame(
                [
                    {
                        "path": "BrownBearHeads/2018/Fisher/img_1.jpg",
                        "keypoints": [1.2, 2.7, np.nan, np.nan],
                        "keypoint_scores": [0.9, np.nan],
                        "min_keypoint_score": 0.9,
                        "mean_keypoint_score": 0.9,
                        "n_out_of_bounds_keypoints": 0,
                    }
                ]
            )

            sidecar = write_head_keypoints_sidecar(metadata, output_csv)

            self.assertEqual(sidecar.loc[0, "path"], "BrownBearHeads/2018/Fisher/img_1.jpg")
            self.assertEqual(sidecar.loc[0, "keypoint_00_x"], 1)
            self.assertEqual(sidecar.loc[0, "keypoint_00_y"], 3)
            self.assertTrue(pd.isna(sidecar.loc[0, "keypoint_01_x"]))
            loaded = pd.read_csv(output_csv)
            self.assertEqual(loaded.loc[0, "path"], "BrownBearHeads/2018/Fisher/img_1.jpg")

    def test_import_pose_code_skips_training_dependencies(self):
        with tempfile.TemporaryDirectory() as root:
            backbones = os.path.join(root, "PoseGuidedReID", "project", "models", "backbones")
            os.makedirs(backbones, exist_ok=True)
            models_init = os.path.join(root, "PoseGuidedReID", "project", "models", "__init__.py")
            os.makedirs(os.path.dirname(models_init), exist_ok=True)

            with open(models_init, "w") as file:
                file.write("import wandb\n")
            with open(os.path.join(backbones, "hrnet.py"), "w") as file:
                file.write("class HRNet:\n    pass\n")
            with open(os.path.join(backbones, "pose_utils.py"), "w") as file:
                file.write("def get_affine_transform():\n    return 'affine'\n")
                file.write("def get_final_preds():\n    return 'preds'\n")
            with open(os.path.join(backbones, "pose_net.py"), "w") as file:
                file.write("from .hrnet import HRNet\n")
                file.write("from .pose_utils import get_affine_transform, get_final_preds\n")
                file.write("def _box2cs():\n    return 'box'\n")
                file.write("class SimpleHRNet:\n    pass\n")

            simple_hrnet, box2cs, get_affine_transform, get_final_preds = import_pose_code(root)

            self.assertEqual(simple_hrnet.__name__, "SimpleHRNet")
            self.assertEqual(box2cs(), "box")
            self.assertEqual(get_affine_transform(), "affine")
            self.assertEqual(get_final_preds(), "preds")


if __name__ == "__main__":
    unittest.main()
