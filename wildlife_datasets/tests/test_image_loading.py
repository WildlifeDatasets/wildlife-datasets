import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import pycocotools.mask as mask_coco
from PIL import Image

from wildlife_datasets.datasets import utils as ds_utils

from .utils import create_dataset

# A small, fully-known image: red (255, 0, 0) everywhere except a blue
# (0, 0, 255) rectangle at BOX = (x, y, w, h). Every test below can therefore
# assert exact pixel values, not just output sizes.
IMG_SIZE = (8, 8)  # (width, height)
BOX = (2, 2, 4, 4)  # x, y, w, h
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


def make_box_mask() -> np.ndarray:
    """Boolean mask matching BOX. Shape is (height, width), matching image array order."""
    w_img, h_img = IMG_SIZE
    mask = np.zeros((h_img, w_img), dtype=bool)
    x, y, w, h = BOX
    mask[y : y + h, x : x + w] = True
    return mask


def make_polygon() -> list:
    """BOX as a COCO-style polygon: [x0, y0, x1, y1, x2, y2, x3, y3]."""
    x, y, w, h = BOX
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def mask_to_uncompressed_rle_counts(mask: np.ndarray) -> list:
    """Encodes a boolean mask as COCO uncompressed-RLE `counts` (column-major run-lengths).

    Written from scratch (not derived via pycocotools) so it exercises the
    "list counts" branch of `_prepare_segmentation` independently of whatever
    `mask_coco.encode` happens to produce.
    """

    flat = mask.flatten(order="F")
    counts = []
    current = False
    run_length = 0
    for val in flat:
        if bool(val) == current:
            run_length += 1
        else:
            counts.append(run_length)
            current = bool(val)
            run_length = 1
    counts.append(run_length)
    return counts


def make_test_image() -> Image.Image:
    img = Image.new("RGB", IMG_SIZE, color=RED)
    x, y, w, h = BOX
    for i in range(x, x + w):
        for j in range(y, y + h):
            img.putpixel((i, j), BLUE)
    return img


def make_dataframe(root, image_name="img.png", bbox=None, segmentation=None):
    make_test_image().save(os.path.join(root, image_name))
    row = {"image_id": 0, "identity": "id0", "path": image_name}
    if bbox is not None:
        row["bbox"] = bbox
    if segmentation is not None:
        row["segmentation"] = segmentation
    return pd.DataFrame([row])


class ImageLoadingTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, bbox=None, segmentation=None, **kwargs):
        df = make_dataframe(self.root, bbox=bbox, segmentation=segmentation)
        return create_dataset(df, root=self.root, check_files=True, **kwargs)


class TestImgLoadModes(ImageLoadingTestCase):
    def test_full_returns_image_unchanged(self):
        dataset = self.build(bbox=list(BOX), segmentation=make_polygon(), img_load="full")
        img = dataset[0]
        self.assertEqual(img.size, IMG_SIZE)
        self.assertEqual(img.getpixel((0, 0)), RED)
        self.assertEqual(img.getpixel((3, 3)), BLUE)

    def test_bbox_crops_to_box(self):
        dataset = self.build(bbox=list(BOX), img_load="bbox")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLUE))

    def test_full_mask_zeroes_background_keeps_object(self):
        dataset = self.build(segmentation=make_polygon(), img_load="full_mask")
        img = dataset[0]
        self.assertEqual(img.size, IMG_SIZE)
        self.assertEqual(img.getpixel((0, 0)), BLACK)
        self.assertEqual(img.getpixel((3, 3)), BLUE)

    def test_full_hide_zeroes_object_keeps_background(self):
        dataset = self.build(segmentation=make_polygon(), img_load="full_hide")
        img = dataset[0]
        self.assertEqual(img.size, IMG_SIZE)
        self.assertEqual(img.getpixel((0, 0)), RED)
        self.assertEqual(img.getpixel((3, 3)), BLACK)

    def test_bbox_mask_crops_to_object_bbox_and_keeps_it(self):
        dataset = self.build(segmentation=make_polygon(), img_load="bbox_mask")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLUE))

    def test_bbox_hide_of_object_filling_its_own_bbox_is_all_black(self):
        dataset = self.build(segmentation=make_polygon(), img_load="bbox_hide")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLACK))

    def test_auto_picks_bbox_mask_when_segmentation_present(self):
        dataset = self.build(bbox=list(BOX), segmentation=make_polygon(), img_load="auto")
        self.assertEqual(dataset.img_load, "bbox_mask")

    def test_auto_picks_bbox_when_only_bbox_present(self):
        dataset = self.build(bbox=list(BOX), img_load="auto")
        self.assertEqual(dataset.img_load, "bbox")

    def test_auto_picks_full_when_neither_present(self):
        dataset = self.build(img_load="auto")
        self.assertEqual(dataset.img_load, "full")

    def test_invalid_img_load_raises(self):
        dataset = self.build(img_load="not_a_real_mode")
        with self.assertRaises(ValueError):
            dataset[0]

    def test_bbox_mode_without_bbox_column_raises(self):
        dataset = self.build(img_load="bbox")
        with self.assertRaises(ValueError):
            dataset[0]

    def test_bbox_mask_mode_without_segmentation_column_raises(self):
        dataset = self.build(bbox=list(BOX), img_load="bbox_mask")
        with self.assertRaises(ValueError):
            dataset[0]


class TestSegmentationFormats(ImageLoadingTestCase):
    def test_polygon_list(self):
        dataset = self.build(segmentation=make_polygon(), img_load="bbox_mask")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLUE))

    def test_uncompressed_rle_list_counts(self):
        mask = make_box_mask()
        rle = {"counts": mask_to_uncompressed_rle_counts(mask), "size": [IMG_SIZE[1], IMG_SIZE[0]]}
        dataset = self.build(segmentation=rle, img_load="bbox_mask")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLUE))

    def test_compressed_rle_string_counts(self):
        mask = make_box_mask()
        fortran_mask = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_coco.encode(fortran_mask)
        if isinstance(rle["counts"], bytes):
            rle["counts"] = rle["counts"].decode("ascii")
        dataset = self.build(segmentation=rle, img_load="bbox_mask")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLUE))

    def test_segmentation_as_path_to_mask_image(self):
        mask = make_box_mask().astype(np.uint8)
        mask_path = "mask.png"
        Image.fromarray(mask, mode="L").save(os.path.join(self.root, mask_path))
        dataset = self.build(segmentation=mask_path, img_load="bbox_mask")
        img = dataset[0]
        self.assertEqual(img.size, (BOX[2], BOX[3]))
        self.assertTrue(np.all(np.asarray(img) == BLUE))

    def test_null_segmentation_is_passthrough(self):
        dataset = self.build(segmentation=np.nan, img_load="full_mask")
        img = dataset[0]
        self.assertEqual(img.size, IMG_SIZE)
        self.assertEqual(img.getpixel((0, 0)), RED)
        self.assertEqual(img.getpixel((3, 3)), BLUE)

    def test_unrecognized_segmentation_type_raises(self):
        dataset = self.build(segmentation=12345, img_load="full_mask")
        with self.assertRaises(Exception):
            dataset[0]


class TestCropBlackWhite(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def _make_bordered_image(self, border_color, content_color):
        size = (10, 10)  # width, height
        content_box = (3, 2, 6, 4)  # x0, y0, x1, y1 (exclusive), 3 wide x 2 tall
        img = Image.new("RGB", size, color=border_color)
        for x in range(content_box[0], content_box[2]):
            for y in range(content_box[1], content_box[3]):
                img.putpixel((x, y), content_color)
        return img, content_box

    def test_crop_black_trims_to_exact_content_box(self):
        img, content_box = self._make_bordered_image(BLACK, (255, 255, 255))
        cropped = ds_utils.crop_black(img)
        expected_size = (content_box[2] - content_box[0], content_box[3] - content_box[1])
        self.assertEqual(cropped.size, expected_size)

    def test_crop_white_trims_to_exact_content_box(self):
        img, content_box = self._make_bordered_image((255, 255, 255), BLACK)
        cropped = ds_utils.crop_white(img)
        expected_size = (content_box[2] - content_box[0], content_box[3] - content_box[1])
        self.assertEqual(cropped.size, expected_size)

    def test_crop_black_on_blank_image_returns_unchanged(self):
        img = Image.new("RGB", (8, 8), color=BLACK)
        cropped = ds_utils.crop_black(img)
        self.assertEqual(cropped.size, img.size)
        self.assertTrue(np.all(np.asarray(cropped) == np.asarray(img)))

    def test_crop_white_on_blank_image_returns_unchanged(self):
        img = Image.new("RGB", (8, 8), color=(255, 255, 255))
        cropped = ds_utils.crop_white(img)
        self.assertEqual(cropped.size, img.size)
        self.assertTrue(np.all(np.asarray(cropped) == np.asarray(img)))

    def test_crop_black_on_fully_white_image_returns_unchanged(self):
        img = Image.new("RGB", (8, 8), color=(255, 255, 255))
        cropped = ds_utils.crop_black(img)
        self.assertEqual(cropped.size, img.size)

    def test_crop_white_on_fully_black_image_returns_unchanged(self):
        img = Image.new("RGB", (8, 8), color=BLACK)
        cropped = ds_utils.crop_white(img)
        self.assertEqual(cropped.size, img.size)


if __name__ == "__main__":
    unittest.main()
