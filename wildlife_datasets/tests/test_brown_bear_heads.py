import os
import tempfile
import unittest

import pandas as pd
from PIL import Image

from wildlife_datasets.datasets import BrownBearHeads


class TestBrownBearHeads(unittest.TestCase):
    def test_create_catalogue_from_prepared_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            metadata = pd.DataFrame(
                [
                    {
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


if __name__ == "__main__":
    unittest.main()
