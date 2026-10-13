DEFAULT_DATA_ROOT = "/data/wildlife_datasets/data"


def pytest_addoption(parser):
    parser.addoption(
        "--data-root",
        action="store",
        default=DEFAULT_DATA_ROOT,
        help="Root folder containing locally downloaded real datasets (used by test_real_datasets.py).",
    )
    parser.addoption(
        "--no-load-keypoints",
        action="store_true",
        default=False,
        help="Do not pass load_keypoints=True to datasets whose create_catalogue supports it "
        "(used by test_real_datasets.py). Keypoints loading can be slow for some datasets "
        "(e.g. CHIRP scans every keypoints.csv sidecar file), so this speeds up local runs.",
    )
