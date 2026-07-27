DEFAULT_DATA_ROOT = "/data/wildlife_datasets/data"


def pytest_addoption(parser):
    parser.addoption(
        "--data-root",
        action="store",
        default=DEFAULT_DATA_ROOT,
        help="Root folder containing locally downloaded real datasets (used by test_real_datasets.py).",
    )
