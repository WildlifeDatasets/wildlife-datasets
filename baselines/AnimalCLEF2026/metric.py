import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_rand_score


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> dict:
    submission = submission.reset_index(drop=True)
    solution = solution.reset_index(drop=True)

    # Checks for correct input format
    if "image_id" not in submission.columns:
        return {}
    if "cluster" not in submission.columns:
        return {}
    if len(submission) != len(solution):
        return {}
    if not np.array_equal(submission["image_id"], solution["image_id"]):
        return {}
    if not pd.api.types.is_string_dtype(submission["cluster"]):
        return {}
    prefixes = (
        "cluster_LynxID2025_",
        "cluster_SalamanderID2025_",
        "cluster_SeaTurtleID2022_",
        "cluster_TexasHornedLizards_",
    )
    if not submission["cluster"].str.startswith(prefixes).all():
        return {}

    # Compute adjusted random index (ARI) for each dataset
    results = {}
    for name, solution_dataset in solution.groupby("dataset"):
        predictions = submission.loc[solution_dataset.index, "cluster"].to_numpy()
        labels = solution_dataset["identity"].to_numpy()
        results[name] = adjusted_rand_score(labels, predictions)

    # Add the mean score
    results["mean"] = np.mean(list(results.values()))

    return results
