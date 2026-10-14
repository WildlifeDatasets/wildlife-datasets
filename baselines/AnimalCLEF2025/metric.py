import numpy as np
import pandas as pd
import pandas.api.types

from wildlife_datasets.metrics import BAKS, BAUS


def score(solution: pd.DataFrame, submission: pd.DataFrame) -> dict:

    submission = submission.reset_index(drop=True)
    solution = solution.reset_index(drop=True)

    # Checks for correct input format
    if "image_id" not in submission.columns:
        return {}
    if "identity" not in submission.columns:
        return {}
    if len(submission) != len(solution):
        return {}
    if not np.array_equal(submission["image_id"], solution["image_id"]):
        return {}
    if not pandas.api.types.is_string_dtype(submission["identity"]):
        return {}
    prefixes = ("SeaTurtleID2022_", "SalamanderID2025_", "LynxID2025_")
    is_valid = submission["identity"].str.startswith(prefixes) | (submission["identity"] == "new_individual")
    if not is_valid.all():
        return {}

    # Extract the data
    results = {}
    unknown_identities = solution[solution["new_identity"]]["identity"].unique()
    for name, solution_dataset in solution.groupby("dataset"):
        predictions = submission.loc[solution_dataset.index, "identity"].to_numpy()
        labels = solution_dataset["identity"].to_numpy()

        # Compute the balances accuracies on known and unknown samples
        acc_known = BAKS(labels, predictions, unknown_identities)
        acc_unknown = BAUS(labels, predictions, unknown_identities, "new_individual")

        # Compute precision
        unknown_true = np.isin(labels, unknown_identities)
        unknown_pred = predictions == "new_individual"
        if sum(unknown_pred) == 0:
            precision = 0
        else:
            precision = sum(unknown_true & unknown_pred) / sum(unknown_pred)

        # Fix possible nans
        if np.isnan(acc_known):
            acc_known = 0
        if np.isnan(acc_unknown):
            acc_unknown = 0

        # Save to dataframe
        results[name] = {
            "BAKS": acc_known,
            "BAUS": acc_unknown,
            "precision": precision,
            "normalized": np.sqrt(acc_known * acc_unknown),
        }
    results = pd.DataFrame(results).T

    results2 = {}
    cols = ["BAKS", "BAUS", "precision", "normalized"]
    for col in cols:
        for name in solution["dataset"].unique():
            results2[f"{name}_{col}"] = results.loc[name, col]
        results2[col] = results[col].mean()

    return results2
