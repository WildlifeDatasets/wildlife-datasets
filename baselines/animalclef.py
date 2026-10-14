from collections.abc import Callable

import pandas as pd

from wildlife_datasets.datasets.utils import find_images


def extract_participant_info(folder: str, participants_replace: dict = {}) -> tuple[str, int]:
    folder_split = folder.split(" ")
    participant_rank = int(folder_split[0])
    participant_name = " ".join(folder_split[1:])
    if folder in participants_replace:
        participant_name = participants_replace[folder]
    elif participant_name in participants_replace:
        participant_name = participants_replace[participant_name]
    return participant_name, participant_rank


def load_submissions(root: str, participants_replace: dict = {}) -> pd.DataFrame:
    submissions = find_images(root, img_extensions=(".csv",)).reset_index(drop=True)
    mask = submissions["path"].str.contains("/")
    if mask.any():
        raise ValueError(f"There are multiple folders in the submission: {submissions.loc[mask, 'path'].unique()}")

    info = submissions["path"].apply(lambda x: extract_participant_info(x, participants_replace))
    submissions["participant"] = info.str[0]
    submissions["participant_rank"] = info.str[1].astype(int)
    submissions["file_name"] = root + "/" + submissions["path"] + "/" + submissions["file"]
    return submissions


def evaluate_submissions(
    submissions: pd.DataFrame,
    solution: pd.DataFrame,
    score_fn: Callable[[pd.DataFrame, pd.DataFrame], dict],
    idx_public: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx_private = ~idx_public
    scores_public = []
    scores_private = []
    for file_name in submissions["file_name"]:
        try:
            submission = pd.read_csv(file_name)
        except Exception:
            print(f"Loading failed: {file_name}")
            submission = pd.DataFrame()

        if solution.index.equals(submission.index):
            score_public = score_fn(solution[idx_public], submission[idx_public])
            score_private = score_fn(solution[idx_private], submission[idx_private])
        else:
            score_public = {}
            score_private = {}
        scores_public.append(score_public)
        scores_private.append(score_private)

    public = pd.DataFrame(scores_public, index=submissions.index)
    private = pd.DataFrame(scores_private, index=submissions.index)
    return public, private


def select_best(submissions: pd.DataFrame, public: pd.DataFrame, private: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = {}
    for participant, df_participant in submissions.groupby("participant"):
        scores_participant = public.loc[df_participant.index, metric].dropna()
        if scores_participant.empty:
            continue
        i_submission = scores_participant.idxmax()
        row = private.loc[i_submission].to_dict()
        row["rank"] = int(df_participant["participant_rank"].iloc[0])
        row["submission"] = int(i_submission)
        rows[participant] = row
    return pd.DataFrame(rows).T.astype({"rank": int, "submission": int}).sort_values("rank")


def get_submission(results: pd.DataFrame, submissions: pd.DataFrame, i: int) -> pd.DataFrame:
    i_submission = results["submission"].iloc[i]
    file_name = submissions.loc[i_submission, "file_name"]
    return pd.read_csv(file_name)
