import numpy as np
import pandas as pd


def display_statistics(
    df: pd.DataFrame,
    unknown_name: str = "",
    col_label: str = "identity",
) -> None:
    """Prints statistics about the dataframe.

    Args:
        df (pd.DataFrame): A full dataframe of the data.
        unknown_name (str, optional): Name of the unknown class.
        col_label (str, optional): Column name containing individual animal names (labels).
    """

    # Remove the unknown identities
    df_red = df.loc[df[col_label] != unknown_name, col_label]
    df_red.value_counts().reset_index(drop=True).plot(xlabel="identities", ylabel="counts")

    # Compute the total number of identities
    if unknown_name in list(df[col_label].unique()):
        n_identity = len(df.identity.unique()) - 1
    else:
        n_identity = len(df.identity.unique())
    n_one = len(df.groupby(col_label).filter(lambda x: len(x) == 1))
    n_unidentified = sum(df[col_label] == unknown_name)

    # Print general statistics
    print(f"Number of identitites            {n_identity}")
    print(f"Number of all animals            {len(df)}")
    print(f"Number of animals with one image {n_one}")
    print(f"Number of unidentified animals   {n_unidentified}")

    # Print statistics about video if present
    if "video" in df.columns:
        print(f"Number of videos                 {len(df[[col_label, 'video']].drop_duplicates())}")

    # Print statistics about time span if present
    if "date" in df.columns:
        span_years = compute_span(df, col_label=col_label) / (60 * 60 * 24 * 365.25)
        if span_years > 1:
            print("Images span                      %1.1f years" % (span_years))
        elif span_years / 12 > 1:
            print("Images span                      %1.1f months" % (span_years * 12))
        else:
            print("Images span                      %1.0f days" % (span_years * 365.25))


def compute_span(df: pd.DataFrame, col_label: str = "identity") -> float:
    """Compute the time span of the dataset.

    The span is defined as the latest time minus the earliest time of image taken.
    The times are computed separately for each individual.

    Args:
        df (pd.DataFrame): A full dataframe of the data.
        col_label (str, optional): Column name containing individual animal names (labels).

    Returns:
        The span of the dataset in seconds.
    """

    # Convert the dates into timedelta
    df = df.loc[~df["date"].isnull()]
    dates = pd.to_datetime(df["date"]).to_numpy()

    # Find the maximal span across individuals
    identities = df[col_label].unique()
    span_seconds = -np.inf
    for identity in identities:
        idx = df[col_label] == identity
        span_seconds = np.maximum(span_seconds, (max(dates[idx]) - min(dates[idx])) / np.timedelta64(1, "s"))
    return span_seconds
