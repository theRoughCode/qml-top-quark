import pandas as pd


def prepare_data(path, variables, num_samples=None, background_to_signal_ratio=0.5):
    """Process dataset and extract columns specified by `variables`.

    Args:
        path (str): Path to dataset parquet file.
        variables (List[str]): List of variable names to filtere by.
        num_samples (int, optional): Number of samples to load in. If None, use whole dataset. Defaults to None.
        background_to_signal_ratio (float, optional): Ratio of background vs signal data to include.
            Only used if `num_samples` is not None. Defaults to 0.5.

    Returns:
        [type]: [description]
    """
    df = pd.read_parquet(path)
    df = df[variables + [b"type"]]

    if num_samples is not None:
        num_background = int(background_to_signal_ratio * num_samples)
        num_signals = num_samples - num_background
        background_samples = df.loc[df[b"type"] == 0.0].sample(num_background)
        signal_samples = df.loc[df[b"type"] == 1.0].sample(num_signals)
        df = pd.concat([background_samples, signal_samples])

    # Shuffle dataset
    df = df.sample(frac=1)

    x = df[variables].to_numpy()
    y = df[b"type"].to_numpy()

    return x, y


def load_data(filepath, ranked_vars_filepath, num_vars=9, num_samples=None):
    """Load data from `filepath`.

    Args:
        filepath (str): Path to dataset.
        ranked_vars_filepath (str): Path to comma-separated file of ranked variables from highest to lowest.
        num_vars (int, optional): Use only the top N variables. Defaults to 9.
        num_samples (int, optional): Number of samples to load in. If None, use whole dataset. Defaults to None.

    Returns:
        ds_x: features for dataset
        ds_y: labels for dataset
    """
    with open(ranked_vars_filepath, "rb") as f:
        ranked_vars = f.readline()
        top_n_vars = ranked_vars.split(b',')[:num_vars]

    ds_x, ds_y = prepare_data(filepath, top_n_vars,
                              num_samples, background_to_signal_ratio=0.5)

    return ds_x, ds_y
