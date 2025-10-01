from pathlib import Path
import pandas as pd
from datasets import load_from_disk


def load_texts(path: Path, input_key: str = "text", split: str = "test"):
    # TODO from txt
    dataset = load_dataset(path)
    subset = get_split(dataset, split)
    return list(subset[input_key])


def load_dataset(data_path):
    if data_path.suffix == ".csv":
        dataset = pd.read_csv(data_path)
    else:
        dataset = load_from_disk(data_path)
    return dataset


def get_split(dataset, split):
    if isinstance(dataset, pd.DataFrame):
        if split is None:
            return dataset
        elif isinstance(split, str):
            return dataset[dataset.split==split]
        else:
            return dataset[dataset.split.isin(split)]
    else:
        if split is None:
            return dataset
        return dataset[split]
        