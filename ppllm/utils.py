from pathlib import Path
import pandas as pd
import datasets
import torch


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def find_batch_size(texts, model, tokenizer, tokenizer_kwargs, device, window: int = None):
    if str(device) == "cpu":
        raise ValueError(f"{device} not supported, set the batch size manually")
    
    batch_size = 1
    ok_batch_size = None
    while batch_size < len(texts):
        input_ids = tokenizer(texts[:batch_size], **tokenizer_kwargs)["input_ids"].to(device)
        if window is not None:
            input_ids = input_ids[:, :window]
        try:
            _ = model(input_ids, return_dict=True).logits
        except Exception as e:
            if ok_batch_size is None:
                raise ValueError(f"Got Exception {e=} (likely OOM) with {batch_size=}, try using a smaller {window=}")
            else:
                print(f"Found {ok_batch_size=}")
                return ok_batch_size
        else:
            ok_batch_size = batch_size
            batch_size *= 2
    return ok_batch_size


def unsort(sorted_values, indices):
    unsorted = torch.empty_like(sorted_values)
    unsorted[indices] = sorted_values
    return unsorted


def load_texts(path: Path, input_key: str = "text", split: str = "test"):
    # TODO from txt
    dataset = load_dataset(path)
    subset = get_split(dataset, split)
    return list(subset[input_key])


def load_dataset(data_path: Path):
    if data_path.suffix == ".csv":
        dataset = pd.read_csv(data_path)
    elif (data_path/"dataset_info.json").exists() or (data_path/"dataset_dict.json").exists():
        dataset = datasets.load_from_disk(data_path)
    else:
        dataset = datasets.load_dataset(data_path)
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
        