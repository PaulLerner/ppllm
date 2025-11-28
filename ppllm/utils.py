from pathlib import Path
import pandas as pd
import datasets
import torch


def fix_tokenizer(tokenizer):
    # ensure right padding so we don't need attention mask
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    # FIXME: in this case the surprisal of EOS will not be computed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


@torch.no_grad()
def find_batch_size(texts, model, tokenizer, tokenizer_kwargs, device, window: int = None):
    if str(device) == "cpu":
        raise ValueError(f"{device} not supported, set the batch size manually")
    
    batch_size = 1
    ok_batch_size = None
    while ok_batch_size is None or ok_batch_size < len(texts):
        input_ids = tokenizer(texts[:batch_size], **tokenizer_kwargs)["input_ids"].to(device)
        if window is not None:
            input_ids = input_ids[:, :window]
        try:
            _ = model(input_ids, return_dict=True).logits
        except Exception as e:
            if ok_batch_size is None:
                raise ValueError(f"Got Exception {e=} (likely OOM) with {batch_size=}, try using a smaller {window=}")
            else:
                break
        else:
            ok_batch_size = batch_size
            batch_size *= 2
    print(f"Found {ok_batch_size=}")
    return ok_batch_size


def unsort(sorted_values, indices):
    unsorted = torch.empty_like(sorted_values)
    unsorted[indices] = sorted_values
    return unsorted


def load_dataset(data_path: Path, split: str = "test"):
    if data_path.suffix == ".csv":
        dataset = pd.read_csv(data_path)
        if split is None:
            subset = dataset
        elif isinstance(split, str):
            subset = dataset[dataset.split==split]
        else:
            subset = dataset[dataset.split.isin(split)]
        subset = subset.to_dict('records')
    elif (data_path/"dataset_info.json").exists() or (data_path/"dataset_dict.json").exists():
        dataset = datasets.load_from_disk(data_path)
        subset = list(dataset[split])
    else:
        dataset = datasets.load_dataset(data_path)
        subset = list(dataset[split])
    return subset
