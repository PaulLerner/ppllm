from jsonargparse import CLI
import json
from dataclasses import asdict
from pathlib import Path
from jsonargparse import CLI

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from ppllm.utils import load_texts, get_device, find_batch_size
from ppllm.ppl import ModelKwargs,LoaderKwargs,TokenizerKwargs,compute_nll,compute_metrics,count_tokens_chars


def main(output_dir: Path, data_path: Path, model_kwargs: ModelKwargs, window: int = None, input_key: str = "text", split: str = "test",
         tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), loader_kwargs: LoaderKwargs = LoaderKwargs()):
    """Compute the PPL and Surprisal of an LLM"""
    tokenizer_kwargs = asdict(tokenizer_kwargs)
    assert window is None or window%2 == 0, f"window must be dividible by 2, got {window}"
    output_dir.mkdir(exist_ok=True, parents=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_kwargs.pretrained_model_name_or_path, 
        add_prefix_space=False, 
        # FIXME option for EOS
        add_eos_token=False, 
        trust_remote_code=model_kwargs.trust_remote_code
    )
    # ensure right padding so we don't need attention mask
    if tokenizer.padding_side != "right":
        tokenizer.padding_side = "right"
    # FIXME: in this case the surprisal of EOS will not be computed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = get_device()
    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs)).to(device)
    texts = load_texts(data_path, input_key=input_key, split=split)
    total_chars, total_tokens = count_tokens_chars(texts, tokenizer)
    indices = torch.randperm(len(texts))
    sorted_texts = [texts[i] for i in indices]
    if loader_kwargs.batch_size is None:
        loader_kwargs.batch_size = find_batch_size(sorted_texts, model, tokenizer, tokenizer_kwargs, device, window=window)
    loader = DataLoader(sorted_texts, **asdict(loader_kwargs), shuffle=False, collate_fn=None)
    outputs = compute_nll(loader, indices, model, tokenizer, tokenizer_kwargs, window=window, device=device)
    outputs.update(dict(total_chars=total_chars, total_tokens=total_tokens))
    metrics = compute_metrics(**{k: outputs[k] for k in ["total_losses", "total_chars", "total_tokens"]})

    print(metrics)
    metrics.update(dict(window=window, batch_size=loader_kwargs.batch_size, software=Path(__file__).stem))
    with open(output_dir/"metrics.json", "wt") as file:
        json.dump(metrics, file)
    for k, v in outputs.items():
        torch.save(v, output_dir/f"{k}.bin")


if __name__ == "__main__":
    CLI(main)