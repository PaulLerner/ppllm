from jsonargparse import CLI
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union
from jsonargparse import CLI
import os
import warnings
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import load_texts, unsort


@dataclass
class ModelKwargs:
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = None
    #device_map: str = "auto"
    config: Optional[Union[str, os.PathLike]] = None
    cache_dir: Optional[Union[str, os.PathLike]] = None
    ignore_mismatched_sizes: bool = False
    force_download: bool = False
    local_files_only: bool = False
    token: Optional[Union[str, bool]] = None
    revision: str = "main"
    use_safetensors: bool = None
    resume_download: bool = False
    output_loading_info: bool = False
    dtype: str = "float16"
    load_in_8bit: bool = False
    load_in_4bit: bool = False
   # use_flash_attention_2: bool = False
    trust_remote_code: bool = True


@dataclass
class LoaderKwargs:
    batch_size: Optional[int] = 64
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""


@dataclass
class TokenizerKwargs:
    return_tensors: str = 'pt'
    padding: Union[bool, str] = 'longest'
    truncation: bool = False
    return_overflowing_tokens: bool = False
    max_length: int = None


@torch.no_grad()
def compute_nll(loader, indices, model, tokenizer, tokenizer_kwargs, window: int = None):
    stride = window // 2
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    total_losses, total_chars, total_tokens = [], [], []
    all_losses, all_indices = [], []
    i = 0
    for batch in tqdm(loader, total=len(loader.dataset)//loader.batch_size):
        inputs = tokenizer(batch, **tokenizer_kwargs)
        batch_size, seq_len = inputs["input_ids"].shape
        for k, v in inputs.items():
            inputs[k] = v.to(model.device)
        if window is None:
            logits = model(**inputs, return_dict=True).logits
            labels = inputs['input_ids']        
            logits = logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
            labels = labels[:, 1:].contiguous().view(-1)
            losses = loss_fct(logits, labels).view(batch_size, seq_len-1)
            total_losses.append(losses.sum(1).cpu())
            all_losses.append(losses.reshape(-1).cpu())
            all_indices.append(indices[i: i+batch_size].repeat_interleave(seq_len-1))
        else:
            for j in range(0, seq_len-stride, stride):
                input_ids = inputs["input_ids"][:, j: j+window]
                logits = model(input_ids, return_dict=True).logits
                if j > 0:
                    logits = logits[:, stride:-1].contiguous().view(-1, model.config.vocab_size)
                    labels = input_ids[:, stride+1:].contiguous().view(-1)
                    losses = loss_fct(logits, labels).view(batch_size, stride-1)
                else:
                    logits = logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
                    labels = input_ids[:, 1:].contiguous().view(-1)
                    losses = loss_fct(logits, labels).view(batch_size, window-1)
                total_losses.append(losses.sum(1).cpu())
                all_losses.append(losses.reshape(-1).cpu())
                all_indices.append(indices[i: i+batch_size].repeat_interleave(window-1))
        i += len(batch)
        tokenized_texts = tokenizer(batch, add_special_tokens=False)
        # FIXME: if there's no BOS, we should not count the first token
        for text, tokens in zip(batch, tokenized_texts["input_ids"]):
            total_chars.append(len(text))
            total_tokens.append(len(tokens))
    all_losses, all_indices = torch.cat(all_losses), torch.cat(all_indices)
    total_losses = unsort(torch.cat(total_losses).to(torch.float32), indices)
    total_chars, total_tokens = unsort(torch.tensor(total_chars), indices), unsort(torch.tensor(total_tokens), indices)
    outputs = dict(
        total_losses=total_losses, 
        total_chars=total_chars, 
        total_tokens=total_tokens, 
        all_losses=all_losses, 
        all_indices=all_indices
    )
    return outputs


def main(output_dir: Path, data_path: Path, model_kwargs: ModelKwargs, window: int = None, input_key: str = "text", split: str = "test",
         tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), loader_kwargs: LoaderKwargs = LoaderKwargs()):
    output_dir.mkdir(exist_ok=True, parents=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_kwargs.pretrained_model_name_or_path, 
        add_prefix_space=False, 
        add_eos_token=False, 
        trust_remote_code=model_kwargs.trust_remote_code
    )
    if tokenizer.pad_token is None:
        warnings.warn(f"{tokenizer.pad_token=}, setting to {tokenizer.eos_token=}")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs)).cuda()
    texts = load_texts(data_path, input_key=input_key, split=split)
    indices = torch.tensor([len(text) for text in texts]).argsort(descending=True)
    loader = DataLoader([texts[i] for i in indices], **asdict(loader_kwargs), shuffle=False, collate_fn=None)
    outputs = compute_nll(loader, indices, model, tokenizer, asdict(tokenizer_kwargs), window=window)
    # surprisal is expressed in bits
    total_surprisal = outputs["total_losses"].sum()/torch.log(torch.tensor(2))
    metrics = {
        "ppl": 2**(total_surprisal/outputs["total_tokens"].sum()).item(),
        "bpc": (total_surprisal/outputs["total_chars"].sum()).item(),
        "surprisal": total_surprisal.item()
    }

    print(metrics)
    with open(output_dir/"metrics.json", "wt") as file:
        json.dump(metrics, file)
    for k, v in outputs.items():
        torch.save(v, output_dir/f"{k}.bin")


def cli():
    CLI(main)