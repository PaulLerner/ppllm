import warnings
from jsonargparse import CLI
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union
from jsonargparse import CLI
import os
from tqdm import tqdm
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from .utils import load_texts, unsort, get_device, find_batch_size


@dataclass
class ModelKwargs:
    """Arguments for HF's model"""
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
    attn_implementation: str = None
    trust_remote_code: bool = True


@dataclass
class LoaderKwargs:
    """Arguments for torch's DataLoader"""
    batch_size: Optional[int] = None
    num_workers: int = 4
    pin_memory: bool = False
    drop_last: bool = False
    timeout: float = 0
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    pin_memory_device: str = ""


@dataclass
class TokenizerKwargs:
    """Arguments for HF's tokenizer"""
    return_tensors: str = 'pt'
    padding: Union[bool, str] = 'longest'
    truncation: bool = False
    return_overflowing_tokens: bool = False
    max_length: int = None


@torch.no_grad()
def compute_nll(loader, indices, model, tokenizer, tokenizer_kwargs, window: int = None, device: str = None):
    start_time = time.time()
    if device is None:
        device = model.device
    if window is not None:
        stride = window // 2
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
    total_losses = []
    all_losses, all_indices = [], []
    i = 0
    for batch in tqdm(loader, total=len(loader.dataset)//loader.batch_size):
        input_ids = tokenizer(batch, **tokenizer_kwargs)["input_ids"].to(device)
        batch_size, seq_len = input_ids.shape
        if window is None:
            logits = model(input_ids, return_dict=True).logits
            labels = input_ids     
            logits = logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
            labels = labels[:, 1:].contiguous().view(-1)
            losses = loss_fct(logits, labels).view(batch_size, seq_len-1)
            total_losses.append(losses.sum(1).cpu())
            all_losses.append(losses.reshape(-1).cpu())
            all_indices.append(indices[i: i+batch_size].repeat_interleave(seq_len-1))
        else:
            window_losses = []
            # adapted from https://huggingface.co/docs/transformers/perplexity
            for j in range(0, max(seq_len-stride, stride), stride):
                window_ids = input_ids[:, j: j+window]
                logits = model(window_ids, return_dict=True).logits
                if j > 0:
                    logits = logits[:, stride:-1].contiguous().view(-1, model.config.vocab_size)
                    labels = window_ids[:, stride+1:].contiguous().view(-1)
                else:
                    logits = logits[:, :-1].contiguous().view(-1, model.config.vocab_size)
                    labels = window_ids[:, 1:].contiguous().view(-1)
                losses = loss_fct(logits, labels).view(batch_size, -1)
                all_indices.append(indices[i: i+batch_size].repeat_interleave(losses.shape[1]))
                all_losses.append(losses.reshape(-1).cpu())
                window_losses.append(losses.sum(1))
            total_losses.append(torch.stack(window_losses).sum(0).cpu())
        i += len(batch)
    all_losses, all_indices = torch.cat(all_losses), torch.cat(all_indices)
    total_losses = unsort(torch.cat(total_losses).to(torch.float32), indices)
    outputs = dict(
        total_losses=total_losses, 
        all_losses=all_losses, 
        all_indices=all_indices,
        duration=time.time()-start_time
    )
    return outputs


def sample_level(total_losses, total_chars, total_tokens):
    surprisals = total_losses/torch.log(torch.tensor(2))
    metrics = {
        "ppls": 2**(surprisals/total_tokens),
        "bpcs": (surprisals/total_chars),
        "surprisals": surprisals
    }
    return metrics


def compute_metrics(total_losses, total_chars, total_tokens):
    # surprisal is expressed in bits
    total_surprisal = total_losses.sum()/torch.log(torch.tensor(2))
    metrics = {
        "ppl": 2**(total_surprisal/total_tokens.sum()).item(),
        "bpc": (total_surprisal/total_chars.sum()).item(),
        "surprisal": total_surprisal.item()
    }
    return metrics


def count_tokens_chars_discount_bos(texts, tokenizer):
    # if there's no BOS, we should not count the first token
    raise NotImplementedError()


def count_tokens_chars(texts, tokenizer):
    total_chars, total_tokens = [], []
    if tokenizer.bos_token is None:
        warnings.warn(f"{tokenizer.bos_token=} so the first token surprisal is not computed but it's number of characters is counted so BPC is a bit optimistic")
        #return count_tokens_chars_discount_bos(texts, tokenizer)
    all_tokens = tokenizer(texts, add_special_tokens=False)
    for text, tokens in zip(texts, all_tokens["input_ids"]):
        total_chars.append(len(text))
        total_tokens.append(len(tokens))
    total_chars, total_tokens = torch.tensor(total_chars), torch.tensor(total_tokens)
    return total_chars, total_tokens


def compute_ppl(texts, model, tokenizer, tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), 
                loader_kwargs: LoaderKwargs = LoaderKwargs(), window: int = None, device: str = None):
    tokenizer_kwargs = asdict(tokenizer_kwargs)
    if device is None:
        device = model.device
    total_chars, total_tokens = count_tokens_chars(texts, tokenizer)
    indices = total_tokens.argsort(descending=True)
    sorted_texts = [texts[i] for i in indices]
    if loader_kwargs.batch_size is None:
        loader_kwargs.batch_size = find_batch_size(sorted_texts, model, tokenizer, tokenizer_kwargs, device, window=window)
    loader = DataLoader(sorted_texts, **asdict(loader_kwargs), shuffle=False, collate_fn=None)
    outputs = compute_nll(loader, indices, model, tokenizer, tokenizer_kwargs, window=window, device=device)
    outputs.update(dict(total_chars=total_chars, total_tokens=total_tokens))
    return outputs


def main(output_dir: Path, data_path: Path, model_kwargs: ModelKwargs, window: int = None, input_key: str = "text", split: str = "test",
         tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), loader_kwargs: LoaderKwargs = LoaderKwargs()):
    """Compute the PPL and Surprisal of an LLM"""
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
    outputs = compute_ppl(texts, model, tokenizer, tokenizer_kwargs=tokenizer_kwargs, loader_kwargs=loader_kwargs, window=window, device=device)
    metrics = compute_metrics(**{k: outputs[k] for k in ["total_losses", "total_chars", "total_tokens"]})
    metrics.update({k: v for k, v in outputs.items() if isinstance(v, float)})
    print(metrics)
    metrics.update(dict(window=window, batch_size=loader_kwargs.batch_size, software=Path(__file__).stem))
    with open(output_dir/"metrics.json", "wt") as file:
        json.dump(metrics, file)
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            torch.save(v, output_dir/f"{k}.bin")


def cli():
    CLI(main, description=main.__doc__)