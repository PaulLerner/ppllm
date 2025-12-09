import json
from pathlib import Path
from jsonargparse import CLI
from dataclasses import asdict
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from iso639 import Lang

from .utils import load_dataset, fix_tokenizer
from .ppl import compute_ppl, compute_metrics, ModelKwargs, TokenizerKwargs, LoaderKwargs


def main(output_dir: Path, data_path: Path, model_kwargs: ModelKwargs, srcs: List[str], tgts: List[str], window: int = None, split: str = "test",
         tokenizer_kwargs: TokenizerKwargs = TokenizerKwargs(), loader_kwargs: LoaderKwargs = LoaderKwargs(), template="{src_lang}: {src_text}\n{tgt_lang}:", chat: bool = False):
    """Compute the PPL and Surprisal of an LLM on a translation conditioned on the source text"""
    if chat:
        raise NotImplementedError(f"{chat=}")
    assert window is None or window%2 == 0, f"window must be dividible by 2, got {window}"
    output_dir.mkdir(exist_ok=True, parents=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_kwargs.pretrained_model_name_or_path, 
        add_prefix_space=False, 
        add_eos_token=False, 
        trust_remote_code=model_kwargs.trust_remote_code
    )
    fix_tokenizer(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(**asdict(model_kwargs))
    dataset = load_dataset(data_path, split=split)
    for src in srcs:
        src_name = Lang(src).name
        (output_dir/src).mkdir(exist_ok=True)
        for tgt in tgts:
            if tgt == src:
                continue
            tgt_name = Lang(tgt).name
            lp_output_dir = output_dir/src/tgt
            lp_output_dir.mkdir(exist_ok=True)
            for item in dataset:
                item["context"] = template.format(src_lang=src_name, src_text=item[src], tgt_lang=tgt_name)
                item["text"] = f'{item["context"]} {item[tgt]}'
            outputs = compute_ppl(dataset, model, tokenizer, tokenizer_kwargs=tokenizer_kwargs, loader_kwargs=loader_kwargs, window=window, input_key="text", context_key="context")
            metrics = compute_metrics(**{k: outputs[k] for k in ["total_losses", "total_chars", "total_tokens"]})
            metrics.update({k: v for k, v in outputs.items() if isinstance(v, float)})
            metrics.update(dict(src=src, tgt=tgt, window=window, template=template, batch_size=loader_kwargs.batch_size, software=Path(__file__).stem))
            print(metrics)
            with open(lp_output_dir/"metrics.json", "wt") as file:
                json.dump(metrics, file)
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    torch.save(v, lp_output_dir/f"{k}.bin")


if __name__ == "__main__":
    CLI(main, description=main.__doc__)
