from jsonargparse import CLI
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Any, Dict
from jsonargparse import CLI

import numpy as np

from vllm import LLM, SamplingParams

from .utils import load_texts


@dataclass
class LlmArguments:
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    trust_remote_code: bool = True
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    quantization: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    swap_space: float = 4
    cpu_offload_gb: float = 0
    enforce_eager: Optional[bool] = False
    max_seq_len_to_capture: int = 8192
    disable_custom_all_reduce: bool = False
    disable_async_output_proc: bool = False
    mm_processor_kwargs: Optional[Dict[str, Any]] = None
    max_model_len: int = None


def main(output_dir: Path, data_path: Path, llm_arguments: LlmArguments, input_key: str = "text", split: str = "test"):
    output_dir.mkdir(exist_ok=True, parents=True)
    llm = LLM(**asdict(llm_arguments))
    texts = load_texts(data_path, input_key=input_key, split=split)
    sampling_params = SamplingParams(prompt_logprobs=1, max_tokens=1)
    outputs = llm.generate(texts, sampling_params)
    assert len(outputs) == len(texts)
    total_logps, total_chars, total_tokens = [], [], []
    for text, output in zip(texts, outputs):
        assert output.prompt_logprobs[0] is None
        total_logp = 0
        for logps in output.prompt_logprobs[1:]:
            assert len(logps) <= 2
            logps = list(logps.values())
            # vllm always include the most probable token as well, 
            # we want to make sure it's at index 1 and that the attested token is at index 0
            if len(logps) == 2:
                assert logps[0].logprob <= logps[1].logprob
            total_logp += logps[0].logprob
        total_logps.append(total_logp)
        total_chars.append(len(text))
        total_tokens.append(len(output.prompt_logprobs[1:]))
    total_logps, total_chars, total_tokens = np.array(total_logps), np.array(total_chars), np.array(total_tokens)
    
    # surprisal is expressed in bits
    total_surprisal = -total_logps.sum()/np.log(2)
    metrics = {
        # N.B. this is the same as `np.exp(total_logps.sum()/total_tokens.sum())`
        "ppl": 2**(total_surprisal/total_tokens.sum()).item(),
        "bpc": (total_surprisal/total_chars.sum()).item(),
        "surprisal": total_surprisal.item()
    }

    print(metrics)
    with open(output_dir/"metrics.json", "wt") as file:
        json.dump(metrics, file)
    for name, array in zip(["log_prob", "char", "tokens"], [total_logps, total_chars, total_tokens]):
        with open(output_dir/f"{name}.npy", "wb") as file:
            np.save(file, array)


def cli():
    CLI(main)