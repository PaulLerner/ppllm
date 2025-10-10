# 🤔 ppllm
Blazing-Fast Python Library to Compute LLM's Perplexity and Surprisal

## Features
### Windowed PPL
Because Transformers have a quadratic complexity 
([Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
[Tay et al., 2023](https://doi.org/10.1145/3530811)),
computing the PPL of long texts is expensive.
Windowed PPL restrains the context size to a fixed window as illustrated below (e.g. of 64 tokens, so the quadratic complexity is not so bad)

#### Without window (context size may get long)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_full.gif)

#### With window (fixed context size)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_sliding.gif)

In practice, 🤔 ppllm uses a stride of half the window size, instead of the unit stride illustrated here 
(which would require as many forward passes as the number of tokens in the sequence, which would defeat the purpose)

(Illustration by https://huggingface.co/docs/transformers/perplexity)

## Installation
TODO

## Usage
### CLI
```bash
python -m ppllm /path/to/output /path/to/data --model_kwargs.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B --window=64
```

Omit `--window` to compute PPL with the entire context

Use `python -m ppllm -h` to see all arguments

🤔 ppllm relies on `jsonargparse` so you can use yaml configs:
```yaml
>>> python -m ppllm /path/to/output /path/to/data --model_kwargs.pretrained_model_name_or_path=meta-llama/Llama-3.1-8B --window=64 --print_config
output_dir: /path/to/output
data_path: /path/to/data
model_kwargs:
  pretrained_model_name_or_path: meta-llama/Llama-3.1-8B
  config: null
  cache_dir: null
  ignore_mismatched_sizes: false
  force_download: false
  local_files_only: false
  token: null
  revision: main
  use_safetensors: null
  resume_download: false
  output_loading_info: false
  dtype: float16
  load_in_8bit: false
  load_in_4bit: false
  attn_implementation: null
  trust_remote_code: true
window: 64
input_key: text
split: test
tokenizer_kwargs:
  return_tensors: pt
  padding: longest
  truncation: false
  return_overflowing_tokens: false
  max_length: null
loader_kwargs:
  batch_size: null
  num_workers: 4
  pin_memory: false
  drop_last: false
  timeout: 0
  prefetch_factor: null
  persistent_workers: false
  pin_memory_device: ''

>>> python -m ppllm --config=/path/to/config.yaml
```

## Benchmark

Setup: 
- NVIDIA V100 (32GB)
- Llama-3.1-8B

### wikitext-2-v1

software | throughput in seconds
-----------|------
vllm | 328
hf_shuffle | 364
transformers sorted v2 b4bf445eb46c2d73439a5046540e6649bd86f368 | 79
transformers sorted v2 window=128 batch=32 b4bf445eb46c2d73439a5046540e6649bd86f368 | 108
transformers sorted v2 window=128 batch=128 b4bf445eb46c2d73439a5046540e6649bd86f368 | 121

It seems that the sequences are too short to take advantage of windowed PPL and that sorting the text by length is enough 
(to get rid of the long tail/inefficient padding)
![](docs/wikitext-2-v1_Llama-3.1-8B.png)


### 23k english sentences from EuroParl

software | throughput in seconds
-----------|------
vllm | TODO
transformers naive v1 a2e221a3bb64d7d1a8b0c9de149ea39e7371f3fc | 328
transformers sorted v2 007adde493d98c185870b4743495e9e2ae03fa5d | 222
transformers sorted v2 window=64 007adde493d98c185870b4743495e9e2ae03fa5d | 234

transformers static batch size (higher gets OOM): 32 

#### Length distribution
It seems that the sequences are too short to take advantage of windowed PPL and that sorting the text by length is enough 
(to get rid of the long tail/inefficient padding)

![](docs/21-europarl-en.png)

### 500k multilingual sentences from Parlamint
both vllm and transformers (naive v1) get OOM, even with a batch size of 1, even when scaling to H100

Windowed PPL should fix it (TODO)
