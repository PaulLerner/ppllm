<a target="_blank" href="https://colab.research.google.com/github/PaulLerner/ppllm/blob/main/ppllm_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

# 🤔 ppllm
A Python Library to Compute LLM's Perplexity and Surprisal

## Features
🤔 ppllm allows to compute various information-theoretic metrics given a text and an LLM, 
including Perplexity (PPL), Surprisal, and bits per character (BPC).

🤔 ppllm implements windowed PPL, which allows to compute the PPL of arbitrarily long texts.
It offers both a CLI and a python API.

Software | PPL | Surprisal | BPC | Long texts | CLI | API
---------|-----|-----------|-----|------------|-----|-----
[lmppl](https://github.com/asahi417/lmppl) |  ✅ | ❌ | ❌ | ❌ | ❌|  ✅
[surprisal_from_llm](https://github.com/remo-help/surprisal_from_llm) | ❌|  ✅| ❌ | ❌|  ✅| ❌
[evaluate](https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py) |  ✅ | ❌ | ❌|  ✅| ❌|  ✅
🤔 ppllm | ✅ | ✅ | ✅|  ✅|  ✅|  ✅


Upcoming metrics (see [the roadmap](github.com/PaulLerner/ppllm/issues/1)):
- word-level surprisal
- bits per byte (BPB)


🤔 ppllm is benchmarked against:
- a [vllm-based implementation](benchmark/vllmppl.py): 4.15 times faster!
- a [naive hugging face implementation](benchmark/hf_shuffle.py), which does not sort texts by length: 4.61 times faster!

### Windowed PPL
Some texts are too long to fit in a model, especially since Transformers have a quadratic complexity 
([Vaswani et al., 2017](https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html),
[Tay et al., 2023](https://doi.org/10.1145/3530811)).
Windowed PPL restrains the context size to a fixed window as illustrated below (e.g. of 1024 tokens)

#### Without window (context size may get long)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_full.gif)

#### With window (fixed context size)
![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/ppl_sliding.gif)

In practice, 🤔 ppllm uses a stride of half the window size, instead of the unit stride illustrated here.

(Illustration by https://huggingface.co/docs/transformers/perplexity)

### Metrics
All metrics are defined assuming access to a (large) language model that defines a probability distribution over a sequence of tokens $x=(x_1,x_2,\dots,x_L)$:

$P(x) = P(x_1|x_0)  P(x_2|x_{<2})  \dots P(x_L|x_{<L})$

Where $x_0$ denotes a special token marking the beginning of the sequence (`bos_token` in `transformers`).
Note that some models do not have such a token. In this case, the probability of $x_1$ is not taken into account (i.e. we assume that $P(x_1|x_0)=1$).


For numerical stability, we compute the log probability:


$\log P(x) = \log P(x_1|x_0)  + \log P(x_2|x_{<2})+  ... +  \log P(x_L|x_{<L})$

From this, we can compute the negative log probability (aka negative log likelihood, aka cross-entropy), which is the loss LLMs are trained to minimize (during pretraining):

$$\mathcal{L}(x)=-\log P(x)$$

Then comes surprisal, which is the same but is usually expressed in bits, using a $\log_2$ logarithm:


$$S(x)=-\log_2 P(x)=\frac{\mathcal{L}(x)}{\log(2)}$$

From surprisal, we can define bits per character (BPC), which simply normalizes the surprisal by the number of characters $C$ of $x$:

$$\mathrm{BPC}(x)=\frac{S(x)}{C}$$

Similarly, we define perplexity (PPL), which normalizes the invert probability by the number of tokens $L$, which is equivalent to the exponentiate of the surprisal normalized by $L$:

$$\mathrm{PPL}(x)=\sqrt[L]{\frac{1}{P(x)}}=2^{\frac{S(x)}{L}}=\exp\left(\frac{\mathcal{L}(x)}{L}\right)$$

## Installation
### via pip
`pip install ppllm`

### via uv
`uv add ppllm`

### editable
```bash
git clone https://github.com/PaulLerner/ppllm.git
cd ppllm
uv sync
```

## Usage
### Python
🤔 ppllm is a pythonic library, see the [example notebook](ppllm_example.ipynb) 
to see how to use it from python 
(you can also [open it in Colab](https://colab.research.google.com/github/PaulLerner/ppllm/blob/main/ppllm_example.ipynb))

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

TODO describe data input/output formats

## Contributing
Feel free to open an issue or PR to contribute. 
The [roadmap](github.com/PaulLerner/ppllm/issues/1) will probably never happen without your help :)

### Building
TODO bump version manually pyproject.toml or using `uv bump`?

```bash
uv build
uv publish --token=<TOKEN>
```

## Benchmark

Setup: 
- NVIDIA V100 (32GB)
- Llama-3.1-8B
- wikitext-2-v1

software | compute time in seconds (↓)
-----------|------
vllm | 328
hf_shuffle | 364
🤔 ppllm (window=128)  | 108
🤔 ppllm (no window) | 79

On Wikitext, because texts are quite short, it's no use computing windowed PPL and directly computing PPL of the full text is faster.
However, if texts get longer than 10,000 tokens, a V100 will probably go OOM even with a batch size of 1, so windowed PPL is essential.

Apart from this, we can see that the naive hugginface based-implementation (which does not sort texts by length) is on par with vllm.
However, when sorting texts by length as in 🤔 ppllm, we get more than 4 times faster than vllm!

![](docs/wikitext-2-v1_Llama-3.1-8B.png)
