# ppllm
Blazing-Fast Python Library to Compute LLM's Perplexity and Surprisal

## Benchmark

Setup:
- NVIDIA V100 (32GB)
- Qwen3-8B
- 23k english sentences from EuroParl
- throughput in seconds

operation  | vllm | transformers
-----------|------|-------------
load model | 15   | 60
compute nll| 711  | 357
