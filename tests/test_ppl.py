import unittest

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import copy

from ppllm.ppl import count_tokens_chars, compute_ppl

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"#croissantllm/CroissantLLMBase"#
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# ensure right padding so we don't need attention mask
if tokenizer.padding_side != "right":
    tokenizer.padding_side = "right"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
dataset = [
    {"text": "I have a dream"},
    {"text": "I has a dream"},
    {"text": "I'm out for dead presidents to represent me"},
    {"text": "A language is a dialect with an army and navy"}
]
true_total_chars, true_total_tokens = torch.tensor([13, 12, 42, 44]), torch.tensor([3, 3, 8, 9])
true_total_losses = torch.tensor([10.7958, 14.0682, 51.1817, 46.1391])


class TestBase(unittest.TestCase):
    def assertAllTrue(self, tensor):
        return self.assertTrue(tensor.all())
    
    def assertAllEqual(self, a, b):
        return self.assertAllTrue(a==b)

    def assertAllClose(self, a, b):
        return self.assertAllTrue(torch.isclose(a, b, atol=1e-3))


class TestPpl(TestBase):
    def test_count_tokens_chars(self):
        total_chars, total_tokens = count_tokens_chars(dataset, tokenizer)
        self.assertAllEqual(total_chars, true_total_chars)
        self.assertAllEqual(total_tokens, true_total_tokens)

    def test_count_tokens_chars_context(self):
        total_chars, total_tokens = count_tokens_chars(dataset, tokenizer)
        dataset_with_context = copy.deepcopy(dataset)
        for item in dataset_with_context:
            item["context"] = ""
        context_total_chars, context_total_tokens = count_tokens_chars(dataset_with_context, tokenizer)
        self.assertAllEqual(total_chars, context_total_chars)
        self.assertAllEqual(total_tokens, context_total_tokens)

    def test_count_tokens_chars_context_non_empty(self):
        self.assertIsNotNone(tokenizer.bos_token)
        total_chars, _ = count_tokens_chars(dataset, tokenizer)
        context = "Some context "
        dataset_with_context = copy.deepcopy(dataset)
        for item in dataset_with_context:
            item["context"] = context_total_chars
            item["text"] = context + item["text"]
        context_total_chars, _ = count_tokens_chars(dataset_with_context, tokenizer)
        self.assertAllEqual(total_chars, context_total_chars-len(context))
    
    def test_compute_ppl(self):
        outputs = compute_ppl(dataset, model, tokenizer)
        self.assertAllClose(outputs["total_losses"], true_total_losses)

    def test_compute_ppl_context(self):
        context = "Some context "
        dataset_with_context = copy.deepcopy(dataset)
        for item in dataset_with_context:
            item["context"] = context
            item["text"] = context + item["text"]
        outputs = compute_ppl(dataset_with_context, model, tokenizer)
        self.assertAllTrue(outputs["all_losses"].reshape(len(dataset), -1)[:, :2]==0.)


if __name__ == '__main__':
    unittest.main()