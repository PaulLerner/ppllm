import unittest

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ppllm.ppl import count_tokens_chars, compute_ppl
from ppllm.utils import fix_tokenizer


class TestBase(unittest.TestCase):
    def assertAllTrue(self, tensor):
        return self.assertTrue(tensor.all())
    
    def assertAllEqual(self, a, b):
        return self.assertAllTrue(a==b)

    def assertAllClose(self, a, b):
        return self.assertAllTrue(torch.isclose(a, b, atol=1e-3))


class AbstractTestPpl:
    def setUp(self):
        self.dataset = [
            {"text": "I have a dream"},
            {"text": "I has a dream"},
            {"text": "I'm out for dead presidents to represent me"},
            {"text": "A language is a dialect with an army and navy"}
        ]

    def test_count_tokens_chars(self):
        total_chars, total_tokens = count_tokens_chars(self.dataset, self.tokenizer)
        self.assertAllEqual(total_chars, self.true_total_chars)
        self.assertAllEqual(total_tokens, self.true_total_tokens)

    def test_count_tokens_chars_context(self):
        total_chars, total_tokens = count_tokens_chars(self.dataset, self.tokenizer)
        for item in self.dataset:
            item["context"] = ""
        context_total_chars, context_total_tokens = count_tokens_chars(self.dataset, self.tokenizer)
        self.assertAllEqual(total_chars, context_total_chars)
        self.assertAllEqual(total_tokens, context_total_tokens)

    def test_count_tokens_chars_context_non_empty(self):
        if self.tokenizer.bos_token is None:
            return
        total_chars, _ = count_tokens_chars(self.dataset, self.tokenizer)
        context = "Some context "
        for item in self.dataset:
            item["context"] = context_total_chars
            item["text"] = context + item["text"]
        context_total_chars, _ = count_tokens_chars(self.dataset, self.tokenizer)
        self.assertAllEqual(total_chars, context_total_chars-len(context))
    
    def test_compute_ppl(self):
        outputs = compute_ppl(self.dataset, self.model, self.tokenizer)
        self.assertAllClose(outputs["total_losses"], self.true_total_losses)

    def test_compute_ppl_context(self):
        context = "Some context "
        for item in self.dataset:
            item["context"] = context
            item["text"] = context + item["text"]
        outputs = compute_ppl(self.dataset, self.model, self.tokenizer)
        self.assertAllTrue(outputs["all_losses"].reshape(len(self.dataset), -1)[:, :2]==0.)


class TestQwen3_0_6B_Base(AbstractTestPpl, TestBase):
    @classmethod
    def setUpClass(cls):
        MODEL_NAME = "Qwen/Qwen3-0.6B-Base"#croissantllm/CroissantLLMBase"#
        cls.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.true_total_chars = torch.tensor([13, 12, 42, 44])
        cls.true_total_tokens = torch.tensor([3, 3, 8, 9])
        cls.true_total_losses = torch.tensor([10.7958, 14.0682, 51.1817, 46.1391])
        fix_tokenizer(cls.tokenizer)
        

if __name__ == '__main__':
    unittest.main()