import unittest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

from ppllm.ppl import LoaderKwargs, count_tokens_chars, compute_ppl
from ppllm.utils import fix_tokenizer


os.environ["TOKENIZERS_PARALLELISM"]="true"


def format_msg(msg):
    if msg is None:
        return ""
    return f" : {msg}"


class TestBase(unittest.TestCase):
    def assertAllTrue(self, tensor, msg=None):
        return self.assertTrue(tensor.all(), msg=f"{tensor} is not all true{format_msg(msg)}")
    
    def assertAllEqual(self, a, b, msg=None):
        return self.assertAllTrue(a==b, msg=f"{a} and {b} are not all equal{format_msg(msg)}")

    def assertAllClose(self, a, b, msg=None):
        return self.assertAllTrue(torch.isclose(a, b, atol=1e-3), msg=f"{a} and {b} are not all close{format_msg(msg)}")


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
        context_total_chars, context_total_tokens = count_tokens_chars(self.dataset, self.tokenizer, context_key="context")
        self.assertAllEqual(total_chars, context_total_chars)
        self.assertAllEqual(total_tokens, context_total_tokens)

    def test_count_tokens_chars_context_non_empty(self):
        if self.tokenizer.bos_token is None:
            return
        total_chars, _ = count_tokens_chars(self.dataset, self.tokenizer)
        context = "Some context "
        for item in self.dataset:
            item["context"] = context
            item["text"] = context + item["text"]
        context_total_chars, _ = count_tokens_chars(self.dataset, self.tokenizer, context_key="context")
        self.assertAllEqual(total_chars, context_total_chars)
    
    def test_compute_ppl(self):
        outputs = compute_ppl(self.dataset, self.model, self.tokenizer, loader_kwargs=LoaderKwargs(batch_size=len(self.dataset)))
        self.assertAllClose(outputs["total_losses"], self.true_total_losses)

    def test_compute_ppl_context(self):
        context = "Some context "
        for item in self.dataset:
            item["context"] = context
            item["text"] = context + item["text"]
        outputs = compute_ppl(self.dataset, self.model, self.tokenizer, loader_kwargs=LoaderKwargs(batch_size=len(self.dataset)), context_key="context")
        self.assertAllTrue(outputs["all_losses"].reshape(len(self.dataset), -1)[:, :2]==0.)


class TestQwen3_0_6B_Base(AbstractTestPpl, TestBase):
    @classmethod
    def setUpClass(cls):
        MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
        cls.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.true_total_chars = torch.tensor([13, 12, 42, 44])
        cls.true_total_tokens = torch.tensor([3, 3, 8, 9])
        cls.true_total_losses = torch.tensor([10.7958, 14.0682, 51.1817, 46.1391])
        fix_tokenizer(cls.tokenizer)


class TestGemma_3_4b_it(AbstractTestPpl, TestBase):
    @classmethod
    def setUpClass(cls):
        MODEL_NAME = "google/gemma-3-4b-it"
        cls.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=True)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.true_total_chars = torch.tensor([14, 13, 43, 45])
        cls.true_total_tokens = torch.tensor([ 4,  4, 10, 10])
        cls.true_total_losses = torch.tensor([28.1614, 21.5981, 80.7367, 47.4722])
        fix_tokenizer(cls.tokenizer)


class TestCroissantLLMBase(AbstractTestPpl, TestBase):
    @classmethod
    def setUpClass(cls):
        MODEL_NAME = "croissantllm/CroissantLLMBase"
        cls.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        cls.true_total_chars = torch.tensor([14, 13, 43, 45])
        cls.true_total_tokens = torch.tensor([ 4,  4,  9, 13])
        cls.true_total_losses = torch.tensor([14.2056, 22.8031, 54.3119, 42.1241])
        fix_tokenizer(cls.tokenizer)
        

if __name__ == '__main__':
    unittest.main()