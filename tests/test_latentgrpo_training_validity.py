import types
import sys
import unittest

import torch

transformers_stub = types.ModuleType("transformers")


class UnusedAutoClass:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise AssertionError("This test should not load Hugging Face models.")


transformers_stub.AutoModelForCausalLM = UnusedAutoClass
transformers_stub.AutoTokenizer = UnusedAutoClass
transformers_stub.set_seed = lambda seed: None
previous_transformers = sys.modules.get("transformers")
sys.modules["transformers"] = transformers_stub
try:
    from models.latentgrpo import LatentGRPO
finally:
    if previous_transformers is None:
        sys.modules.pop("transformers", None)
    else:
        sys.modules["transformers"] = previous_transformers


class TinyTokenizer:
    eos_token = "<eos>"
    eos_token_id = 0
    pad_token = eos_token

    def __init__(self, vocab_size=32):
        self.vocab_size = vocab_size

    def __call__(
        self,
        text,
        return_tensors="pt",
        truncation=True,
        max_length=32,
        add_special_tokens=False,
    ):
        del return_tensors, truncation, add_special_tokens
        ids = [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]
        ids = ids[: int(max_length)] or [self.eos_token_id]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    def decode(self, ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(i)) for i in ids)


class TinyOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class TinyCausalLM(torch.nn.Module):
    def __init__(self, vocab_size=32, hidden_size=8):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden_size)
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        output_hidden_states=False,
    ):
        del attention_mask, output_hidden_states
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        causal_context = inputs_embeds.cumsum(dim=1)
        hidden_states = torch.tanh(self.hidden(causal_context))
        logits = self.lm_head(hidden_states)
        return TinyOutput(logits=logits, hidden_states=(hidden_states,))

    def generate(self, inputs_embeds, attention_mask=None, max_new_tokens=2, **kwargs):
        del inputs_embeds, attention_mask, kwargs
        return torch.tensor([[4] * max_new_tokens], dtype=torch.long)


def make_tiny_latentgrpo():
    hidden_size = 8
    model = LatentGRPO.__new__(LatentGRPO)
    torch.nn.Module.__init__(model)
    model.config = "small"
    model.llm_model_name = "tiny"
    model.llm_model = TinyCausalLM(hidden_size=hidden_size)
    model.llm_model.eval()
    model.llm_tokenizer = TinyTokenizer()
    for param in model.llm_model.parameters():
        param.requires_grad = False
    model.proj = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.GELU(),
        torch.nn.Linear(hidden_size, hidden_size),
        torch.nn.LayerNorm(hidden_size),
    )
    model.ref_proj = None
    return model


class LatentGRPOTrainingValidityTest(unittest.TestCase):
    def test_answer_logprob_backpropagates_to_projection(self):
        model = make_tiny_latentgrpo()
        query_embeddings, _ = model.get_input_embeddings("question", 16, "cpu")
        thoughts = model.generate_continuous_thoughts(
            query_embeddings, num_thoughts=2, device="cpu"
        )

        log_probs = model.generate_answer(
            query_embeddings, thoughts, answer_text="42", max_gen_length=8
        )
        self.assertEqual(log_probs.shape, torch.Size([1]))
        self.assertTrue(log_probs.requires_grad)

        (-log_probs.mean()).backward()

        grad_norm = sum(
            param.grad.abs().sum().item()
            for param in model.proj.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)

    def test_advantages_are_finite_for_single_or_equal_rewards(self):
        model = make_tiny_latentgrpo()

        single = model.compute_advantages([1.0])
        equal = model.compute_advantages([2.0, 2.0, 2.0])

        self.assertTrue(torch.isfinite(single).all())
        self.assertTrue(torch.isfinite(equal).all())
        self.assertTrue(torch.equal(single, torch.zeros_like(single)))
        self.assertTrue(torch.equal(equal, torch.zeros_like(equal)))

    def test_policy_kl_regularizer_backpropagates_to_projection(self):
        model = make_tiny_latentgrpo()
        model.save_reference_projection()
        with torch.no_grad():
            for param in model.proj.parameters():
                param.add_(0.1)

        log_probs = torch.tensor([-0.5, -1.5], requires_grad=True)
        advantages = torch.tensor([1.0, -1.0])
        total_loss, policy_loss, kl_loss = model.compute_policy_loss(
            log_probs, advantages, beta=0.1
        )

        self.assertTrue(total_loss.requires_grad)
        self.assertTrue(policy_loss.requires_grad)
        self.assertTrue(kl_loss.requires_grad)

        total_loss.backward()

        self.assertIsNotNone(log_probs.grad)
        grad_norm = sum(
            param.grad.abs().sum().item()
            for param in model.proj.parameters()
            if param.grad is not None
        )
        self.assertGreater(grad_norm, 0.0)


if __name__ == "__main__":
    unittest.main()
