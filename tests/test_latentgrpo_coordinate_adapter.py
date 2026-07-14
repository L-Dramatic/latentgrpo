import types
import unittest

import torch

from research.coordinate_invariance import (
    AffineChart,
    ContinuationEvaluator,
    LatentGRPOChartAdapter,
    LatentGRPOContinuationAdapter,
)


class TinyTokenizer:
    def __init__(self, vocabulary_size=29):
        self.vocabulary_size = vocabulary_size

    def __call__(
        self,
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64,
        add_special_tokens=False,
    ):
        del return_tensors, truncation, add_special_tokens
        ids = [1 + ord(character) % (self.vocabulary_size - 1) for character in text]
        ids = ids[:max_length] or [0]
        return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class TinyOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class TinyCausalLM(torch.nn.Module):
    def __init__(self, hidden_size=6, vocabulary_size=29):
        super().__init__()
        self.embed = torch.nn.Embedding(vocabulary_size, hidden_size, dtype=torch.float64)
        self.hidden = torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64)
        self.head = torch.nn.Linear(hidden_size, vocabulary_size, dtype=torch.float64)

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
        accumulated = inputs_embeds.cumsum(dim=1)
        hidden = torch.tanh(self.hidden(accumulated))
        return TinyOutput(self.head(hidden), (hidden,))


def make_tiny_model():
    torch.manual_seed(73)
    hidden_size = 6
    llm = TinyCausalLM(hidden_size=hidden_size)
    projection = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden_size, hidden_size, dtype=torch.float64),
    )
    return types.SimpleNamespace(
        config="small",
        llm_model=llm,
        llm_tokenizer=TinyTokenizer(),
        proj=projection,
    )


def native_thoughts(model, query_embeddings, count):
    current = query_embeddings
    thoughts = []
    with torch.no_grad():
        for _ in range(count):
            outputs = model.llm_model(
                inputs_embeds=current,
                attention_mask=torch.ones(current.shape[:2], dtype=torch.long),
                output_hidden_states=True,
            )
            thought = model.proj(outputs.hidden_states[-1][:, -1, :])
            thoughts.append(thought)
            current = torch.cat([current, thought.unsqueeze(1)], dim=1)
    return thoughts


class LatentGRPOCoordinateAdapterTest(unittest.TestCase):
    def setUp(self):
        self.model = make_tiny_model()
        token_ids = self.model.llm_tokenizer("question")["input_ids"]
        self.query_embeddings = self.model.llm_model.get_input_embeddings()(token_ids)

    def test_noop_affine_chart_preserves_recursive_thoughts(self):
        expected = native_thoughts(self.model, self.query_embeddings, 3)
        chart = AffineChart.with_condition_number(
            6, 8.0, seed=79, dtype=torch.float64, bias_scale=0.1
        )
        adapter = LatentGRPOChartAdapter(self.model, chart)

        trajectory = adapter.generate(
            self.query_embeddings,
            num_thoughts=3,
            seed=83,
            capture_trace=True,
            sample_ids=["q-1"],
        )

        self.assertEqual(len(trajectory.consumed_native_thoughts), 3)
        self.assertEqual(len(trajectory.trace), 3)
        for observed, native in zip(trajectory.consumed_native_thoughts, expected):
            torch.testing.assert_close(observed, native, atol=1e-12, rtol=1e-12)
        self.assertEqual(
            trajectory.trace.get("q-1", 2).prefix_state["remaining_latent_steps"],
            0,
        )

    def test_chart_operation_changes_consumed_latent(self):
        chart = AffineChart.with_condition_number(6, 5.0, seed=89, dtype=torch.float64)
        adapter = LatentGRPOChartAdapter(self.model, chart)

        baseline = adapter.generate(
            self.query_embeddings, num_thoughts=2, seed=97
        )

        def shift_first_step(charted, step_index, generator):
            del generator
            if step_index == 0:
                return charted + 0.1
            return charted

        shifted = adapter.generate(
            self.query_embeddings,
            num_thoughts=2,
            seed=97,
            operation=shift_first_step,
        )

        self.assertFalse(
            torch.allclose(
                baseline.consumed_native_thoughts[0],
                shifted.consumed_native_thoughts[0],
            )
        )
        self.assertFalse(
            torch.allclose(
                baseline.proposed_native_thoughts[1],
                shifted.proposed_native_thoughts[1],
            )
        )

    def test_trace_replay_has_exact_matched_seed_control(self):
        chart = AffineChart.identity(6, dtype=torch.float64)
        chart_adapter = LatentGRPOChartAdapter(self.model, chart)
        trajectory = chart_adapter.generate(
            self.query_embeddings,
            num_thoughts=3,
            seed=101,
            capture_trace=True,
            sample_ids=["q-2"],
        )
        record = trajectory.trace.get("q-2", 1)
        evaluator = ContinuationEvaluator(
            LatentGRPOContinuationAdapter(self.model, temperature=0.8)
        )

        control = evaluator.compare_record(record, record.latent.clone(), horizon=4)
        candidate = record.latent + torch.tensor(
            [0.2, 0.0, -0.1, 0.0, 0.1, -0.2], dtype=torch.float64
        )
        changed = evaluator.compare_record(
            record, candidate, horizon=4, seeds=[103, 107]
        )

        self.assertEqual(control.mean_divergence, 0.0)
        self.assertGreater(changed.mean_divergence, 0.0)
        self.assertEqual(changed.compute.rollout_calls, 6)


if __name__ == "__main__":
    unittest.main()

