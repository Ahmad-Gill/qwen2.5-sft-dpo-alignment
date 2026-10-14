"""Basic public checks for the submission adapter contract.

These tests intentionally cover only common mistakes. Passing them does not
guarantee that the implementation will pass the private correctness tests.
"""

from __future__ import annotations

import inspect

import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


REQUIRED_FUNCTIONS = {
    "build_sft_batch": ["tokenizer", "prompts", "responses", "max_length"],
    "compute_response_logprobs": ["model", "tokenizer", "prompts", "responses", "max_length"],
    "compute_sft_loss": ["model", "tokenizer", "prompts", "responses", "max_length"],
    "build_dpo_batch": ["tokenizer", "prompts", "chosen_responses", "rejected_responses", "max_length"],
    "compute_dpo_loss_from_logps": [
        "policy_chosen_logps",
        "policy_rejected_logps",
        "reference_chosen_logps",
        "reference_rejected_logps",
        "beta",
    ],
    "compute_dpo_loss": [
        "policy_model",
        "reference_model",
        "tokenizer",
        "prompts",
        "chosen_responses",
        "rejected_responses",
        "beta",
        "max_length",
    ],
}


def test_required_functions_have_the_documented_signatures():
    for name, expected_parameters in REQUIRED_FUNCTIONS.items():
        function = getattr(adapter, name, None)
        assert callable(function), f"submission_adapter.py must define {name}"
        assert list(inspect.signature(function).parameters) == expected_parameters


def test_sft_batch_masks_prompt_and_padding_and_includes_eos():
    tokenizer = ToyTokenizer()
    prompts = ["Task: ", "Q: "]
    responses = ["ok", "yes"]
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)

    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert all(isinstance(value, torch.Tensor) for value in batch.values())
    assert batch["input_ids"].shape == batch["attention_mask"].shape == batch["labels"].shape
    assert batch["input_ids"].shape[0] == len(prompts)

    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
        response_length = len(tokenizer.encode(response, add_special_tokens=False)) + 1
        assert batch["labels"][row, :prompt_length].eq(-100).all()
        assert batch["labels"][row, prompt_length : prompt_length + response_length].ne(-100).all()
        assert batch["labels"][row, prompt_length + response_length :].eq(-100).all()
        assert batch["labels"][row, prompt_length + response_length - 1] == tokenizer.eos_token_id

    unmasked = batch["labels"].ne(-100)
    assert torch.equal(batch["input_ids"][unmasked], batch["labels"][unmasked])
    assert batch["attention_mask"][batch["labels"].eq(-100)].eq(0).any()


def test_response_logprobs_are_one_value_per_row_and_response_only():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.7)
    prompts = ["aaaa: ", "zzzz: "]
    responses = ["same", "same"]

    logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=20)

    assert logps.shape == (2,)
    assert torch.isfinite(logps).all()
    torch.testing.assert_close(logps[0], logps[1])


def test_sft_loss_uses_an_accepted_reduction_and_has_policy_gradients():
    tokenizer = ToyTokenizer()
    model = ToyCausalLM(tokenizer.vocab_size, bias_scale=0.2)
    prompts = ["Do: ", "Say: "]
    responses = ["a", "longer"]
    loss = adapter.compute_sft_loss(model, tokenizer, prompts, responses, max_length=16)
    response_logps = adapter.compute_response_logprobs(model, tokenizer, prompts, responses, max_length=16)
    batch = adapter.build_sft_batch(tokenizer, prompts, responses, max_length=16)
    response_token_count = batch["labels"][:, 1:].ne(-100).sum().clamp_min(1)
    accepted = (-response_logps.mean(), -response_logps.sum() / response_token_count)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert any(torch.allclose(loss, expected, rtol=1e-5, atol=1e-6) for expected in accepted)
    loss.backward()
    assert model.logit_bias.grad is not None
    assert model.logit_bias.grad.abs().sum() > 0


def test_dpo_batch_has_chosen_and_rejected_lm_batches():
    tokenizer = ToyTokenizer()
    prompts = ["Task: ", "Q: "]
    chosen = ["good", "yes"]
    rejected = ["bad", "no"]
    batch = adapter.build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length=16)

    assert set(batch) == {"chosen", "rejected"}
    for side in ("chosen", "rejected"):
        assert set(batch[side]) == {"input_ids", "attention_mask", "labels"}
        assert all(isinstance(value, torch.Tensor) for value in batch[side].values())
        assert batch[side]["input_ids"].shape == batch[side]["attention_mask"].shape
        assert batch[side]["input_ids"].shape == batch[side]["labels"].shape
        assert batch[side]["input_ids"].shape[0] == len(prompts)
        assert batch[side]["labels"].ne(-100).any()


def test_dpo_loss_from_equal_logps_is_log_two():
    values = torch.tensor([-2.0, -5.0])
    loss = adapter.compute_dpo_loss_from_logps(values, values, values, values, beta=0.2)

    assert loss.ndim == 0
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))


def test_dpo_loss_is_scalar_and_does_not_update_reference_model():
    tokenizer = ToyTokenizer()
    policy = ToyCausalLM(tokenizer.vocab_size)
    reference = ToyCausalLM(tokenizer.vocab_size)
    loss = adapter.compute_dpo_loss(
        policy,
        reference,
        tokenizer,
        ["Pick: ", "Choose: "],
        ["a", "a"],
        ["b", "b"],
        beta=0.2,
        max_length=12,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert policy.logit_bias.grad is not None
    assert policy.logit_bias.grad.abs().sum() > 0
    assert reference.logit_bias.grad is None or reference.logit_bias.grad.abs().sum() == 0
