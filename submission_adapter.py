"""Stable grading interface for the SFT and DPO implementation.

Keep the function signatures below unchanged. Each function should be a thin
wrapper around your own implementation; you do not need to place the training
code itself in this file.
"""

from __future__ import annotations


def build_sft_batch(tokenizer, prompts, responses, max_length):
    """Return input_ids, attention_mask, and response-only labels."""

    raise NotImplementedError("Connect this adapter to your SFT batch builder")


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    """Return one summed response-token log-probability per input row."""

    raise NotImplementedError("Connect this adapter to your log-probability code")


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Return a scalar response-only SFT loss using either documented reduction."""

    raise NotImplementedError("Connect this adapter to your SFT loss code")


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    """Return separate response-masked LM batches for chosen and rejected rows."""

    raise NotImplementedError("Connect this adapter to your DPO batch builder")


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return the scalar mean DPO objective from per-example log probabilities."""

    raise NotImplementedError("Connect this adapter to your tensor-level DPO loss")


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    """Return a scalar mean DPO loss with a frozen reference computation."""

    raise NotImplementedError("Connect this adapter to your DPO loss code")
