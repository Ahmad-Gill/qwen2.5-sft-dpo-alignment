from __future__ import annotations
import sft

def build_sft_batch(tokenizer, prompts, responses, max_length):
    return sft.build_sft_batch(tokenizer, prompts, responses, max_length)

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return sft.compute_response_logprobs(model, tokenizer, prompts, responses, max_length)

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return sft.compute_sft_loss(model, tokenizer, prompts, responses, max_length)

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    raise NotImplementedError("Task 4")

def compute_dpo_loss_from_logps(policy_chosen_logps, policy_rejected_logps,
                                reference_chosen_logps, reference_rejected_logps, beta):
    raise NotImplementedError("Task 4")

def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts,
                     chosen_responses, rejected_responses, beta, max_length):
    raise NotImplementedError("Task 4")
