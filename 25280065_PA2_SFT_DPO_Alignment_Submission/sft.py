import torch, torch.nn.functional as F

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction)

def build_sft_batch(tokenizer, prompts, responses, max_length):
    """prompt + response + EOS, truncate, right-pad. Labels: -100 on prompt and padding."""
    eos = [] if tokenizer.eos_token_id is None else [tokenizer.eos_token_id]
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (eos[0] if eos else 0)
    rows = []
    for p, r in zip(prompts, responses):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        r_ids = tokenizer.encode(r, add_special_tokens=False) + eos        # EOS before truncation
        rows.append(((p_ids + r_ids)[:max_length], ([-100] * len(p_ids) + r_ids)[:max_length]))
    width = max(len(ids) for ids, _ in rows)
    return {
        "input_ids": torch.tensor([ids + [pad] * (width - len(ids)) for ids, _ in rows]),
        "attention_mask": torch.tensor([[1] * len(ids) + [0] * (width - len(ids)) for ids, _ in rows]),
        "labels": torch.tensor([lab + [-100] * (width - len(lab)) for _, lab in rows]),
    }

def sequence_logprobs(model, batch):
    """Per row: summed log-prob of response tokens, and number of response tokens."""
    batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1].float()
    labels = batch["labels"][:, 1:]                                         # token t predicts t+1
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                          ignore_index=-100, reduction="none").view(labels.shape)
    mask = labels.ne(-100)
    return -(nll * mask).sum(1), mask.sum(1)

def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return sequence_logprobs(model, build_sft_batch(tokenizer, prompts, responses, max_length))[0]

def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    """Mean negative log-prob over all response tokens in the batch."""
    logps, counts = sequence_logprobs(model, build_sft_batch(tokenizer, prompts, responses, max_length))
    return -logps.sum() / counts.sum().clamp_min(1)
