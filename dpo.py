import torch, torch.nn.functional as F
from sft import build_sft_batch, sequence_logprobs

def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
   
    return {"chosen": build_sft_batch(tokenizer, prompts, chosen_responses, max_length),
            "rejected": build_sft_batch(tokenizer, prompts, rejected_responses, max_length)}

def get_logps(policy, reference, batch):

    pc = sequence_logprobs(policy, batch["chosen"])[0]
    pr = sequence_logprobs(policy, batch["rejected"])[0]
    with torch.no_grad():
        rc = sequence_logprobs(reference, batch["chosen"])[0].to(pc.device)
        rr = sequence_logprobs(reference, batch["rejected"])[0].to(pc.device)
    return pc, pr, rc, rr

def dpo_loss(pc, pr, rc, rr, beta):

    r_chosen, r_rejected = beta * (pc - rc), beta * (pr - rr)
    loss = -F.logsigmoid(r_chosen - r_rejected).mean()
    return loss, r_chosen.detach(), r_rejected.detach()

def compute_dpo_loss(policy, reference, tokenizer, prompts, chosen, rejected, beta, max_length):
    batch = build_dpo_batch(tokenizer, prompts, chosen, rejected, max_length)
    return dpo_loss(*get_logps(policy, reference, batch), beta)[0]
