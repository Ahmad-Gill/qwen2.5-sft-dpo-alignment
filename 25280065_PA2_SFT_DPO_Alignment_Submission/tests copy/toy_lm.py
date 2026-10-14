"""Small offline tokenizer and causal LM used by the public adapter tests."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self):
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-_")
        self.vocab = {self.pad_token: self.pad_token_id, self.eos_token: self.eos_token_id}
        for char in chars:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)
        self.inv_vocab = {token_id: token for token, token_id in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def encode(self, text, add_special_tokens=False):
        token_ids = []
        index = 0
        while index < len(text):
            if text.startswith(self.eos_token, index):
                token_ids.append(self.eos_token_id)
                index += len(self.eos_token)
                continue
            if text.startswith(self.pad_token, index):
                token_ids.append(self.pad_token_id)
                index += len(self.pad_token)
                continue
            token_ids.append(self.vocab[text[index]])
            index += 1
        if add_special_tokens:
            token_ids.append(self.eos_token_id)
        return token_ids

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
    ):
        is_batch = isinstance(text, list)
        texts = text if is_batch else [text]
        encoded = [self.encode(item, add_special_tokens=add_special_tokens) for item in texts]
        if truncation and max_length is not None:
            encoded = [token_ids[:max_length] for token_ids in encoded]

        attention_mask = [[1] * len(token_ids) for token_ids in encoded]
        if padding:
            width = max(len(token_ids) for token_ids in encoded)
            if padding == "max_length" and max_length is not None:
                width = max_length
            encoded = [token_ids + [self.pad_token_id] * (width - len(token_ids)) for token_ids in encoded]
            attention_mask = [mask + [0] * (width - len(mask)) for mask in attention_mask]

        if return_tensors == "pt":
            return SimpleNamespace(
                input_ids=torch.tensor(encoded, dtype=torch.long),
                attention_mask=torch.tensor(attention_mask, dtype=torch.long),
            )
        if is_batch:
            return {"input_ids": encoded, "attention_mask": attention_mask}
        return SimpleNamespace(input_ids=encoded[0], attention_mask=attention_mask[0])


class ToyCausalLM(nn.Module):
    """Causal LM with trainable token logits independent of input position."""

    def __init__(self, vocab_size, bias_scale=0.0):
        super().__init__()
        initial_bias = torch.linspace(-0.3, 0.3, vocab_size) * bias_scale
        self.logit_bias = nn.Parameter(initial_bias.clone())

    def forward(self, input_ids, attention_mask=None, labels=None):
        batch_size, sequence_length = input_ids.shape
        logits = self.logit_bias.view(1, 1, -1).expand(batch_size, sequence_length, -1).contiguous()
        return SimpleNamespace(logits=logits)
