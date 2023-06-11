from typing import Tuple

import torch
import torch.nn as nn

from transformers import GPT2LMHeadModel


class MLP(nn.Module):

    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) -1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size, device):
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens, prefix, mask=None, labels=None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length, prefix_size=512):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(prefix_size, self.gpt_embedding_size * prefix_length)
        else:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2, self.gpt_embedding_size * prefix_length))


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


# ============================================================================ #
#                                    Utils                                     #
# ============================================================================ #
def pad_tokens(text, tokenizer, prefix_length, max_seq_len=67):
    tokens = tokenizer(text, return_tensors='pt').input_ids[0]
    padding = max_seq_len - tokens.shape[0]
    if padding > 0:
        tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
    elif padding < 0:
        tokens = tokens[:max_seq_len]
    mask = tokens.ge(0)  # mask is zero where we out of sequence
    tokens[~mask] = 0
    mask = mask.float()
    mask = torch.cat((torch.ones(prefix_length), mask), dim=0)  # adding prefix mask
    return tokens, mask
