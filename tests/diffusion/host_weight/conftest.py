# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn


class TinyBlock(nn.Module):
    def __init__(self, start: float) -> None:
        super().__init__()
        base = torch.arange(start, start + 12, dtype=torch.bfloat16).reshape(3, 4)
        self.weight = nn.Parameter(base.t(), requires_grad=False)
        self.register_buffer("weight_scale", torch.tensor([start], dtype=torch.float32))
        self.register_buffer("scratch", torch.zeros(2), persistent=False)


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
        self.blocks = nn.ModuleList([TinyBlock(1.0), TinyBlock(20.0)])


class TinyPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = TinyTransformer()
        self.text_encoder = nn.Linear(2, 2, bias=False)
