#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    def __init__(self, e_word):
        super(Highway, self).__init__()
        self.e_word = e_word
        self.proj = nn.Linear(in_features=self.e_word, out_features=self.e_word)
        self.gate = nn.Linear(in_features=self.e_word, out_features=self.e_word)

    def forward(self, x_conv_out):
        x_proj = F.relu(self.proj(x_conv_out))
        x_gate = torch.sigmoid(self.gate(x_conv_out))
        return x_gate * x_proj + (torch.tensor(1) - x_gate) * x_proj


### END YOUR CODE


if __name__ == '__main__':
    model = Highway(7)
    x_input = torch.randn(10, 7)
    x_highway = model(x_input)
    assert x_input.shape == x_highway.shape
