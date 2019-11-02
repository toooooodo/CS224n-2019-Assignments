#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, in_channel, out_channel, k=5):
        """

        :param in_channel: e_char
        :param out_channel: e_word
        :param k:
        """
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.k = k
        self.conv_layer = nn.Conv1d(self.in_channel, self.out_channel, kernel_size=self.k)

    def forward(self, x):
        """

        :param x: (N, C_in, L_in)
        :return:
        """
        x = self.conv_layer(x)  # (N, C_out, L_out)  ->  (N, C_out)
        x = torch.max(F.relu(x), dim=2)[0]
        return x


### END YOUR CODE


if __name__ == '__main__':
    model = CNN(50, 256)  # char_e=50, word_e=256
    input = torch.randn(100, 50, 30)  # N=100, char_e=50, word_len=30
    output = model(input)
    assert output.shape == torch.Size([100, 256])
