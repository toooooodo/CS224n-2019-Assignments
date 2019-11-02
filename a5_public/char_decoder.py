#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn


class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.char_embedding_size = char_embedding_size
        self.target_vocab = target_vocab
        self.charDecoder = nn.LSTM(input_size=self.char_embedding_size, hidden_size=self.hidden_size)
        self.char_output_projection = nn.Linear(in_features=self.hidden_size,
                                                out_features=len(self.target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(num_embeddings=len(self.target_vocab.char2id),
                                           embedding_dim=self.char_embedding_size,
                                           padding_idx=self.target_vocab.char2id['<pad>'])
        ### END YOUR CODE

    def forward(self, input: torch.Tensor, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        input_emb = self.decoderCharEmb(input)  # (length, batch, emb_size)
        last_hidden = dec_hidden
        s_t_list = []
        for input_emb_t in torch.split(input_emb, split_size_or_sections=1, dim=0):  # (1, batch, emb_size)
            output, new_hidden = self.charDecoder(input_emb_t,
                                                  last_hidden)  # (1, batch, hidden_size), (1, batch, hidden_size)
            h_t = new_hidden[0].permute(1, 0, 2)  # (batch, 1, hidden_size)
            s_t = self.char_output_projection(h_t)  # (batch, 1, self.vocab_size)
            s_t_list.append(s_t)
            last_hidden = new_hidden
        s_ts = torch.cat(s_t_list, dim=1).permute(1, 0, 2)  # (batch, length, self.vocab_size) -> (length, batch, ..)
        return s_ts, last_hidden

    ### END YOUR CODE

    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        scores, _ = self.forward(char_sequence[:-1], dec_hidden)  # scores: (length-1, batch, vocab_size)
        criterion = nn.CrossEntropyLoss(ignore_index=self.target_vocab.char2id['<pad>'], reduction='sum')
        scores = scores.permute(1, 2, 0)  # (batch, vocab_size, length-1)
        target = char_sequence[1:].permute(1, 0)  # (batch, length-1)
        loss = criterion(scores, target)
        return loss
        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        output_word = []
        decoded_words = []
        batch_size = initialStates[0].shape[1]
        current_char = torch.tensor([[self.target_vocab.start_of_word] * batch_size], device=device)  # (1, batch)
        hidden = initialStates
        for _ in range(max_length):
            scores, hidden = self.forward(current_char, hidden)  # scores: (1, batch, vocab_size)
            current_char = torch.argmax(scores, dim=2)  # (1, batch)
            output_word += [current_char]
        # output_word: [(1,batch), (1,batch), ..., (1, batch)] -> (max_length, batch) -> (batch, max_length)
        output_word = torch.transpose(torch.cat(output_word, dim=0), 0, 1)
        for batch in range(batch_size):
            word = ""
            for idx in range(max_length):
                ch_index = output_word[batch, idx].item()
                if ch_index == self.target_vocab.end_of_word:
                    break
                word += self.target_vocab.id2char[ch_index]
            decoded_words.append(word)
        return decoded_words
    ### END YOUR CODE
