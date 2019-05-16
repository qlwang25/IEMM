# -*- coding: utf-8 -*-

# !/usr/bin/env python
# @Time    : 18-6-19
# @Author  : wang shen
# @File    : denote_model_two_cnn.py

"""
cnn_2 + cnn_3 + multi-attention
"""
import torch.nn.functional as fun
import torch.nn as nn
import torch

_inf = -999.9


class DenoteModel(nn.Module):
    def __init__(self, pars):
        super(DenoteModel, self).__init__()
        self.device = pars.device
        self.dropout = 0.2
        self.word_size = len(pars.word2id)
        self.type_size = len(pars.type2id)
        self.word_embed_size = pars.word_embed_size

        self.word_lookup = nn.Embedding(self.word_size, self.word_embed_size)
        if pars.pre_word_embed:
            self.word_lookup.weight.data.copy_(torch.from_numpy(pars.embedding))

        self.word_out_channel = 128
        self.cnn_2 = nn.Conv1d(self.word_embed_size, self.word_out_channel, 2)
        self.cnn_3 = nn.Conv1d(self.word_embed_size, self.word_out_channel, 3)
        self.cnn_norm = nn.BatchNorm1d(self.word_out_channel)

        # attention
        self.f_linear = nn.Linear(2 * self.word_out_channel, 2 * self.word_out_channel)
        self.g_linear = nn.Linear(2 * self.word_out_channel, 2 * self.word_out_channel)
        self.h_linear = nn.Linear(2 * self.word_out_channel, 2 * self.word_out_channel)

        self.linear = nn.Linear(4 * self.word_out_channel, 2 * self.word_out_channel)

        self.denote_linear = nn.Linear(2 * self.word_out_channel, 2)

        # self.cnn_type = nn.Conv1d(2 * self.word_out_channel, self.type_size, 3)
        # self.type_norm = nn.BatchNorm1d(self.type_size)
        self.type_linear = nn.Linear(2 * self.word_out_channel, self.type_size)

        self.denote_loss = nn.NLLLoss(torch.Tensor([0.1, 0.9]))
        self.kind_loss = nn.NLLLoss()

    @staticmethod
    def batch_norm(norm, inp):
        inp = torch.transpose(inp, 1, 2)
        inp = inp.contiguous()
        out = norm(inp)
        out = torch.transpose(out, 1, 2)
        return out

    # @staticmethod
    # def self_attention(inp, mask):
    #     other = inp.clone()
    #     other = torch.transpose(other, 1, 2)
    #     _weights = torch.matmul(inp, other)
    #
    #     _mask = mask.clone()
    #     _mask = torch.unsqueeze(_mask, 1)
    #     _mask = _mask.expand_as(_weights)
    #     _mask = (1.0 - _mask) * _inf
    #
    #     weights = fun.softmax(_weights + _mask, 2)
    #     out = torch.matmul(weights, inp)
    #     return out

    def self_attention(self, inp, mask):
        f = self.f_linear(inp)
        g = self.g_linear(inp)
        h = self.h_linear(inp)

        g = torch.transpose(g, 1, 2)
        score = torch.matmul(f, g)

        _mask = mask.clone()
        _mask = torch.unsqueeze(_mask, 1)
        _mask = _mask.expand_as(score)
        _mask = (1.0 - _mask) * _inf

        weights = fun.softmax(score + _mask.float(), 2)

        out = torch.matmul(weights, h)
        return out

    def forward(self, inps):
        sent, mask, _, _ = inps
        sent_embed = self.word_lookup(sent)
        sent_embed = fun.dropout(sent_embed, self.dropout)

        cnn_ip = torch.transpose(sent_embed, 1, 2)

        padding = self.cnn_2.kernel_size[0] - 1
        cnn_ip_2 = fun.pad(cnn_ip, (0, padding))
        cnn_out_2 = self.cnn_2(cnn_ip_2)
        cnn_out_2 = torch.transpose(cnn_out_2, 1, 2)
        cnn_out_2 = fun.relu(cnn_out_2)
        out_2 = self.batch_norm(self.cnn_norm, cnn_out_2)

        padding = self.cnn_3.kernel_size[0] - 1
        cnn_ip_3 = fun.pad(cnn_ip, (0, padding))
        cnn_out_3 = self.cnn_3(cnn_ip_3)
        cnn_out_3 = torch.transpose(cnn_out_3, 1, 2)
        cnn_out_3 = fun.relu(cnn_out_3)
        out_3 = self.batch_norm(self.cnn_norm, cnn_out_3)

        out = torch.cat([out_2, out_3], 2)
        attention = self.self_attention(out, mask)
        attention = fun.tanh(attention)

        logits = torch.cat([out, attention], 2)
        logits = self.linear(logits)
        return logits

    def get_denote(self, inps, mode):
        sent, mask, denote, kind = inps
        out = self.forward(inps)
        logits = self.denote_linear(out)
        logits = fun.log_softmax(fun.relu(logits), 2)

        if mode == "loss":
            loss = self.denote_loss(torch.transpose(logits, 1, 2), denote)
            return loss

        elif mode == "prediction":
            prd_denote = torch.argmax(logits, 2)
            return prd_denote

        else:
            raise ValueError

    def get_kind(self, inps, mode):
        sent, mask, denote, kind = inps
        out = self.forward(inps)
        # out = torch.transpose(fun.relu(out), 1, 2)
        # cnn_out = self.cnn_type(out)
        # max_out = nn.MaxPool1d(cnn_out.size(2))(cnn_out)
        # max_out = torch.transpose(max_out, 1, 2)
        #
        # norm_out = self.batch_norm(self.type_norm, max_out)
        # logits = fun.log_softmax(torch.squeeze(norm_out), 1)

        out = torch.transpose(out, 1, 2)
        out = nn.MaxPool1d(out.size(2))(out)
        out = torch.squeeze(out)
        logits = fun.log_softmax(self.type_linear(out), 1)

        if mode == "loss":
            loss = self.kind_loss(logits, torch.squeeze(kind))
            return loss

        elif mode == "prediction":
            prd_kind = torch.argmax(logits, 1)
            return prd_kind

        else:
            raise ValueError
