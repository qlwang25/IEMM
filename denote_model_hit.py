# -*- coding: utf-8 -*-

# !/usr/bin/env python
# @Time    : 18-5-27
# @Author  : wang shen
# @File    : denote_model_hit.py

"""
bilstm + cnn_2 + cnn_3
"""
import torch.nn.functional as fun
import torch.nn as nn
import torch


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
        self.lstm = nn.LSTM(self.word_embed_size, self.word_out_channel, bidirectional=True)
        self.cnn_2 = nn.Conv1d(self.word_embed_size, self.word_out_channel, 2)
        self.cnn_3 = nn.Conv1d(self.word_embed_size, self.word_out_channel, 3)
        self.cnn_norm = nn.BatchNorm1d(self.word_out_channel)

        self.denote_linear = nn.Linear(2 * self.word_out_channel, 2)
        self.kine_linear = nn.Linear(2 * self.word_out_channel, self.type_size)

        self.loss = nn.NLLLoss()

    @staticmethod
    def get_pack_inputs(x, lengths):
        _, id_sort = torch.sort(lengths, 0, True)
        _, id_unsort = torch.sort(id_sort, 0)

        lengths = list(lengths[id_sort])
        x = torch.index_select(x, 0, id_sort)
        x = torch.transpose(x, 0, 1).contiguous()
        ips = nn.utils.rnn.pack_padded_sequence(x, lengths)
        return ips, id_unsort

    def get_pad_outputs(self, x, lengths, id_unsort):
        out = nn.utils.rnn.pad_packed_sequence(x)[0]

        out = torch.transpose(out, 0, 1).contiguous()
        out = torch.index_select(out, 0, id_unsort)

        if out.size(1) != lengths:
            padding = torch.zeros(out.size(0), lengths-out.size(1), out.size(2)).type(out.data.type())
            out = torch.cat([out, padding.to(self.device)], 1)

        return out

    def forward(self, inps):
        sent, mask, _, _ = inps
        sent_embed = self.word_lookup(sent)
        sent_embed = fun.dropout(sent_embed, self.dropout)

        lstm_ip, id_unsort = self.get_pack_inputs(sent_embed, mask.long().sum(1))
        lstm_out, _ = self.lstm(lstm_ip)
        lstm_out = self.get_pad_outputs(lstm_out, sent_embed.size(1), id_unsort)

        # cnn_ip = torch.transpose(sent_embed, 1, 2)
        # cnn_2_out = self.cnn_2(cnn_ip)
        # pool_2_out = nn.MaxPool1d(cnn_2_out.size(2))(cnn_2_out)
        # cnn_2_out = torch.transpose(pool_2_out, 1, 2)
        #
        # cnn_3_out = self.cnn_3(cnn_ip)
        # pool_3_out = nn.MaxPool1d(cnn_3_out.size(2))(cnn_3_out)
        # cnn_3_out = torch.transpose(pool_3_out, 1, 2)
        #
        # cnn_out = torch.cat([cnn_2_out, cnn_3_out], 2)
        # cnn_out = cnn_out.expand_as(lstm_out)
        #
        # out = torch.cat([lstm_out, cnn_out], 2)

        return lstm_out

    def get_denote(self, inps, mode):
        _, _, denote, _ = inps
        out = self.forward(inps)
        logits = self.denote_linear(out)
        logits = fun.log_softmax(logits, 2)

        if mode == "loss":
            loss = self.loss(torch.transpose(logits, 1, 2), denote)
            return loss

        elif mode == "prediction":
            prd_denote = torch.argmax(logits, 2)
            return prd_denote

        else:
            raise ValueError

    def get_kind(self, inps, mode):
        _, _, _, kind = inps
        out = self.forward(inps)
        out = torch.transpose(out, 1, 2)
        max_out = nn.MaxPool1d(out.size(2))(out)
        max_out = torch.squeeze(max_out)

        logits = fun.log_softmax(self.kine_linear(max_out), 1)

        if mode == "loss":
            loss = self.loss(logits, torch.squeeze(kind))
            return loss

        elif mode == "prediction":
            prd_kind = torch.argmax(logits, 1)
            return prd_kind

        else:
            raise ValueError
