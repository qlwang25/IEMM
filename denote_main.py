# -*- coding: utf-8 -*-

# !/usr/bin/env python
# @Time    : 18-5-26
# @Author  : wang shen
# @File    : denote_main.py

"""
denote main is first step:
input is sentence,
output is denote and type
"""
import torch
from src.denote_model_two_cnn import DenoteModel
from src.util import precision_recall_fscore
from datetime import datetime
import os
from src.util import load, load_model, load_embed, arrangement
from src.util import generate_batch_sample_iter, standard_batch
from sklearn.metrics import accuracy_score
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Parameters:
    model_dir = "model"
    load_model = False

    batch_size_train = 32
    batch_size_test = 32

    nb_epoch = 100
    word_embed_size = 300

    learning_rate = 0.01
    leaning_rate_decay = True
    decay_epoch = 50

    process_data_path = "data/data.pickle"
    log_path = "data/denote_log_information.txt"
    embedding_path = "data/word2vec100.txt"

    use_cuda = True
    device = torch.device("cpu")

    # word vector dim == word_embed_size
    pre_word_embed = False
    embedding = None
    word2id, type2id = None, None
    id2word, id2type = None, None


def train_epoch(epoch, model, data_set, opt, batch_size, device):
    global global_step
    model.train()
    e_loss, nb, f = 0.0, 0, 0
    iter_denote, iter_kind = [], []
    iter_denote_prd, iter_kind_prd = [], []
    for samples in generate_batch_sample_iter(data_set, batch_size):
        sent, mask, denote, kind = standard_batch(samples, "denote")
        global_step += 1
        nb += 1
        print("--------------------------------------------")
        print("train---epoch: %d, learn rate: %.6f, global step: %d" % (epoch, opt.param_groups[0]["lr"], global_step))

        s_var = torch.Tensor(sent).long().to(device)
        m_var = torch.Tensor(mask).long().to(device)
        d_var = torch.Tensor(denote).long().to(device)
        k_var = torch.Tensor(kind).long().to(device)
        ips = (s_var, m_var, d_var, k_var)

        loss_denote = model.get_denote(ips, "loss")
        loss_kind = model.get_kind(ips, "loss")

        prd_denote = model.get_denote(ips, "prediction")
        prd_kind = model.get_kind(ips, "prediction")

        opt.zero_grad()
        torch.add(loss_denote, 0.5, loss_kind).backward()
        opt.step()

        e_loss += loss_denote.item()
        print("denote loss: %.8f, kind loss: %.8f" % (loss_denote.item(), loss_kind.item()))

        iter_kind.extend(np.reshape([node["type_idx"] for node in samples], (-1)).tolist())
        iter_kind_prd.extend(prd_kind.tolist())
        # p, r, f = precision_recall_fscore(iter_kind, iter_kind_prd, average="micro", flag="kind", sklearn=True)
        # print("micro type---P: {:.6f}, R: {:.6f}, F: {:.6f}".format(p, r, f))
        p, r, f = precision_recall_fscore(iter_kind, iter_kind_prd, average="macro", flag="kind", sklearn=True)
        print("macro type---P: {:.6f}, R: {:.6f}, F: {:.6f}".format(p, r, f))
        # print("type accuracy: %.6f" % accuracy_score(iter_kind, iter_kind_prd))

        iter_denote, iter_denote_prd = arrangement(iter_denote, iter_denote_prd, samples, prd_denote.tolist(), "denote")
        p, r, f = precision_recall_fscore(iter_denote, iter_denote_prd, average="macro", flag="denote", sklearn=True)
        print("macro denote---P: {:.6f}, R: {:.6f}, F: {:.6f}".format(p, r, f))
        # p, r, f = precision_recall_fscore(iter_denote, iter_denote_prd, flag="denote")
        # print("denote---P: {:.6f}, R: {:.6f}, F: {:.6f}".format(p, r, f))

    del iter_denote, iter_denote_prd, iter_kind, iter_kind_prd
    return e_loss / nb, f


def eval_epoch(epoch, model, data_set, batch_size, device, log_path=None):
    global global_step, collect_denote, collect_kind, collect_denote_sklearn
    model.eval()
    e_loss, nb = 0.0, 0
    p_k, r_k, f_k = 0.0, 0.0, 0.0
    p_d, r_d, f_d = 0.0, 0.0, 0.0
    p_d_s, r_d_s, f_d_s = 0.0, 0.0, 0.0
    iter_denote, iter_kind = [], []
    iter_denote_prd, iter_kind_prd = [], []
    for samples in generate_batch_sample_iter(data_set, batch_size, False, False):
        sent, mask, denote, kind = standard_batch(samples, "denote")
        nb += 1
        print("--------------------------------------------")
        print("eval---epoch: {:d}, global step: {:d}".format(epoch, global_step))

        s_var = torch.Tensor(sent).long().to(device)
        m_var = torch.Tensor(mask).to(device)
        d_var = torch.Tensor(denote).long().to(device)
        k_var = torch.Tensor(kind).long().to(device)
        ips = (s_var, m_var, d_var, k_var)

        loss_denote = model.get_denote(ips, "loss")
        loss_kind = model.get_kind(ips, "loss")

        prd_denote = model.get_denote(ips, "prediction")
        prd_kind = model.get_kind(ips, "prediction")

        e_loss += (loss_denote.item() + loss_kind.item() * 0.5)
        print("denote loss: %.8f, kind loss: %.8f" % (loss_denote.item(), loss_kind.item()))

        iter_kind.extend(np.reshape([node["type_idx"] for node in samples], (-1)).tolist())
        iter_kind_prd.extend(prd_kind.tolist())
        p_k, r_k, f_k = precision_recall_fscore(iter_kind, iter_kind_prd, average="macro", flag="kind", sklearn=True)
        print("type---P: %.6f, R: %.6f, F: %.6f" % (p_k, r_k, f_k))
        # print("type accuracy: %.6f" % accuracy_score(iter_kind, iter_kind_prd))

        iter_denote, iter_denote_prd = arrangement(iter_denote, iter_denote_prd, samples, prd_denote.tolist(), "denote")
        p_d_s, r_d_s, f_d_s = precision_recall_fscore(iter_denote, iter_denote_prd, average="macro",
                                                      flag="denote", sklearn=True)
        print("micro denote---P: {:.6f}, R: {:.6f}, F: {:.6f}".format(p_d_s, r_d_s, f_d_s))
        # p_d, r_d, f_d = precision_recall_fscore(iter_denote, iter_denote_prd, flag="denote")
        # print("denote---P: {:.6f}, R: {:.6f}, F: {:.6f}".format(p_d, r_d, f_d))

    del iter_denote, iter_denote_prd, iter_kind, iter_kind_prd

    if log_path:
        collect_denote.append([epoch, p_d, r_d, f_d])
        collect_kind.append([epoch, p_k, r_k, f_k])
        collect_denote_sklearn.append([epoch, p_d_s, r_d_s, f_d_s])
        collect_denote = sorted(collect_denote, key=lambda c: c[3], reverse=True)[:5]
        collect_denote_sklearn = sorted(collect_denote_sklearn, key=lambda c: c[3], reverse=True)[:5]
        collect_kind = sorted(collect_kind, key=lambda c: c[3], reverse=True)[:5]
        with open(log_path, "w") as fp:
            fp.write("epoch\tprecision\trecall\tf1\n")
            for d in collect_denote:
                fp.write("%d\t%.6f\t%.6f\t%.6f\n" % (d[0], d[1], d[2], d[3]))
            fp.write("\n")
            fp.write("\n")
            for k in collect_kind:
                fp.write("%d\t%.6f\t%.6f\t%.6f\n" % (k[0], k[1], k[2], k[3]))
            fp.close()

    else:
        return e_loss / nb


def train(train_set, dev_set, test_set, pars):
    model_dir = pars.model_dir + "/" + DenoteModel.__name__ + "_" + str(datetime.now()).split('.')[0].split()[0]
    print(model_dir)
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)

    model = DenoteModel(pars).to(pars.device)

    model_par = filter(lambda p: p.requires_grad, model.parameters())
    count = np.sum([np.prod(p.shape) for p in model.parameters()])
    print("model parameter num: %d" % count)

    f, loss = 0.8, 1.0
    opt = torch.optim.SGD(model_par, pars.learning_rate, 0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min", 0.88, 5, min_lr=1e-7)
    for epoch in range(1, pars.nb_epoch+1):
        e_loss, e_f1 = train_epoch(epoch, model, train_set, opt, pars.batch_size_train, pars.device)

        if pars.leaning_rate_decay and epoch > pars.decay_epoch:
            dev_loss = eval_epoch(epoch, model, dev_set, pars.batch_size_test, pars.device, None)
            scheduler.step(dev_loss)

        eval_epoch(epoch, model, test_set, pars.batch_size_test, pars.device, pars.log_path)

        if loss > e_loss and f < e_f1:
            loss = e_loss
            f = e_f1
            print("save model. loss is %.6f, f value is %.6f" % (loss, f))
            # save_model(model, epoch, loss, model_dir)


def test(data_set, pars):
    if not pars.load_model:
        print(" load model is false")
        return

    _dir = os.path.join(pars.model_dir, sorted(os.listdir(pars.model_dir))[-1])
    path = os.path.join(_dir, sorted(os.listdir(_dir))[-1])
    print("load model path: {:s}".format(path))

    model = load_model(DenoteModel, pars, path).to(pars.device)
    eval_epoch(-1, model, data_set, pars.batch_size_test, pars.device)


def main():
    start_time = str(datetime.now()).split('.')[0]
    pars = Parameters()

    # data = {"train": train_dataset, "dev": dev_dataset, "test": test_dataset,
    #        "word2id": word2id, "arg2id": arg2id, "type2id": type2id}
    data = load(pars.process_data_path)
    train_set, dev_set, test_set = data["train"] + data["dev"], data["dev"], data["test"]
    pars.word2id, pars.type2id = data["word2id"], data["type2id"]
    pars.id2word = {v: k for k, v in pars.word2id.items()}
    pars.id2type = {v: k for k, v in pars.type2id.items()}

    if pars.use_cuda and torch.cuda.is_available():
        pars.device = torch.device("cuda:0")

    if pars.pre_word_embed:
        pars.embedding = load_embed(pars.word2id, pars.word_embed_size, pars.embedding_path)

    print("train model")
    train(train_set, dev_set, test_set, pars)

    end_time = str(datetime.now()).split('.')[0]
    print("start time: {:s} ------>>>  end time: {:s} ".format(start_time, end_time))


if __name__ == '__main__':
    global_step = 0
    collect_denote, collect_denote_sklearn, collect_kind = [], [], []
    main()
    print("kind", collect_kind[0])
    print("denote", collect_denote[0])
    print("sklearn denote", collect_denote_sklearn[0])
