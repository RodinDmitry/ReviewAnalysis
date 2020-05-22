import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import re
import nltk
from nltk.corpus import stopwords
import math
import time
import numpy as np

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor

    device = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor

    device = torch.device('cpu')

np.random.seed(42)

LABELS_SIZE = 4


def update_stat(logits, target, tp, fp, pos, mse_sum, mse_loss):
    size = logits.shape[0]
    labels = LABELS_SIZE if len(logits.shape) > 1 else 1
    for i in range(size):
        for j in range(labels):
            if labels == LABELS_SIZE:
                t = target[i, j].item() > 0.5
                l = logits[i, j].item() > 0.5
            else:
                t = target[i].item() > 0.5
                l = logits[i].item() > 0.5
            if t:
                pos[j] += 1
            if l and t:
                tp[j] += 1
            if not t and l:
                fp[j] += 1
    n_mse_loss = mse_loss.sum(0).cpu().detach().numpy()
    mse_sum += n_mse_loss


def calc_score(tp, fp, pos):
    eps1 = (tp + fp) == 0
    eps2 = pos == 0
    precision = tp / (tp + fp + eps1)
    recall = tp / (pos + eps2)
    eps3 = (precision + recall) == 0
    f1 = 2 * (precision * recall) / (precision + recall + eps3)
    return precision, recall, f1


def do_epoch(model, criterion_cls, criterion_scores, data_iter, label_size=LABELS_SIZE, optimizer=None):
    tp = np.zeros(label_size, dtype=np.int32)
    fp = np.zeros(label_size, dtype=np.int32)
    pos = np.zeros(label_size, dtype=np.int32)
    mse_sum = np.zeros(label_size, dtype=np.float)
    epoch_loss, epoch_tp, epoch_fp, epoch_fn = 0, 0, 0, 0

    is_train = not optimizer is None
    model.train(is_train)

    with torch.autograd.set_grad_enabled(is_train):
        for batch in data_iter:
            cls_y, scores_y = model.build_target(batch)
            result = model(batch)
            logits_cls, logits_scores = result[0], result[1]
            logits_scores = (torch.sigmoid(logits_scores) * 2) - 1
            # print(logits_cls.shape)
            # print(y_batch.shape)
            loss = 5 * criterion_cls(logits_cls, cls_y)
            mse_loss = criterion_scores(logits_scores, scores_y)
            mse_loss = mse_loss * cls_y
            loss += mse_loss.sum(-1).mean()
            epoch_loss += loss.item()

            if is_train:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            update_stat(torch.sigmoid(logits_cls), cls_y, tp, fp, pos, mse_sum, mse_loss)

    pr, rc, f1 = calc_score(tp, fp, pos)
    return epoch_loss, f1, mse_sum / pos


def fit(model, criterion_cls, criterion_scores, optimizer, train_data, epochs_count=1, val_data=None):
    for epoch in range(epochs_count):
        start_time = time.time()
        train_loss, train_f1, train_mse = do_epoch(
            model, criterion_cls, criterion_scores, train_data, optimizer=optimizer
        )

        output_info = '\rEpoch {} / {}, Epoch Time = {:.2f}s: Train Loss = {:.4f}'
        cls_info = 'F1 Image = {:.2%} Plot = {:.2%} Music = {:.2%} Actors = {:.2%}'
        mse_info = 'MSE Scores Image = {:.4f} Plot = {:.4f} Music = {:.4f} Actors = {:.4f}'
        if not val_data is None:
            val_loss, val_f1, val_mse = do_epoch(model, criterion_cls, criterion_scores, val_data, optimizer=None)

            epoch_time = time.time() - start_time
            output_info += ', Val Loss = {:.4f}'
            print(output_info.format(epoch + 1, epochs_count, epoch_time,
                                     train_loss, val_loss))
            print('\r\tTrain ' + cls_info.format(*list(train_f1)))
            print('\r\tValid ' + cls_info.format(*val_f1))
            print('\r\tTrain ' + mse_info.format(*train_mse))
            print('\r\tValid ' + mse_info.format(*val_mse))
        else:
            epoch_time = time.time() - start_time
            print(output_info.format(epoch + 1, epochs_count, epoch_time, train_loss))
            print('\r\tTrain ' + cls_info.format(*train_f1))
            print('\r\tTrain ' + mse_info.format(*train_mse))


def trasform_score(score):
    if score < -0.33:
        return -1
    elif -0.33 <= score < 0.33:
        return 0
    else:
        return 1


def update_stat_detection(classes, scores, target_scores, accuracy, size):
    for idx, score in enumerate(scores):
        cls_labels = classes[idx]
        t_score = target_scores[idx]
        for i in range(LABELS_SIZE):
            if cls_labels[i] > 0.5:
              size[i] += 1
              p = trasform_score(score[i])
              y = t_score[i]
              if p == y:
                  accuracy[i] += 1

def do_eval_epoch(model, criterion_cls, criterion_scores, data_iter):
    total_tp = np.zeros(LABELS_SIZE, dtype=np.int32)
    total_fp = np.zeros(LABELS_SIZE, dtype=np.int32)
    total_pos = np.zeros(LABELS_SIZE, dtype=np.int32)
    total_mse_sum = np.zeros(LABELS_SIZE, dtype=np.float)
    acc = np.zeros(LABELS_SIZE, dtype=np.int32)

    model.train(False)
    total_size = np.zeros(LABELS_SIZE, dtype=np.int32)
    with torch.autograd.set_grad_enabled(False):
        for batch in data_iter:
            cls_y, scores_y = model.build_target(batch)
            result = model(batch)
            logits_cls, logits_scores = result[0], result[1]
            logits_scores = (torch.sigmoid(logits_scores) * 2) - 1

            mse_loss = criterion_scores(logits_scores, scores_y)
            mse_loss = mse_loss * cls_y
            update_stat(torch.sigmoid(logits_cls), cls_y, total_tp, total_fp, total_pos, total_mse_sum, mse_loss)
            update_stat_detection(logits_cls, logits_scores, scores_y, acc, total_size)

    mse_info = '{:.4f} {:.4f} {:.4f} {:.4f}'
    print('Header           Image Plot Music Actors')
    pr, rc, f1 = calc_score(total_tp, total_fp, total_pos)
    print("Precision:      " + mse_info.format(*pr))
    print("Recall:         " + mse_info.format(*rc))
    print("F1:             " + mse_info.format(*f1))
    acc = acc / total_size
    print("Accuracy class: " + mse_info.format(*acc)
