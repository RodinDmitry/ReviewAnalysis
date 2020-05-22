import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor
    device = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor
    device = torch.device('cpu')


np.random.seed(42)

class BiLSTMClassifier(nn.Module):
    def __init__(self, emb_dim, vocab_size, lstm_hidden_dim, lstm_layers_count):
        super().__init__()
        self._embed = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
        self.rnn = nn.LSTM(emb_dim, lstm_hidden_dim, num_layers=lstm_layers_count, bidirectional=True, dropout=0.5)
        self.attention = nn.Sequential(nn.Linear(lstm_hidden_dim * 2, 4))
        self.out_layer_cls = nn.Linear(lstm_hidden_dim * 2, 1)
        self.out_layer_scores = nn.Linear(lstm_hidden_dim * 2, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        mask = (inputs == 1).float() * -10000.0
        mask = mask.unsqueeze(-1).repeat_interleave(4, dim=-1)
        outputs = self._embed(inputs)
        outputs, _ = self.rnn(outputs)
        outputs = self.dropout(outputs)
        attn_scores = self.attention(outputs) + mask
        attn_scores_ = F.softmax(attn_scores, -2)
        attn_scores = attn_scores_.transpose(1, 0).transpose(1, 2)
        outputs = outputs.transpose(1, 0)
        outputs = torch.bmm(attn_scores, outputs) # B 4 hidden
        outputs_cls = self.out_layer_cls(outputs).squeeze(-1)
        outputs_scores = self.out_layer_scores(outputs).squeeze(-1)
        return outputs_cls, outputs_scores, attn_scores_

class BiLSTMClassifier2(nn.Module):
    def __init__(self, emb_dim, vocab_size, lstm_hidden_dim, lstm_layers_count):
        super().__init__()
        self._embed = nn.Embedding(vocab_size, emb_dim, padding_idx=1)
        self.rnn = nn.LSTM(emb_dim, lstm_hidden_dim, num_layers=lstm_layers_count, bidirectional=True, dropout=0.5)
        self.attention = nn.Sequential(nn.Linear(lstm_hidden_dim * 2, 8), nn.Dropout(0.5))
        self.out_layer_cls = nn.Linear(lstm_hidden_dim * 2, 1)
        self.out_layer_scores = nn.Linear(lstm_hidden_dim * 2, 1)

    def forward(self, inputs):
        batch_size = inputs.shape[1]
        mask = (inputs == 1).float() * -100000.0
        outputs = self._embed(inputs)
        outputs, _ = self.rnn(outputs)
        attn_scores = self.attention(outputs)
        attn_scores = F.softmax(attn_scores, -2)
        attn_scores = attn_scores.transpose(1, 0).transpose(1, 2)
        outputs = outputs.transpose(1, 0)
        outputs = torch.bmm(attn_scores, outputs) # B 4 hidden
        outputs_cls = self.out_layer_cls(outputs).squeeze(-1)[:,:4]
        outputs_scores = self.out_layer_scores(outputs).squeeze(-1)[:,4:]
        return outputs_cls, outputs_scores


class AspectModel():
    def __init__(self, model):
        self.model = model

    def train(self, is_train):
        self.model.train(is_train)

    def __call__(self, batch):
        return self.model(batch.text)

    def build_target(self, batch):
        pass


class ActorsModel(AspectModel):
    def build_target(self, batch):
        return batch.actors_label, \
               batch.actors_score


class PlotModel(AspectModel):
    def build_target(self, batch):
        return batch.plot_label, \
               batch.plot_score


class MusicModel(AspectModel):
    def build_target(self, batch):
        return batch.music_label, \
               batch.music_score


class ImageModel(AspectModel):
    def build_target(self, batch):
        return batch.image_label, \
               batch.image_score


class MultiModel(AspectModel):
    def build_target(self, batch):
        cls_y = torch.cat((batch.image_label.unsqueeze(-1),
                           batch.plot_label.unsqueeze(-1),
                           batch.music_label.unsqueeze(-1),
                           batch.actors_label.unsqueeze(-1)), -1)

        scores_y = torch.cat((batch.image_score.unsqueeze(-1),
                              batch.plot_score.unsqueeze(-1),
                              batch.music_score.unsqueeze(-1),
                              batch.actors_score.unsqueeze(-1)), -1)
        return cls_y, scores_y