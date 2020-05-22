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
from model import MultiModel
from model import BiLSTMClassifier
from utils import do_eval_epoch, fit

np.random.seed(42)

if torch.cuda.is_available():
    from torch.cuda import FloatTensor, LongTensor

    device = torch.device('cuda')
else:
    from torch import FloatTensor, LongTensor

    device = torch.device('cpu')

np.random.seed(42)

kp_toloka_result_path = "./KP_reviews/kp_results_filtered.tsv"
ivi_toloka_result_path = "./KP_reviews/ivi_result_filtered.tsv"
union_toloka_result_path = "./KP_reviews/union_results.tsv"
union_toloka_result_proc_path = "./KP_reviews/union_results_proc.tsv"

kp_golden_path = "./KP_reviews/kp_golden_tasks.tsv"
ivi_golden_path = "./KP_reviews/ivi_golden.tsv"
union_golden_path = "./KP_reviews/union_golden.tsv"
union_golden_proc_path = "./KP_reviews/union_golden_proc.tsv"
union_golden_proc_path2 = "./KP_reviews/union_golden_fixed_proc.tsv"


def pre_label(label):
    if label == 'yes':
        return 1
    else:
        return 0


def pre_score(score):
    if score == 'pos':
        return 1.0
    elif score == 'neg':
        return -1.0
    else:
        return 0.0


TEXT = data.Field(sequential=True, batch_first=False)
PLOT_LABEL = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_label, dtype=torch.float)
IMAGE_LABEL = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_label, dtype=torch.float)
MUSIC_LABEL = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_label, dtype=torch.float)
ACTORS_LABEL = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_label, dtype=torch.float)
PLOT_SCORE = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_score, dtype=torch.float)
IMAGE_SCORE = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_score, dtype=torch.float)
MUSIC_SCORE = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_score, dtype=torch.float)
ACTORS_SCORE = data.Field(sequential=False, use_vocab=False, is_target=True, preprocessing=pre_score, dtype=torch.float)


def get_dataset(path):
    return data.TabularDataset(path=path, format='tsv', fields=
    [('input_id', None), ('text', TEXT), ('output_id', None), ('plot_label', PLOT_LABEL),
     ('image_label', IMAGE_LABEL), ('music_label', MUSIC_LABEL), ('actors_label', ACTORS_LABEL),
     ('plot_score', PLOT_SCORE), ('image_score', IMAGE_SCORE), ('music_score', MUSIC_SCORE),
     ('actors_score', ACTORS_SCORE), ('name0', None)], skip_header=True)


train, val = get_dataset(union_toloka_result_proc_path).split()
golden_train = get_dataset(union_golden_proc_path2)
TEXT.build_vocab(train, max_size=30000)

model_path = "./models/model"
rnn_model = MultiModel(model=BiLSTMClassifier(300, len(TEXT.vocab.stoi), 256, 2).to(device))
# rnn_model.load_state_dict(torch.load(model_path))


batch_size = 32
train_iter, val_iter = data.BucketIterator.splits(
    (train, val), sort_key=lambda x: len(x.text),
    batch_sizes=(batch_size, batch_size), device=device)
golden_iter = data.BucketIterator(golden_train, sort_key=lambda x: len(x.text), batch_size=batch_size, device=device)

criterion_cls = nn.BCEWithLogitsLoss().to(device)
criterion_scores = nn.MSELoss(reduction='none').to(device)
criterion_scores_l1 = nn.L1Loss(reduction='none').to(device)

rnn_model = MultiModel(model=BiLSTMClassifier(300, len(TEXT.vocab.stoi), 256, 2).to(device))

optimizer = optim.Adam([param for param in rnn_model.model.parameters() if param.requires_grad])
fit(rnn_model, criterion_cls, criterion_scores, optimizer, train_iter, epochs_count=30, val_data=val_iter)
torch.save(rnn_model.model.state_dict(), model_path)

do_eval_epoch(rnn_model, None, criterion_scores_l1, val_iter)
