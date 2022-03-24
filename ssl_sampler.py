import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import time


class CosineSimilarity_custom(nn.Module):
    def __init__(self, dim: int = 1, eps: float = 1e-8):
        super(CosineSimilarity_custom, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x1, x2):
        return 1 - F.cosine_similarity(x1, x2, self.dim, self.eps)


def get_output(outputs, seq_len):
    output_ = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        if i == 0:
            output_ = output
        else:
            output_ = torch.cat((output_, output), dim=0)
    return output_


def get_seg_label(label, seq_len):
    label_seg = 0
    for i in range(len(seq_len)):
        label_ = label[i].repeat(seq_len[i])
        if i == 0:
            label_seg = label[i].repeat(seq_len[i])
        else:
            label_seg = torch.cat((label_seg, label[i].repeat(seq_len[i])))
    return label_seg


def random_negative_samples_dict(output, output_labels, seq_len, n_lang):
    dict = {}
    seq_len = torch.LongTensor(seq_len)
    for i in range(n_lang):
        output_ = output[(output_labels != i)]
        seq_len_ = seq_len[(output_labels != i)]
        seq_len_ = seq_len_.tolist()
        dict[i] = get_output(output_, seq_len_)
    return dict


def get_anchors_positives(outputs, seq_len):
    output_ = 0
    positive = 0
    for i in range(len(seq_len)):
        length = seq_len[i]
        output = outputs[i, :length, :]
        perm_pos = [i - 1 if i == (length - 1) else i + 1 for i in range(length)]
        output_pos = outputs[i, perm_pos, :]
        if i == 0:
            output_ = output
            positive = output_pos
        else:
            output_ = torch.cat((output_, output), dim=0)
            positive = torch.cat((positive, output_pos), dim=0)
    return output_, positive


def positive_negative_sampler_general(output, labels, seq_len, n_lang):
    anchors, positives = get_anchors_positives(output, seq_len)
    negative_dict = random_negative_samples_dict(output, labels, torch.LongTensor(seq_len), n_lang)
    negatives = 0
    for idx, label in enumerate(labels):
        length = seq_len[idx]
        negatives_alters = negative_dict[label.item()]
        rand_perm = np.random.choice(negatives_alters.size(0), length)
        if idx == 0:
            negatives = negatives_alters[rand_perm]
        else:
            negatives = torch.cat((negatives, negatives_alters[rand_perm]), dim=0)
    return anchors, positives, negatives



class Phoneme_SSL_loss(nn.Module):
    def __init__(self, num_frames, num_sample=5):
        super(Phoneme_SSL_loss, self).__init__()
        self.all_ind = torch.LongTensor(list(range(num_frames)))
        self.num_sample = num_sample

    def get_output_phn(self, outputs, seq_len):
        output_ = 0
        for i in range(len(seq_len)):
            length = seq_len[i]
            output = outputs[i, :length, :, :]
            if i == 0:
                output_ = output
            else:
                output_ = torch.cat((output_, output), dim=0)
        return output_

    def forward(self, output, seq_len):
        output_seg = self.get_output_phn(output, seq_len)
        num_seg, num_frame, dim = output_seg.size()
        sim_pos = F.cosine_similarity(output_seg[:, :-1, :], output_seg[:, 1:, :], dim=-1).unsqueeze(-1)
        output_seg_group = output_seg.transpose(0, 1).reshape(num_frame, -1)
        random_index = [torch.randperm(num_frame - 3).tolist()[:self.num_sample] for i in range(num_frame-1)]
        output_frames = [
            output_seg_group[(self.all_ind != i - 1) * (self.all_ind != i) * (self.all_ind != i + 1), :][random_index[i],
            :] for i in self.all_ind[:-1]]
        negatives = torch.cat(output_frames, dim=0).reshape(-1, num_seg, dim).transpose(0, 1)
        anchors = output_seg[:, :-1, :].repeat(1, 1, self.num_sample).reshape(num_seg, -1, dim)
        sim_neg = F.cosine_similarity(anchors, negatives, dim=-1).reshape(-1, 19, self.num_sample)
        sim_all = torch.cat((sim_pos, sim_neg), dim=-1)
        loss_seg = torch.mean(torch.mean(-F.log_softmax(sim_all, dim=-1)[:,:, 0], dim=-1))
        return loss_seg

