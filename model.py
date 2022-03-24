from transformer import *
from pooling_layers import *


class CNN_Trans_LID(nn.Module):
    def __init__(self, input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(CNN_Trans_LID, self).__init__()
        self.input_dim = input_dim
        self.feat_dim = feat_dim
        self.dropout = nn.Dropout(p=dropout)
        self.shared_TDNN = nn.Sequential(nn.Dropout(p=dropout),
                                         nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=True),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=True),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1, affine=True),
                                         )
        self.fc_xv = nn.Linear(1024, feat_dim)

        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)

        self.fc1 = nn.Linear(self.d_model * 2, self.d_model)
        self.fc2 = nn.Linear(self.d_model, self.d_model)
        self.fc3 = nn.Linear(self.d_model, n_lang)

    def mean_std_pooling(self, x, batchsize, seq_lens, mask_mean, weight_mean, mask_std, weight_unb):
        max_len = seq_lens[0]
        feat_dim = x.size(-1)
        if mask_mean is not None:
            assert mask_mean.size() == x.size()
            x.masked_fill_(mask_mean, 0)
        correct_mean = x.mean(dim=1).transpose(0, 1) * weight_mean
        correct_mean = correct_mean.transpose(0, 1)
        center_seq = x - correct_mean.repeat(1, 1, max_len).view(batchsize, -1, feat_dim)
        variance = torch.mean(torch.mul(torch.abs(center_seq) ** 2, mask_std), dim=1).transpose(0,1) \
                   * weight_unb * weight_mean
        std = torch.sqrt(variance.transpose(0, 1))
        return torch.cat((correct_mean, std), dim=1)

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = self.dropout(x)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        if std_mask_ is not None:
            stats = self.mean_std_pooling(output, batch_size, seq_len, mean_mask_, weight_mean,
                                          std_mask_, weight_unbaised)
        else:
            stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        output = F.relu(self.fc1(stats))
        output = F.relu(self.fc2(output))
        output = self.fc3(output)
        return output

class PHOLID(nn.Module):
    def __init__(self,input_dim, feat_dim,
                 d_k, d_v, d_ff, n_heads=8,
                 dropout=0.1, n_lang=3, max_seq_len=10000):
        super(PHOLID, self).__init__()
        self.input_dim = input_dim

        self.d_model = feat_dim * n_heads
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.shared_TDNN = nn.Sequential(nn.Dropout(p=dropout),
                                         nn.Conv1d(in_channels=input_dim, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1),
                                         nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1),
                                         nn.ReLU(),
                                         nn.BatchNorm1d(512, momentum=0.1))
        self.phoneme_proj = nn.Linear(512, 64)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.fc_xv = nn.Linear(1024, feat_dim)
        self.layernorm1 = LayerNorm(feat_dim)
        self.pos_encoding = PositionalEncoding(max_seq_len=max_seq_len, features_dim=feat_dim)
        self.layernorm2 = LayerNorm(feat_dim)
        self.attention_block1 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.attention_block2 = EncoderBlock(self.d_model, d_k, d_v, d_ff, n_heads, dropout=dropout)
        self.lid_clf = nn.Sequential(nn.Linear(self.d_model * 2, self.d_model),
                                     nn.ReLU(),
                                     nn.Linear(self.d_model, self.d_model),
                                     nn.ReLU(),
                                     nn.Linear(self.d_model, n_lang))

    def mean_std_pooling(self, x, batchsize, seq_lens, mask_mean, weight_mean, mask_std, weight_unb):
        max_len = seq_lens[0]
        feat_dim = x.size(-1)
        if mask_mean is not None:
            assert mask_mean.size() == x.size()
            x.masked_fill_(mask_mean, 0)
        correct_mean = x.mean(dim=1).transpose(0, 1) * weight_mean
        correct_mean = correct_mean.transpose(0, 1)
        center_seq = x - correct_mean.repeat(1, 1, max_len).view(batchsize, -1, feat_dim)
        variance = torch.mean(torch.mul(torch.abs(center_seq) ** 2, mask_std), dim=1).transpose(0, 1) \
                   * weight_unb * weight_mean
        std = torch.sqrt(variance.transpose(0, 1))
        return torch.cat((correct_mean, std), dim=1)

    def forward(self, x, seq_len, mean_mask_=None, weight_mean=None, std_mask_=None, weight_unbaised=None,
                atten_mask=None, eps=1e-5):
        batch_size = x.size(0)
        T_len = x.size(1)
        x = x.view(batch_size * T_len, -1, self.input_dim).transpose(-1, -2)
        x = self.shared_TDNN(x)
        pho_x = x.transpose(-1, -2)
        pho_out = self.phoneme_proj(pho_x)

        if self.training:
            shape = x.size()
            noise = torch.Tensor(shape)
            noise = noise.type_as(x)
            torch.randn(shape, out=noise)
            x += noise * eps

        seg_stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        embedding = self.fc_xv(seg_stats)
        embedding = embedding.view(batch_size, T_len, self.feat_dim)
        output = self.layernorm1(embedding)
        output = self.pos_encoding(output, seq_len)
        output = self.layernorm2(output)
        output = output.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output, _ = self.attention_block1(output, atten_mask)
        output, _ = self.attention_block2(output, atten_mask)
        if std_mask_ is not None:
            stats = self.mean_std_pooling(output, batch_size, seq_len, mean_mask_, weight_mean,
                                          std_mask_, weight_unbaised)
        else:
            stats = torch.cat((output.mean(dim=1), output.std(dim=1)), dim=1)
        output = self.lid_clf(stats)
        return output, pho_out.reshape(batch_size, T_len, -1, 64)




