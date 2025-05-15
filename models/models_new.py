import os
import sys
# Add the TSL directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'TSL'))
# sys.path.append(f'Time-Series-Library-main/')
# sys.path.append(f'Time-Series-Library-main/layers/')
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from typing import Tuple
from TSL.layers.Transformer_EncDec import Encoder, EncoderLayer
from TSL.layers.SelfAttention_Family import FullAttention, AttentionLayer
from TSL.layers.Embed import DataEmbedding_inverted
from TSL.layers.Pyraformer_EncDec import Encoder as Pyra_Encoder

class LSTM(nn.Module):
    def __init__(self, configs):
        super(LSTM, self).__init__()
        self.hidden_size = configs.hidden_size
        self.num_layers = configs.num_layers
        self.lstm = nn.LSTM(configs.input_size, configs.hidden_size, configs.num_layers, batch_first=True)
        self.fc = nn.Linear(configs.hidden_size, configs.output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)  
        final_out = self.fc(out)  
        return out, final_out.squeeze()
    
class EALSTM(nn.Module):
    def __init__(self, configs):
        super(EALSTM, self).__init__()
        self.configs = configs
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.input_size_dyn = configs.enc_in - configs.input_size_stat
        self.input_size_stat = configs.input_size_stat
        self.hidden_size = configs.hidden_size
        self.batch_first = configs.batch_first
        self.initial_forget_bias = configs.initial_forget_bias

        self.weight_ih = nn.Parameter(torch.FloatTensor(self.input_size_dyn, 3 * self.hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(self.hidden_size, 3 * self.hidden_size))
        self.weight_sh = nn.Parameter(torch.FloatTensor(self.input_size_stat, self.hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * self.hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(self.hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        h_n, c_n = [], []
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))
        bias_s_batch = (self.bias_s.unsqueeze(0).expand(batch_size, *self.bias_s.size()))
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        for t in range(seq_len):
            h_0, c_0 = h_x
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)
            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class MyEALSTM(nn.Module):
    def __init__(self, configs):
        super(MyEALSTM, self).__init__()
        self.configs = configs
        self.device = configs.device
        self.batch_size = configs.batch_size
        self.input_size_dyn = configs.enc_in - configs.input_size_stat
        self.input_size_stat = configs.input_size_stat
        self.hidden_size = configs.hidden_size
        self.batch_first = configs.batch_first
        self.initial_forget_bias = configs.initial_forget_bias
        self.dropout_rate = configs.dropout
        self.concat_static = configs.concat_static
        self.no_static = configs.no_static
        self.lstm = EALSTM(configs=configs)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.fc = nn.Linear(self.hidden_size, configs.output_size)  # Project to 20 features

    def forward(self, x: torch.Tensor):
        x_d = x[:, :, self.input_size_stat:]
        x_s = x[:, 0, :self.input_size_stat]
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)

        h_n = self.dropout(h_n)
        out = self.fc(h_n)  # Shape: (batch, seq, 20)
        return h_n, out
    
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # print('before tcn', x.shape)
        out = self.net(x.unsqueeze(2)).squeeze(2)
        # print('after tcn', out.shape)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, config):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(config.num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            layers += [TemporalBlock(config.seq_len, config.seq_len, config.kernel_size, stride=1, dilation=dilation_size,
                                     padding=(config.kernel_size-1) * dilation_size, dropout=config.dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    
class MultiTCN(nn.Module):
    def __init__(self, config):
        super(MultiTCN, self).__init__()
        self.tcn_blocks = nn.ModuleList([
            TemporalConvNet(config) for _ in range(config.input_size)
        ])
        self.mlp = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.output_size)  # Project to 20 features
        )

    def forward(self, x):
        batch_size, seq_len, num_features = x.shape

        tcn_outputs = []
        for i in range(num_features):
            feature_x = x[:, :, i]
            tcn_output = self.tcn_blocks[i](feature_x)
            tcn_outputs.append(tcn_output)

        combined_tcn_output = torch.stack(tcn_outputs, dim=-1)
        final_output = self.mlp(combined_tcn_output)  # Shape: (batch, seq, 20)
        return combined_tcn_output, final_output

class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(config.input_size, config.hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=5),
            num_layers=config.num_layers
        )
        self.fc_out = nn.Linear(config.hidden_size, config.output_size)  # Project to 20 features

    def forward(self, src):
        src_emb = self.embedding(src)
        transformer_out = self.transformer(src_emb)
        output = self.fc_out(transformer_out)  # Shape: (batch, seq, 20)
        return transformer_out, output
    
class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.projection = nn.Linear(
                configs.d_model, configs.pred_len, bias=True)
        # Linear layer
        self.linear = nn.Linear(configs.enc_in, configs.c_out)
        self.relu = nn.ReLU()

    def forward(self,  x):
        # Normalization from Non-stationary Transformer
        _, _, N = x.shape

        # Embedding
        dec_out = self.enc_embedding(x, x_mark=None)
        dec_out, attns = self.encoder(dec_out, attn_mask=None)

        dec_emb = self.projection(dec_out).permute(0, 2, 1)[:, :, :N]
        dec_out = self.linear(dec_emb)
        # dec_out = self.relu(dec_out[:,:,0])
        dec_out = self.relu(dec_out)
        return dec_emb, dec_out
    
class Pyraformer(nn.Module):
    def __init__(self, configs, window_size=[4, 4], inner_size=5):
        super().__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.encoder = Pyra_Encoder(configs, window_size, inner_size)
        self.projection = nn.Linear((len(window_size) + 1) * self.d_model, self.pred_len * configs.enc_in)
        self.fc_out = nn.Linear(configs.enc_in, configs.output_size)  # Project to 20 features

    def forward(self, x_enc, x_mark_enc=None):
        enc_out = self.encoder(x_enc, x_mark_enc)[:, -1, :]
        dec_emb = self.projection(enc_out).view(enc_out.size(0), self.pred_len, -1)
        dec_out = self.fc_out(dec_emb[:, -self.pred_len:, :])  # Shape: (batch, seq, 20)
        return dec_emb[:, -self.pred_len:, :], dec_out



# class Config:
#     def __init__(self, batch_size=128, enc_in=11, n_hidden=8):
#         self.device = "cuda"
#         self.batch_size = batch_size
#         self.enc_in = enc_in
#         self.hidden_size = 50
#         self.batch_first = True
#         self.num_layers = 3
#         self.input_size = 11
#         self.output_size = 5  # Updated to 20
#         self.seq_len = 365
#         self.initial_forget_bias = 0
#         self.input_size_dyn = 5
#         self.input_size_stat = 10
#         self.dropout = 0.2
#         self.concat_static = False
#         self.no_static = False
#         self.num_channels = [1, 1, 1]
#         self.kernel_size = 5
#         self.d_model = 50
#         self.e_layers = 3
#         self.d_layers = 1
#         self.n_heads = 5
#         self.factor = 3
#         self.embed = "time"
#         self.freq = 'd'
#         self.d_ff = 4 * self.d_model
#         self.c_out = self.output_size  # Updated to 20
#         self.activation = 'relu'
#         self.output_attention = False
#         self.pred_len = 365
#         self.task_name = 'long_term_forecast'

# # Example usage
# model = LSTM(Config())
# print("Model is LSTM")
# x_enc = torch.randn(128, 365, 11)
# _, pred = model(x_enc)
# print(pred.shape)

# model = MyEALSTM(Config())
# print("Model is MyEALSTM")
# _, pred = model(x_enc)
# print(pred.shape)

# model = MultiTCN(Config())
# print("Model is MultiTCN")
# _, pred = model(x_enc)
# print(pred.shape)

# model = Transformer(Config())
# print("Model is Transformer")
# _, pred = model(x_enc)
# print(pred.shape)

# model = iTransformer(Config())
# print("Model is iTransformer")
# _, pred = model(x_enc)
# print(pred.shape)

# model = Pyraformer(Config())
# print("Model is Pyraformer")
# _, pred = model(x_enc)
# print(pred.shape)