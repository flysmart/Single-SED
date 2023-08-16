import torch
import torch.nn as nn
# 定义双向LSTM网络
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        dropout_prob = 0.1
        self.dropout_prob = dropout_prob
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # 双向LSTM输出会被拼接起来，因此乘以2
    def forward(self, x):
        #print(x.shape)  #x形状为[2,3,128,250]，【批量，通道数，特征维度，像素宽度】
        batch_size = x.size(0)
        # 调整输入张量的形状:chan_num=3,H=3*128=384
        #x = x.view(x.size(0), x.size(1) * x.size(2), x.size(3))
        #x = x.permute(0, 2, 1)  #x形状为【批量，像素宽度，特征维度】
        # 调整输入张量的形状:chan_num=1,H=128
        x = x[:, 0, :, :]
        x = x.squeeze(1)
        x = x.permute(0,2,1)
        #print(x.shape)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)# 双向LSTM，因此隐藏状态数量乘以2
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        #print(h0.shape)
        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))
        # 解码最后一个时刻的输出
        out = self.fc(out[:, -1, :])
        return out

