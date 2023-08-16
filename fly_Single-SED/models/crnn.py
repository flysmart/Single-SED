import torch
import torch.nn as nn
import torchvision.models as models

class CRNN(nn.Module):
    def __init__(self,  hidden_size, num_layers, num_classes):
        super(CRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 添加CNN模型，这里以DenseNet为例
        self.cnn = models.densenet201(pretrained=False)
        num_features = self.cnn.classifier.in_features
        self.cnn.classifier = nn.Identity()  # 替换DenseNet的分类器

        self.lstm = nn.LSTM(num_features, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        # 使用CNN提取特征
        features = self.cnn(x)
        features = features.view(features.size(0), -1)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(features.unsqueeze(1), (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)

        return out
