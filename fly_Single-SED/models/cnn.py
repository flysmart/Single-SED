import torch.nn as nn
class CNN(nn.Module):
    def __init__(self, num_chanel, num_classes):
        super(CNN, self).__init__()
        # 卷积层定义
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]
        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = num_chanel if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        self.cnn = cnn
        self.fc = nn.Linear(nm[-1], num_classes)

    def forward(self, x):
        #print(x.size())
        x = self.cnn(x)
        #print(x.size())
        x = x.permute(0, 2, 3, 1)
        #print(x.size())
        batch_size, _, _, num_features = x.size()
        x = x.view(batch_size, -1, num_features)
        #print(x.size())
        x = x.mean(dim=1)  # 取平均值以获得一个二维张量
        #print(x.size())
        x = self.fc(x)
        #print(x.size())
        return x

