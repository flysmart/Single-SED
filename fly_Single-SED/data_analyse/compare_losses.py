import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
densenet_train_losses = np.load('../result/compare/densenet_train_losses.npy')
inception_train_losses = np.load('../result/compare/inception_train_losses.npy')
resnet_train_losses = np.load('../result/compare/resnet_train_losses.npy')
bilstm_train_losses = np.load('../result/compare/bilstm_train_losses.npy')
crnn_train_losses = np.load('../result/compare/crnn_train_losses.npy')
transformer_train_losses = np.load('../result/compare/transformer_train_losses.npy')
conformer_train_losses = np.load('../result/compare/conformer_train_losses.npy')
crnn1_train_losses = np.load('../result/compare/crnn1_train_losses.npy')
cnn_train_losses = np.load('../result/compare/cnn_train_losses.npy')

# 创建 x 轴数据（假设三个文件的长度相同）
x = np.arange(len(densenet_train_losses))

# 绘制折线图
plt.plot(x, cnn_train_losses, linestyle='--', marker='o', fillstyle='none', color='black', label='CNN')
plt.plot(x, inception_train_losses, linestyle='--', marker='s', color='red', label='Inception')
plt.plot(x, resnet_train_losses, linestyle=':', marker='^', color='green', label='ResNet')
plt.plot(x, densenet_train_losses, linestyle='-', marker='o', color='blue', label='DenseNet')
plt.plot(x, bilstm_train_losses, linestyle='-.', marker='v', color='purple', label='BiLSTM')
plt.plot(x, crnn1_train_losses, linestyle='-.', marker='>', color='brown', label='CRNN(CNN+BiLSTM)')
plt.plot(x, crnn_train_losses, linestyle='-', marker='*', color='orange', label='CRNN(DenseNet+BiLSTM)')
plt.plot(x, transformer_train_losses, linestyle='--', marker='D', color='cyan', label='Transformer')
plt.plot(x, conformer_train_losses, linestyle=':', marker='x', color='magenta', label='Conformer')




# 添加标题和标签
plt.title('Training Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# 添加图例
plt.legend()

# 显示图形
plt.show()
