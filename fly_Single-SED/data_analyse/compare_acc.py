import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
densenet_valid_accuracy = np.load('../result/compare/densenet_valid_accuracy.npy')
inception_valid_accuracy = np.load('../result/compare/inception_valid_accuracy.npy')
resnet_valid_accuracy = np.load('../result/compare/resnet_valid_accuracy.npy')
bilstm_valid_accuracy = np.load('../result/compare/bilstm_valid_accuracy.npy')
crnn_valid_accuracy = np.load('../result/compare/crnn_valid_accuracy.npy')
transformer_valid_accuracy = np.load('../result/compare/transformer_valid_accuracy.npy')
conformer_valid_accuracy = np.load('../result/compare/conformer_valid_accuracy.npy')
crnn1_valid_accuracy = np.load('../result/compare/crnn1_valid_accuracy.npy')
cnn_valid_accuracy = np.load('../result/compare/cnn_valid_accuracy.npy')

# 创建 x 轴数据（假设三个文件的长度相同）
x = np.arange(len(densenet_valid_accuracy))


# 绘制折线图
plt.plot(x, cnn_valid_accuracy, linestyle='--', marker='o', fillstyle='none', color='black', label='CNN')
plt.plot(x, inception_valid_accuracy, linestyle='--', marker='s', color='red', label='Inception')
plt.plot(x, resnet_valid_accuracy, linestyle=':', marker='^', color='green', label='ResNet')
plt.plot(x, densenet_valid_accuracy, linestyle='-', marker='o', color='blue', label='DenseNet')
plt.plot(x, bilstm_valid_accuracy, linestyle='-.', marker='v', color='purple', label='BiLSTM')
plt.plot(x, crnn1_valid_accuracy, linestyle='-.', marker='>', color='brown', label='CRNN(CNN+BiLSTM)')
plt.plot(x, crnn_valid_accuracy, linestyle='-', marker='*', color='orange', label='CRNN(DenseNet+BiLSTM)')
plt.plot(x, transformer_valid_accuracy, linestyle='--', marker='D', color='cyan', label='Transformer')
plt.plot(x, conformer_valid_accuracy, linestyle=':', marker='x', color='magenta', label='Conformer')





# 添加标题和标签
plt.title('Validation Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# 添加图例
plt.legend()

# 显示图形
plt.show()
