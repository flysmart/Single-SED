import numpy as np
import matplotlib.pyplot as plt

# 读取五个验证准确数组文件并存储到一个列表中
train_losses = []
for i in range(1, 6):
    filename = f"../result/cnn/valid_accuracy_{i}.npy"
    loss_array = np.load(filename)
    train_losses.append(loss_array)

# 计算平均验证准确率
avg_train_loss = np.mean(train_losses, axis=0)
# 保存平均验证准确率到文件
output_filename = "../result/cnn/avg_valid_accuracy.npy"
np.save(output_filename, avg_train_loss)
# 打印最后一个 epoch 的 accuracy 值
last_epoch_loss = avg_train_loss[-1]
print("Last Epoch Accuracy:", last_epoch_loss)
# 绘制平均验证准确的折线图
epochs = np.arange(1, avg_train_loss.shape[0] + 1)
plt.plot(epochs, avg_train_loss)
plt.title('Average Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
