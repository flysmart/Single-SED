import numpy as np
import matplotlib.pyplot as plt

# 读取五个训练损失值数组文件并存储到一个列表中
train_losses = []
for i in range(1, 6):
    filename = f"../result/cnn/train_losses_{i}.npy"
    loss_array = np.load(filename)
    train_losses.append(loss_array)

# 计算平均训练损失值
avg_train_loss = np.mean(train_losses, axis=0)
# 保存平均训练损失值到文件
output_filename = "../result/cnn/avg_train_losses.npy"
np.save(output_filename, avg_train_loss)
# 绘制平均训练损失值的折线图
epochs = np.arange(1, avg_train_loss.shape[0] + 1)
plt.plot(epochs, avg_train_loss)
plt.title('Average Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
