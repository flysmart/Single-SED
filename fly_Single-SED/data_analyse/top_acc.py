import numpy as np

# 存储最佳准确率的列表
best_accuracies = []

# 读取.npy文件并计算最佳准确率
for i in range(1, 6):
    filename = f"../result/conformer/valid_accuracy_{i}.npy"
    accuracy = np.load(filename)  # 读取.npy文件
    best_accuracy = np.max(accuracy)  # 计算最佳准确率
    best_accuracies.append(best_accuracy)  # 将最佳准确率添加到列表中
    print(f"第 {i} 次训练的最佳准确率: {best_accuracy}")

# 计算平均值
average_accuracy = np.mean(best_accuracies)
print("平均准确率: ", average_accuracy)
