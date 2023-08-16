import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

# 读取五个混淆矩阵文件并存储到一个列表中
confusion_matrices = []
for i in range(1, 6):
    filename = f"../result/cnn/confusion_matrix_{i}.csv"
    cm = pd.read_csv(filename)
    confusion_matrices.append(cm)

# 计算平均混淆矩阵
avg_cm = sum(confusion_matrices) / len(confusion_matrices)

# 转换成DataFrame
df_avg_cm = pd.DataFrame(avg_cm)

# 绘制平均混淆矩阵的热力图
ax = sn.heatmap(df_avg_cm, annot=False, fmt='.20g', cmap="Blues")
ax.set_title('CNN Confusion Matrix')  # 标题
ax.set_xlabel('Predicted')  # x轴
ax.set_ylabel('True')  # y轴

# 显示图形
plt.show()
