import pandas as pd
import numpy as np

print('=== 改进提交文件 ===')

# 读取现有提交文件
print('读取现有提交文件...')
submission = pd.read_csv('keras_cnn_submission.csv')
print(f'现有提交文件: {len(submission)} 行')

# 加载测试数据
test_df = pd.read_csv('test.csv')
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# 使用简单规则改进预测
print('应用改进规则...')
labels = submission['Label'].values

# 规则1: 如果图像几乎全黑，预测为0
threshold = 0.01  # 像素值阈值
for i in range(len(X_test)):
    if np.mean(X_test[i]) < threshold:
        labels[i] = 0

# 规则2: 检查一些常见的混淆情况
# 7 和 9 经常被混淆，检查顶部是否有开口
for i in range(len(X_test)):
    img = X_test[i].reshape(28, 28)
    # 如果当前预测是7，检查是否可能是9
    if labels[i] == 7:
        # 检查顶部是否有明显的横线（9的特征）
        top_row = img[2:5, :].mean()
        if top_row > 0.3:  # 顶部有明显内容，可能是9
            labels[i] = 9
    elif labels[i] == 9:
        # 检查底部是否完整（7的底部通常是开口的）
        bottom_row = img[23:26, :].mean()
        if bottom_row < 0.1:  # 底部几乎空白，可能是7
            labels[i] = 7

# 规则3: 区分 3 和 8
for i in range(len(X_test)):
    img = X_test[i].reshape(28, 28)
    if labels[i] == 3:
        # 检查中间是否有连接（8有两个圈）
        middle = img[10:18, 8:20]
        if middle.mean() > 0.4:  # 中间密度高，可能是8
            labels[i] = 8
    elif labels[i] == 8:
        # 检查中间是否断开（3中间是断开的）
        middle = img[12:16, 10:18]
        if middle.mean() < 0.2:  # 中间密度低，可能是3
            labels[i] = 3

# 规则4: 区分 0 和 6
for i in range(len(X_test)):
    img = X_test[i].reshape(28, 28)
    if labels[i] == 0:
        # 检查底部是否有尾巴（6有尾巴）
        bottom_right = img[20:26, 18:26]
        if bottom_right.mean() > 0.2:  # 右下角有内容，可能是6
            labels[i] = 6
    elif labels[i] == 6:
        # 检查底部是否没有尾巴（0是完整的圆）
        bottom_right = img[22:26, 20:26]
        if bottom_right.mean() < 0.1:  # 右下角空白，可能是0
            labels[i] = 0

# 更新提交文件
submission['Label'] = labels
submission.to_csv('keras_cnn_submission.csv', index=False)

print('完成！提交文件已更新')