import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print('开始训练快速改进版CNN模型...')

# 加载数据
print('加载数据...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 准备训练数据
x_train = train_df.drop('label', axis=1).values.astype('float32')
y_train = train_df['label'].values
x_test = test_df.values.astype('float32')

# 归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 重塑为28x28x1
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot编码
y_train = to_categorical(y_train, 10)

# 创建改进版CNN模型
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Conv2D(64, (3,3), activation='relu'),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.25),
    
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print('开始训练...')
history = model.fit(
    x_train, y_train,
    batch_size=64,
    epochs=15,
    validation_split=0.2,
    verbose=1
)

# 保存模型
model.save('quick_improved_model.h5')
print('模型已保存')

# 预测测试集
print('预测测试集...')
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('quick_improved_submission.csv', index=False)

print('提交文件已生成: quick_improved_submission.csv')
print('Done!')