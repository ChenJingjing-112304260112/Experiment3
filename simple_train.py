import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

print('=== 开始训练 ===')

# 加载数据
print('1. 加载数据...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'   训练集: {X_train.shape[0]} 样本')
print(f'   测试集: {X_test.shape[0]} 样本')

# 创建模型
print('2. 创建模型...')
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print('3. 开始训练...')
for epoch in range(1, 11):
    print(f'   Epoch {epoch}/10')
    history = model.fit(X_train, y_train, batch_size=64, epochs=1, validation_split=0.2, verbose=2)
    train_acc = history.history['accuracy'][0]
    val_acc = history.history['val_accuracy'][0]
    print(f'   训练准确率: {train_acc:.4f}, 验证准确率: {val_acc:.4f}')

# 预测
print('4. 预测测试集...')
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# 创建提交文件
print('5. 创建提交文件...')
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('simple_submission.csv', index=False)

print('=== 训练完成 ===')
print('提交文件: simple_submission.csv')