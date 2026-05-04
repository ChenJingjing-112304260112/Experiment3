import numpy as np
import pandas as pd

print('测试数据加载...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f'训练数据形状: {train_df.shape}')
print(f'测试数据形状: {test_df.shape}')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df['label'].values
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'X_train形状: {X_train.shape}')
print(f'y_train形状: {y_train.shape}')
print(f'X_test形状: {X_test.shape}')

# 简单测试模型
print('\n测试模型创建...')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('训练模型...')
model.fit(X_train, y_train, batch_size=128, epochs=2, verbose=1)

print('预测测试集...')
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

print(f'预测结果形状: {predicted_labels.shape}')
print(f'前5个预测: {predicted_labels[:5]}')

# 创建测试提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('test_submission.csv', index=False)
print('测试提交文件已保存: test_submission.csv')