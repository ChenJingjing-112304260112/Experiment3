import pandas as pd
import numpy as np

print('加载数据...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df['label'].values
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'训练集: {X_train.shape}')
print(f'测试集: {X_test.shape}')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 创建模型1
print('\n训练模型1...')
model1 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model1.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=1)
pred1 = model1.predict(X_test)

# 创建模型2
print('\n训练模型2...')
model2 = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model2.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.2, verbose=1)
pred2 = model2.predict(X_test)

# 创建模型3
print('\n训练模型3...')
model3 = Sequential([
    Conv2D(16, (5,5), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(10, activation='softmax')
])
model3.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model3.fit(X_train, y_train, batch_size=64, epochs=12, validation_split=0.2, verbose=1)
pred3 = model3.predict(X_test)

# 集成预测
print('\n集成预测...')
ensemble_pred = (pred1 + pred2 + pred3) / 3
predicted_labels = np.argmax(ensemble_pred, axis=1)

# 保存提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('keras_cnn_submission.csv', index=False)

print('完成！提交文件已更新: keras_cnn_submission.csv')