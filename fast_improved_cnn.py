import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

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

# 划分训练集和验证集
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print(f'训练集大小: {x_train.shape[0]}')
print(f'验证集大小: {x_val.shape[0]}')
print(f'测试集大小: {x_test.shape[0]}')

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
model.summary()

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

datagen.fit(x_train)

# 学习率调度器
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

# 训练模型
print('开始训练...')
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=64),
    epochs=30,
    validation_data=(x_val, y_val),
    callbacks=[lr_scheduler],
    verbose=1
)

# 保存模型
model.save('fast_improved_model.h5')
print('模型已保存为 fast_improved_model.h5')

# 绘制训练曲线
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练集Loss')
plt.plot(history.history['val_loss'], label='验证集Loss')
plt.title('Loss变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练集准确率')
plt.plot(history.history['val_accuracy'], label='验证集准确率')
plt.title('准确率变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('fast_improved_loss_curve.png', dpi=300, bbox_inches='tight')
print('训练曲线已保存为 fast_improved_loss_curve.png')

# 预测测试集
print('预测测试集...')
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('fast_improved_submission.csv', index=False)

print('提交文件已生成: fast_improved_submission.csv')
print('Done!')