import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# 加载数据
print('Loading data...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)

# 创建CNN模型
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
print('Training model...')
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2)

# 打印loss数据
print('\n=== Training Loss Data ===')
print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12}")
print('-' * 35)
for i in range(len(history.history['loss'])):
    print(f"{i+1:<6} {history.history['loss'][i]:<12.6f} {history.history['val_loss'][i]:<12.6f}")

# 保存loss数据到文本文件
loss_data = pd.DataFrame({
    'Epoch': range(1, len(history.history['loss']) + 1),
    'Train Loss': history.history['loss'],
    'Val Loss': history.history['val_loss'],
    'Train Accuracy': history.history['accuracy'],
    'Val Accuracy': history.history['val_accuracy']
})
loss_data.to_csv('loss_data.csv', index=False)
print('\nLoss data saved to: loss_data.csv')

# 绘制并保存为不同格式
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
plt.title('Training and Validation Loss vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 保存为PNG格式
plt.savefig('loss_curve.png', dpi=100, bbox_inches='tight')
print('Loss curve saved as: loss_curve.png')

# 保存为JPG格式
plt.savefig('loss_curve.jpg', dpi=100, bbox_inches='tight')
print('Loss curve saved as: loss_curve.jpg')

print('\nDone!')