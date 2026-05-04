import pandas as pd
import numpy as np

print('加载数据...')
train_df = pd.read_csv('train.csv')
X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df['label'].values

print(f'训练集: {X_train.shape}')

print('创建模型...')
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print('训练模型...')
model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=1)

print('保存模型...')
model.save('model.h5')
print('模型已保存: model.h5')