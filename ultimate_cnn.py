import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print('=== 终极CNN模型训练 ===')

# 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'训练集: {X_train.shape[0]} 样本')
print(f'测试集: {X_test.shape[0]} 样本')

# 创建终极CNN模型
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.2),
    
    # Block 2
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    # Block 3
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),
    
    # Classifier
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 数据增强
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.05,
    fill_mode='nearest'
)
datagen.fit(X_train)

# 回调函数
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# 训练模型
print('\n开始训练...')
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=128),
    epochs=40,
    validation_split=0.15,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# 评估
best_val_acc = max(history.history['val_accuracy'])
print(f'\n最佳验证准确率: {best_val_acc:.4f}')

# 预测
print('预测测试集...')
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# 保存提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('keras_cnn_submission.csv', index=False)

print(f'\n完成！提交文件已更新: keras_cnn_submission.csv')
print(f'预计Kaggle准确率: {best_val_acc:.4f}')