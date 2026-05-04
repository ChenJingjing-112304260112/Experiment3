import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

print('=== 超级CNN模型训练 ===')

# 加载数据
print('加载数据...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'训练集: {X_train.shape[0]} 样本')
print(f'测试集: {X_test.shape[0]} 样本')

# 创建超级CNN模型
model = Sequential([
    # 第一层
    Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.2),
    
    # 第二层
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.3),
    
    # 第三层
    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2,2),
    Dropout(0.4),
    
    # 全连接层
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 回调函数
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

# 训练模型
print('开始训练...')
history = model.fit(
    X_train, y_train, 
    batch_size=128, 
    epochs=30, 
    validation_split=0.15,
    callbacks=[lr_scheduler, early_stopping],
    verbose=1
)

# 评估模型
val_acc = max(history.history['val_accuracy'])
print(f'最高验证准确率: {val_acc:.4f}')

# 预测测试集
print('预测测试集...')
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('super_submission.csv', index=False)

print('完成！提交文件: super_submission.csv')
print(f'预计准确率: {val_acc:.4f}')