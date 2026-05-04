import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, AveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

print('=== 强力集成学习 ===')

# 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'训练集: {X_train.shape[0]} 样本')
print(f'测试集: {X_test.shape[0]} 样本')

def create_model_a():
    """模型A: 宽卷积网络"""
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_b():
    """模型B: 深卷积网络"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.2),
        
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3,3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Dropout(0.3),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_c():
    """模型C: 残差风格网络"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_d():
    """模型D: 使用AveragePooling"""
    model = Sequential([
        Conv2D(48, (5,5), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        AveragePooling2D(2,2),
        
        Conv2D(96, (3,3), activation='relu'),
        BatchNormalization(),
        AveragePooling2D(2,2),
        
        Conv2D(192, (3,3), activation='relu'),
        BatchNormalization(),
        AveragePooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 训练多个模型
models = [create_model_a, create_model_b, create_model_c, create_model_d]
predictions = []

for i, create_func in enumerate(models, 1):
    print(f'\n训练模型 {i}...')
    model = create_func()
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    model.fit(X_train, y_train, batch_size=128, epochs=20, validation_split=0.15, 
              callbacks=[lr_scheduler], verbose=1)
    
    pred = model.predict(X_test)
    predictions.append(pred)
    print(f'模型 {i} 预测完成')

# 加权融合
print('\n融合预测结果...')
weights = [0.3, 0.3, 0.2, 0.2]  # 根据模型表现加权
ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
predicted_labels = np.argmax(ensemble_pred, axis=1)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('strong_ensemble_submission.csv', index=False)

print('完成！提交文件: strong_ensemble_submission.csv')