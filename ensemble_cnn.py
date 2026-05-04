import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau

print('集成学习 - 训练多个模型并融合预测...')

# 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

def create_model_1():
    """模型1: 基础CNN"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_2():
    """模型2: 更深的CNN"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1), padding='same'),
        Conv2D(32, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Conv2D(64, (3,3), activation='relu', padding='same'),
        Conv2D(64, (3,3), activation='relu', padding='same'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_model_3():
    """模型3: 带BatchNormalization的CNN"""
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 训练多个模型
models = []
predictions = []

print('训练模型1...')
model1 = create_model_1()
model1.fit(X_train, y_train, batch_size=64, epochs=15, validation_split=0.2, verbose=2)
pred1 = model1.predict(X_test)
predictions.append(pred1)
models.append(model1)

print('训练模型2...')
model2 = create_model_2()
model2.fit(X_train, y_train, batch_size=64, epochs=15, validation_split=0.2, verbose=2)
pred2 = model2.predict(X_test)
predictions.append(pred2)
models.append(model2)

print('训练模型3...')
model3 = create_model_3()
model3.fit(X_train, y_train, batch_size=64, epochs=15, validation_split=0.2, verbose=2)
pred3 = model3.predict(X_test)
predictions.append(pred3)
models.append(model3)

# 集成预测 - 加权平均
ensemble_pred = (pred1 + pred2 + pred3) / 3
predicted_labels = np.argmax(ensemble_pred, axis=1)

# 创建提交文件
submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('ensemble_submission.csv', index=False)

print('完成！集成学习提交文件: ensemble_submission.csv')