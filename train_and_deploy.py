import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical

print('=== 训练模型 ===')

# 加载数据
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = to_categorical(train_df['label'].values, 10)

print(f'训练集: {X_train.shape}')

# 创建模型
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
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练
model.fit(X_train, y_train, batch_size=128, epochs=15, validation_split=0.2, verbose=1)

# 保存模型
model.save('model.h5')
print('模型已保存: model.h5')

# 生成提交文件
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)

submission = pd.DataFrame({
    'ImageId': range(1, len(predicted_labels) + 1),
    'Label': predicted_labels
})
submission.to_csv('keras_cnn_submission.csv', index=False)
print('提交文件已生成: keras_cnn_submission.csv')