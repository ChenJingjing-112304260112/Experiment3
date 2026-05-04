import sys
import traceback

try:
    print('=== 开始训练 ===', flush=True)
    
    # 加载数据
    print('步骤1: 加载数据...', flush=True)
    import pandas as pd
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f'训练数据: {train_df.shape}', flush=True)
    print(f'测试数据: {test_df.shape}', flush=True)
    
    # 准备数据
    import numpy as np
    X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = train_df['label'].values
    X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    print(f'X_train: {X_train.shape}', flush=True)
    print(f'y_train: {y_train.shape}', flush=True)
    print(f'X_test: {X_test.shape}', flush=True)
    
    # 创建模型
    print('步骤2: 创建模型...', flush=True)
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
    
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
    print('模型创建完成', flush=True)
    
    # 训练
    print('步骤3: 训练模型...', flush=True)
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_split=0.2, verbose=1)
    
    # 预测
    print('步骤4: 预测...', flush=True)
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # 保存
    print('步骤5: 保存提交文件...', flush=True)
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })
    submission.to_csv('keras_cnn_submission.csv', index=False)
    
    print('完成！', flush=True)
    
except Exception as e:
    print(f'错误: {e}', file=sys.stderr)
    traceback.print_exc(file=sys.stderr)