import numpy as np
import pandas as pd
import sys

try:
    print('=== 调试训练脚本 ===')
    
    # 加载数据
    print('步骤1: 加载数据...')
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    
    print(f'训练数据形状: {train_df.shape}')
    print(f'测试数据形状: {test_df.shape}')
    
    # 准备数据
    X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    y_train = train_df['label'].values
    X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    print(f'X_train形状: {X_train.shape}')
    print(f'y_train形状: {y_train.shape}')
    print(f'X_test形状: {X_test.shape}')
    
    # 创建最简单的模型
    print('步骤2: 创建模型...')
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
    
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # 训练
    print('步骤3: 训练模型...')
    model.fit(X_train, y_train, batch_size=128, epochs=5, validation_split=0.1, verbose=1)
    
    # 预测
    print('步骤4: 预测...')
    predictions = model.predict(X_test)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print(f'预测结果形状: {predicted_labels.shape}')
    print(f'前10个预测: {predicted_labels[:10]}')
    
    # 创建提交文件
    print('步骤5: 创建提交文件...')
    submission = pd.DataFrame({
        'ImageId': range(1, len(predicted_labels) + 1),
        'Label': predicted_labels
    })
    
    print(f'提交文件形状: {submission.shape}')
    print('前5行数据:')
    print(submission.head())
    
    submission.to_csv('debug_submission.csv', index=False)
    print('提交文件已保存: debug_submission.csv')
    
    print('=== 完成 ===')
    
except Exception as e:
    print(f'错误: {e}', file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)