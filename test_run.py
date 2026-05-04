import numpy as np
import pandas as pd

print('测试数据加载...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f'训练数据: {train_df.shape}')
print(f'测试数据: {test_df.shape}')

X_train = train_df.drop('label', axis=1).values.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = train_df['label'].values
X_test = test_df.values.reshape(-1, 28, 28, 1).astype('float32') / 255.0

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')

print('测试完成！')