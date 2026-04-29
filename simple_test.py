import pandas as pd
import numpy as np

print('Testing data loading...')
try:
    # 加载训练数据
    train_df = pd.read_csv('train.csv')
    print(f'Train data shape: {train_df.shape}')
    print(f'Train data columns: {train_df.columns[:5]}...')
    
    # 加载测试数据
    test_df = pd.read_csv('test.csv')
    print(f'Test data shape: {test_df.shape}')
    print(f'Test data columns: {test_df.columns[:5]}...')
    
    # 打印一些样本数据
    print('Sample labels:', train_df['label'].head())
    print('Sample image data:', train_df.iloc[0, 1:].head())
    
    print('Data loading successful!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()