import torch
import pandas as pd
import numpy as np

# 检查基本库是否正常
try:
    print('Testing Python environment...')
    print(f'PyTorch version: {torch.__version__}')
    print(f'Pandas version: {pd.__version__}')
    print(f'NumPy version: {np.__version__}')
    
    # 检查GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 测试基本的张量操作
    x = torch.tensor([1, 2, 3])
    print(f'Tensor test: {x}')
    
    # 测试数据加载
    print('Testing data loading...')
    train_df = pd.read_csv('train.csv')
    print(f'Train data shape: {train_df.shape}')
    test_df = pd.read_csv('test.csv')
    print(f'Test data shape: {test_df.shape}')
    
    print('All tests passed!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()