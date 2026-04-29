import torch

print('Testing PyTorch...')
try:
    # 测试基本的PyTorch功能
    print(f'PyTorch version: {torch.__version__}')
    
    # 测试GPU是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 测试张量创建和操作
    x = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([4, 5, 6]).to(device)
    z = x + y
    print(f'Tensor addition: {z}')
    
    # 测试简单的模型
    model = torch.nn.Linear(3, 1).to(device)
    input = torch.randn(5, 3).to(device)
    output = model(input)
    print(f'Model output shape: {output.shape}')
    
    print('PyTorch test successful!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()