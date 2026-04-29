import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

print('Starting CNN training...')

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 加载数据
print('Loading data...')
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

print(f'Train data shape: {train_df.shape}')
print(f'Test data shape: {test_df.shape}')

# 准备数据
X_train = train_df.drop('label', axis=1).values.reshape(-1, 1, 28, 28).astype('float32') / 255.0
y_train = train_df['label'].values.astype('int64')
X_test = test_df.values.reshape(-1, 1, 28, 28).astype('float32') / 255.0

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')

# 转换为PyTorch张量
X_train_tensor = torch.tensor(X_train).to(device)
y_train_tensor = torch.tensor(y_train).to(device)
X_test_tensor = torch.tensor(X_test).to(device)

# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print('Model initialized:', model)

# 训练模型
epochs = 3
batch_size = 64

print('Starting training...')
for epoch in range(epochs):
    print(f'Epoch {epoch+1}/{epochs}')
    
    # 分批训练
    for i in range(0, len(X_train_tensor), batch_size):
        batch_X = X_train_tensor[i:i+batch_size]
        batch_y = y_train_tensor[i:i+batch_size]
        
        # 前向传播
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 每100批次打印一次
        if (i // batch_size) % 100 == 0:
            print(f'  Batch {i//batch_size}, Loss: {loss.item():.4f}')

# 测试并生成预测
print('Generating predictions...')
predictions = []
with torch.no_grad():
    for i in range(0, len(X_test_tensor), batch_size):
        batch_X = X_test_tensor[i:i+batch_size]
        outputs = model(batch_X)
        _, predicted = outputs.max(1)
        predictions.extend(predicted.cpu().numpy())

# 创建提交文件
print('Creating submission file...')
submission = pd.DataFrame({
    'ImageId': range(1, len(predictions) + 1),
    'Label': predictions
})
submission.to_csv('cnn_submission.csv', index=False)

print('Submission file generated: cnn_submission.csv')
print('Done!')