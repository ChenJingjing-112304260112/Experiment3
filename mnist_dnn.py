import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print('Using CNN model for training')

# 自定义数据集类
class MNISTDataset(Dataset):
    def __init__(self, csv_file, train=True):
        self.data = pd.read_csv(csv_file)
        self.train = train
        
        if self.train:
            self.labels = self.data['label'].values
            self.images = self.data.drop('label', axis=1).values
        else:
            self.images = self.data.values
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(1, 28, 28)  # 1通道，28x28
        image = torch.tensor(image, dtype=torch.float32) / 255.0  # 归一化
        
        if self.train:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        else:
            return image

# 简单的CNN模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 卷积层1：输入1通道，输出16通道， kernel size 3x3
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        # 池化层1：2x2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 卷积层2：输入16通道，输出32通道， kernel size 3x3
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 池化层2：2x2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层1：输入32*7*7=1568，输出128
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        # 全连接层2：输入128，输出10（10个类别）
        self.fc2 = nn.Linear(128, 10)
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 卷积层1 -> 激活 -> 池化
        x = self.pool1(self.relu(self.conv1(x)))
        # 卷积层2 -> 激活 -> 池化
        x = self.pool2(self.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 32 * 7 * 7)
        # 全连接层1 -> 激活
        x = self.relu(self.fc1(x))
        # 全连接层2
        x = self.fc2(x)
        return x

# 超参数
batch_size = 32  # 减少batch size
learning_rate = 0.001
epochs = 5  # 减少训练轮数

# 加载数据
train_dataset = MNISTDataset('train.csv', train=True)
test_dataset = MNISTDataset('test.csv', train=False)

# 分割训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
def train(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 每100批次打印一次
            if (i + 1) % 100 == 0:
                print(f'Batch {i+1}/{len(train_loader)}, Loss: {running_loss/(i+1):.4f}, Acc: {100.*correct/total:.2f}%')
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print('-' * 50)

# 训练模型
train(model, train_loader, val_loader, criterion, optimizer, epochs)

# 测试并生成提交文件
def generate_submission(model, test_loader):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for i, images in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
            
            # 每100批次打印一次
            if (i + 1) % 100 == 0:
                print(f'Processing batch {i+1}/{len(test_loader)}')
    
    # 创建提交文件
    submission = pd.DataFrame({
        'ImageId': range(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print('Submission file generated: submission.csv')

# 生成提交文件
generate_submission(model, test_loader)