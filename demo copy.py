import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import swanlab
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 初始化SwanLab实验
swanlab.init(
    experiment_name="MNIST_CNN_Fixed",
    description="Fixed CNN on MNIST with proper learning rate",
    config={
        "batch_size": 64,
        "epochs": 20,  # 增加轮次
        "learning_rate": 0.001,  # 调低学习率
        "model": "CNN"
    }
)
# 1. 数据准备
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=swanlab.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=swanlab.config.learning_rate)

# 可视化训练数据中的样本图像
def visualize_samples():
    # 获取一个批次的训练数据
    dataiter = iter(train_loader)
    images, labels = next(dataiter)
    
    # 创建一个图像网格
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('MNIST数据集样本图像', fontsize=16)
    
    # 显示10个样本图像和对应的标签
    for i, ax in enumerate(axes.flat):
        # 将图像从张量转换为numpy数组并调整形状
        img = images[i].numpy().squeeze()  # 去除单维度
        ax.imshow(img, cmap='gray')
        ax.set_title(f'标签: {labels[i].item()}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# 可视化模型预测结果
def visualize_predictions():
    # 设置模型为评估模式
    model.eval()
    
    # 获取一个批次的测试数据
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        # 获取预测的类别
        _, predicted = torch.max(outputs, 1)
    
    # 将数据移回CPU
    images = images.cpu().numpy()
    predicted = predicted.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # 创建一个图像网格
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    fig.suptitle('模型预测结果可视化', fontsize=16)
    
    # 显示10个样本的预测结果
    for i, ax in enumerate(axes.flat):
        img = images[i].squeeze()  # 去除单维度
        # 判断预测是否正确
        is_correct = predicted[i] == labels[i]
        color = 'green' if is_correct else 'red'
        
        ax.imshow(img, cmap='gray')
        ax.set_title(f'预测: {predicted[i]}\n实际: {labels[i]}', 
                    color=color, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

# 3. 训练循环
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            swanlab.log({"train_loss": loss.item()}, step=epoch * len(train_loader) + batch_idx)
            print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

# 4. 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    swanlab.log({
        "test_loss": test_loss,
        "accuracy": accuracy,
        "epoch": epoch
    })
    
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

# 可视化样本图像
visualize_samples()

# 执行训练
for epoch in range(1, swanlab.config.epochs + 1):
    train(epoch)
    test(epoch)
    
    # 每5个epoch可视化一次预测结果
    if epoch % 5 == 0:
        visualize_predictions()

print("训练完成！请在 https://swanlab.cn 查看可视化结果")