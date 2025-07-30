import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import swanlab

# 初始化SwanLab实验，用于自动生成训练过程的可视化仪表盘
# experiment_name: 实验名称，用于标识当前训练任务
# description: 实验描述，简要说明实验内容
# config: 实验配置，记录超参数等信息，方便后续对比不同实验
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
# 定义数据预处理步骤：将图像转换为张量并进行归一化
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为PyTorch张量
    # 使用MNIST数据集的全局均值和标准差进行归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载训练集和测试集
# train=True表示加载训练集，download=True表示如果数据不存在则下载
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# 创建数据加载器，用于批量处理数据
# shuffle=True表示在每个epoch开始时打乱数据顺序
train_loader = DataLoader(train_dataset, batch_size=swanlab.config.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 第一层卷积：输入通道1(灰度图像)，输出通道32，卷积核大小3x3
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # 第二层卷积：输入通道32，输出通道64，卷积核大小3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # Dropout层，用于防止过拟合，随机丢弃25%的神经元
        self.dropout = nn.Dropout(0.25)
        # 第一个全连接层：输入特征9216，输出特征128
        self.fc1 = nn.Linear(9216, 128)
        # 第二个全连接层：输入特征128，输出特征10(对应10个数字类别)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 第一层卷积 + ReLU激活函数
        x = self.conv1(x)
        x = nn.functional.relu(x)
        # 第二层卷积 + ReLU激活函数
        x = self.conv2(x)
        x = nn.functional.relu(x)
        # 最大池化操作，使用2x2的窗口，减少特征图尺寸
        x = nn.functional.max_pool2d(x, 2)
        # 应用Dropout防止过拟合
        x = self.dropout(x)
        # 将多维特征展平为一维向量
        x = torch.flatten(x, 1)
        # 第一个全连接层 + ReLU激活
        x = self.fc1(x)
        x = nn.functional.relu(x)
        # 再次应用Dropout
        x = self.dropout(x)
        # 第二个全连接层，输出原始分数
        x = self.fc2(x)
        # 应用log_softmax将分数转换为对数概率分布
        return nn.functional.log_softmax(x, dim=1)

# 设置计算设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 实例化模型并将其移动到指定设备
model = CNN().to(device)
# 使用Adam优化器，传入模型参数和学习率
optimizer = optim.Adam(model.parameters(), lr=swanlab.config.learning_rate)

# 3. 训练循环
def train(epoch):
    # 设置模型为训练模式，启用Dropout等训练专用机制
    model.train()
    # 遍历训练数据加载器中的每个批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据和标签移动到指定设备
        data, target = data.to(device), target.to(device)
        # 清除之前的梯度计算结果
        optimizer.zero_grad()
        # 前向传播：计算模型对当前批次的预测
        output = model(data)
        # 计算负对数似然损失，适用于多分类问题
        loss = nn.functional.nll_loss(output, target)
        # 反向传播：计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        
        # 每100个批次记录一次训练损失并打印进度
        if batch_idx % 100 == 0:
            # 使用SwanLab记录训练损失，step表示全局步数
            swanlab.log({"train_loss": loss.item()}, step=epoch * len(train_loader) + batch_idx)
            
            # 打印当前训练进度和损失值
            print(f"Epoch: {epoch} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

# 4. 测试函数
def test(epoch):
    # 设置模型为评估模式，禁用Dropout等训练专用机制
    model.eval()
    test_loss = 0  # 累积测试损失
    correct = 0    # 正确预测的样本数
    
    # 不计算梯度，节省内存并加速计算
    with torch.no_grad():
        # 遍历测试数据加载器中的每个批次
        for data, target in test_loader:
            # 将数据和标签移动到指定设备
            data, target = data.to(device), target.to(device)
            # 前向传播：计算模型预测
            output = model(data)
            # 累积测试损失，reduction='sum'表示对每个批次的损失求和
            test_loss += nn.functional.nll_loss(output, target, reduction='sum').item()
            # 获取预测结果，argmax返回最大值的索引
            pred = output.argmax(dim=1, keepdim=True)
            # 统计正确预测的样本数
            correct += pred.eq(target.view_as(pred)).sum().item()

    # 计算平均测试损失
    test_loss /= len(test_loader.dataset)
    # 计算准确率
    accuracy = 100. * correct / len(test_loader.dataset)
    
    # 使用SwanLab记录测试指标和当前轮次
    swanlab.log({
        "test_loss": test_loss,
        "accuracy": accuracy,
        "epoch": epoch
    })
    
    # 打印测试结果
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n")

# 5. 执行训练
# 遍历指定的训练轮数
for epoch in range(1, swanlab.config.epochs + 1):
    train(epoch)  # 执行一轮训练
    test(epoch)   # 执行一轮测试

print("训练完成！请在 https://swanlab.cn 查看可视化结果")


