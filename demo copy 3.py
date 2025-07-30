import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
import socket
import requests
import time
import matplotlib.pyplot as plt
import numpy as np

# === SwanLab相关 ===
try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False
    print("⚠️ 未安装 SwanLab，日志将以本地方式保存（无可视化）。\n可通过 pip install swanlab 安装。")

# 1. 诊断网络连接问题
def check_connection():
    print("\n=== 网络连接诊断 ===")

    # 测试基本互联网连接
    try:
        socket.create_connection(("www.bing.com", 80), timeout=5)
        print("✅ 基本互联网连接正常")
    except OSError:
        print("❌ 无法连接到互联网 - 请检查您的网络连接")
        return False

    # 测试 SwanLab API 连接
    try:
        response = requests.get("https://www.swanlab.ai/", timeout=10)
        if response.status_code == 200:
            print("✅ 可以访问 SwanLab API 服务器")
            return True
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到 SwanLab API 服务器: {str(e)}")

    print("\n⚠️ 网络问题解决方案:")
    print("1. 检查代理设置: 确保没有使用VPN或代理")
    print("2. 尝试禁用防火墙/杀毒软件")
    print("3. 使用手机热点测试是否是网络问题")
    print("4. 使用本地模式: swanlab.init(mode='offline')")
    return False

# 2. 配置 SwanLab（带诊断）
def init_swanlab():
    if not SWANLAB_AVAILABLE:
        return None  # SwanLab不可用

    results_dir = "swanlab_results"
    os.makedirs(results_dir, exist_ok=True)

    # 检查网络连接
    online = check_connection()

    # 初始化 SwanLab
    try:
        run = swanlab.init(
            project="pytorch-mnist-demo",
            config={
                "learning_rate": 0.001,
                "batch_size": 64,
                "epochs": 5,
                "architecture": "CNN"
            },
            dir=results_dir,
            mode="online" if online else "offline"
        )
        print(f"\nSwanLab 初始化 {'成功 (在线模式)' if online else '成功 (离线模式)'}")
        return run
    except Exception as e:
        print(f"❌ SwanLab 初始化失败: {str(e)}")
        print("将在本地继续运行训练...")
        return None

# 3. 主程序
def main():
    # 初始化 SwanLab
    swanlab_run = init_swanlab() if SWANLAB_AVAILABLE else None

    # 获取配置参数
    config = {
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 5,
        "architecture": "CNN"
    }
    if swanlab_run is not None:
        config = swanlab.config

    print("\n=== 配置参数 ===")
    print(f"学习率: {config['learning_rate']}")
    print(f"批次大小: {config['batch_size']}")
    print(f"训练周期: {config['epochs']}")
    print(f"模型架构: {config['architecture']}")

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    # 模型定义
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = torch.flatten(x, 1)
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN().to(device)

    # 准备数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_dir = './data'
    os.makedirs(data_dir, exist_ok=True)

    print("\n下载MNIST数据集...")
    train_set = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    test_set = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False
    )

    # 设置优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # 训练循环
    print("\n开始训练...")
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计指标
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 每100批次记录一次
            if (i+1) % 100 == 0:
                avg_loss = running_loss / 100
                accuracy = 100 * correct / total

                train_losses.append(avg_loss)
                train_accuracies.append(accuracy)

                # 日志到 SwanLab
                if swanlab_run is not None:
                    try:
                        swanlab.log({
                            "training_loss": avg_loss,
                            "training_accuracy": accuracy,
                            "step": epoch * len(train_loader) + i
                        })
                    except Exception as e:
                        print(f"⚠️ 无法记录到 SwanLab: {e}")

                print(f"Epoch [{epoch+1}/{config['epochs']}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%")

                running_loss = 0.0
                correct = 0
                total = 0

        # 测试集评估
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)

        test_losses.append(avg_test_loss)
        test_accuracies.append(test_accuracy)

        # 记录测试结果
        if swanlab_run is not None:
            try:
                swanlab.log({
                    "test_loss": avg_test_loss,
                    "test_accuracy": test_accuracy,
                    "epoch": epoch + 1
                })
            except Exception as e:
                print(f"⚠️ 无法记录到 SwanLab: {e}")

        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch+1}/{config['epochs']}] 完成, "
              f"时间: {epoch_time:.1f}秒, "
              f"测试损失: {avg_test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%")

    # 保存模型
    model_path = "mnist_cnn.pth"
    torch.save(model.state_dict(), model_path)

    # SwanLab保存模型
    if swanlab_run is not None:
        try:
            swanlab.save(model_path)
            print("模型已保存到 SwanLab")
        except Exception as e:
            print(f"⚠️ 无法上传模型到 SwanLab: {e}")

        try:
            swanlab.finish()
            print("SwanLab 运行完成")
        except Exception as e:
            print(f"⚠️ 无法正确结束 SwanLab 运行: {e}")

    # 本地保存结果图表
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses)
    plt.title("训练损失")
    plt.xlabel("批次")
    plt.ylabel("损失")

    plt.subplot(2, 2, 2)
    plt.plot(train_accuracies)
    plt.title("训练准确率")
    plt.xlabel("批次")
    plt.ylabel("准确率 (%)")

    plt.subplot(2, 2, 3)
    plt.plot(test_losses)
    plt.title("测试损失")
    plt.xlabel("周期")
    plt.ylabel("损失")

    plt.subplot(2, 2, 4)
    plt.plot(test_accuracies)
    plt.title("测试准确率")
    plt.xlabel("周期")
    plt.ylabel("准确率 (%)")

    plt.tight_layout()
    plt.savefig("training_results.png")
    print("\n训练结果已保存到 training_results.png")

    print("\n训练完成!")

if __name__ == "__main__":
    main()