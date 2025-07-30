import time
import os
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import wandb
run = wandb.init(project="my_first_project", mode="online")


# 保存超参数配置
config = run.config
config.learning_rate = 0.01

# 记录程序开始时间
start_time = time.time()

# 打印运行模式
print(f"🚀 WandB Run Mode: {run.mode}")
if run.mode != "online":
    print("⚠️ 当前为离线模式，数据将缓存本地后续上传")

# 模拟训练循环
for i in range(10):
    # 模拟损失值（随步数减小）
    loss = 1.0 / (i + 1)

    # 当前运行时间（秒）
    elapsed_time = time.time() - start_time

    # 打印当前轮数与运行时间
    print(f"[第 {i+1} 次测试] Loss: {loss:.4f} | 累计耗时: {elapsed_time:.2f} 秒")

    # 尝试记录到wandb（离线模式也会保存到本地）
    run.log({
        "loss": loss,
        "elapsed_time_sec": elapsed_time
    })

    # 模拟训练时间
    time.sleep(1.5)

# 标记结束
run.finish()
