# -*- coding: utf-8 -*-
"""
  用多层感知机 MLP 从车削/铣削的三项工艺参数
  [切削速度 v, 进给量 f, 切削深度 ap]
  回归预测主切削力 Fc。

本文件演示：
  1. 不同激活函数的神经元：Sigmoid / ReLU / Swish / GELU / Maxout
  2. 在同一机械任务上切换不同激活函数进行对比
  3. 使用 matplotlib 画出 “Fc 真值 vs 预测” 的对比图

依赖：
  - torch
  - matplotlib
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ===================== 1. 构造一个“机械加工”风格的模拟数据集 =====================

def generate_synthetic_machining_data(n_samples=2000, seed=42):
    """
    生成模拟的切削实验数据：
      输入 x = [v, f, ap]
      输出 y = Fc (主切削力)

    这里用一个“物理 + 非线性”的人造公式：
      Fc = 100 + 0.8 * v + 150 * f + 120 * ap
           + 50 * sin(v / 100)
           + 噪声
    """
    torch.manual_seed(seed)

    # 切削速度 v
    v = torch.empty(n_samples, 1).uniform_(50.0, 300.0)
    # 进给量 f: 0.05 ~ 0.3 mm/rev
    f = torch.empty(n_samples, 1).uniform_(0.05, 0.30)
    # 切削深度 ap: 0.2 ~ 2.0 mm
    ap = torch.empty(n_samples, 1).uniform_(0.2, 2.0)

    # 拼成特征矩阵 X
    X = torch.cat([v, f, ap], dim=1)

    # 力模型（含非线性 + 噪声）
    Fc_clean = (100.0
                + 0.8 * v
                + 150.0 * f
                + 120.0 * ap
                + 50.0 * torch.sin(v / 100.0))

    noise = torch.randn_like(Fc_clean) * 10.0  # 测量噪声
    y = Fc_clean + noise

    X_min = X.min(dim=0, keepdim=True).values
    X_max = X.max(dim=0, keepdim=True).values
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    return X_norm, y, X_min, X_max


# ===================== 2. 激活函数 & 神经元定义 =====================

class Swish(nn.Module):
    """Swish: y = x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class Maxout(nn.Module):
    """
    Maxout 神经元层：
      y = max(W1 x + b1, W2 x + b2, ..., Wk x + bk)

    - in_features  : 输入维度
    - out_features : 输出维度（神经元个数）
    - num_pieces   : 每个神经元的线性分支数量 K
    """
    def __init__(self, in_features, out_features, num_pieces=2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_pieces = num_pieces
        # 线性层输出维度 = out_features * num_pieces
        self.lin = nn.Linear(in_features, out_features * num_pieces)

    def forward(self, x):
        # [N, out_features * num_pieces]
        out = self.lin(x)
        # reshape 成 [N, out_features, num_pieces]
        out = out.view(-1, self.out_features, self.num_pieces)
        # 在 num_pieces 维度上取最大值 => [N, out_features]
        out, _ = out.max(dim=2)
        return out


def get_activation(act_name):
    """
    根据名字返回对应的激活层实例：
      - sigmoid
      - relu
      - swish
      - gelu
    Maxout 不在此处处理，而是用 Maxout 层替代 “线性 + 激活”。
    """
    act_name = act_name.lower()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "swish":
        return Swish()
    elif act_name == "gelu":
        return nn.GELU()
    else:
        raise ValueError("Unsupported activation: {}".format(act_name))


# ===================== 3. 网络结构：普通激活 & Maxout 版本 =====================

class MLPWithActivation(nn.Module):
    """
    适用于 Sigmoid / ReLU / Swish / GELU 的 MLP：
      输入维度: 3 (v, f, ap)
      隐藏层: 64 -> 64
      输出维度: 1 (Fc)

    结构：
      x -> Linear(3,64) -> act -> Linear(64,64) -> act -> Linear(64,1)
    """
    def __init__(self, act_name="relu"):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.act1 = get_activation(act_name)
        self.fc2 = nn.Linear(64, 64)
        self.act2 = get_activation(act_name)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        x = self.fc_out(x)
        return x


class MaxoutMLP(nn.Module):
    """
    使用 Maxout 神经元的 MLP：
      - 用 Maxout 层替代 “Linear + 激活”
      - 每个 Maxout 单元由 num_pieces 个线性片段组成（缺省 2）
    结构：
      x -> Maxout(3,64) -> Maxout(64,64) -> Linear(64,1)
    """
    def __init__(self, num_pieces=2):
        super().__init__()
        self.max1 = Maxout(3, 64, num_pieces=num_pieces)
        self.max2 = Maxout(64, 64, num_pieces=num_pieces)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x):
        x = self.max1(x)
        x = self.max2(x)
        x = self.fc_out(x)
        return x


# ===================== 4. 训练与评估：返回预测结果用于画图 =====================

def train_one_model(model, X_train, y_train, n_epochs=300, lr=1e-3, verbose=True):
    """
    简单的全批量训练：
      - 损失：MSELoss（回归问题）
      - 优化器：Adam
    返回训练好的模型
    """
    model = model.to(torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(n_epochs):
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0):
            print("  Epoch {:3d} | Loss = {:.4f}".format(epoch + 1, loss.item()))

    return model


def evaluate_model(model, X_test, y_test):
    """
    在测试集上评估模型，计算 MSE 并返回预测值（用于后续画图）
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)

    loss_fn = nn.MSELoss()
    mse = loss_fn(y_pred, y_test).item()
    return y_pred, mse


# ===================== 5. 主流程：训练 + 可视化对比图 =====================

def main():
    # 1) 生成模拟数据（机械行业背景：切削参数 -> 主切削力）
    X, y, X_min, X_max = generate_synthetic_machining_data(n_samples=2000, seed=42)

    # 转为 float32，方便 PyTorch 使用
    X = X.to(torch.float32)
    y = y.to(torch.float32)

    # 2) 划分训练集 / 测试集（这里简单按前 80% 训练，后 20% 测试）
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)

    X_train = X[:n_train]
    y_train = y[:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    print("总样本数: {}, 训练集: {}, 测试集: {}".format(n_samples, n_train, n_samples - n_train))

    # 3) 定义要对比的激活函数 / 神经元
    #    四种标量激活 + Maxout
    act_names = ["sigmoid", "relu", "swish", "gelu"]
    all_model_names = []   # 用于画图时的标签顺序
    all_preds = {}         # 保存每种激活在测试集上的预测
    all_mse = {}           # 保存每种激活在测试集上的 MSE

    # 3.1) Sigmoid / ReLU / Swish / GELU
    for act in act_names:
        print("\n==============================")
        print("训练激活函数为 {:>7s} 的 MLP 模型".format(act))
        print("==============================")
        model = MLPWithActivation(act_name=act)
        model = train_one_model(model, X_train, y_train,
                                n_epochs=300, lr=1e-3, verbose=True)
        y_pred, mse = evaluate_model(model, X_test, y_test)
        print(">> 测试集 MSE ({:>7s}) = {:.4f}".format(act, mse))

        all_model_names.append(act.upper())
        all_preds[act.upper()] = y_pred.detach().cpu().numpy().reshape(-1)
        all_mse[act.upper()] = mse

    # 3.2) Maxout 版本
    print("\n==============================")
    print("训练使用 Maxout 神经元的 MLP 模型")
    print("==============================")
    maxout_model = MaxoutMLP(num_pieces=2)
    maxout_model = train_one_model(maxout_model, X_train, y_train,
                                   n_epochs=300, lr=1e-3, verbose=True)
    y_pred_maxout, mse_maxout = evaluate_model(maxout_model, X_test, y_test)
    print(">> 测试集 MSE (MAXOUT) = {:.4f}".format(mse_maxout))

    all_model_names.append("MAXOUT")
    all_preds["MAXOUT"] = y_pred_maxout.detach().cpu().numpy().reshape(-1)
    all_mse["MAXOUT"] = mse_maxout

    # 4) 将测试集真值转为 numpy，备用
    y_test_np = y_test.detach().cpu().numpy().reshape(-1)

    # 为了图像更清晰，只画前 n_show 个测试样本
    n_show = min(200, len(y_test_np))
    x_index = np.arange(n_show)  # x 轴：样本索引

    # ===================== 6. 可视化 Fc 真值 vs 预测 =====================

    # 6.1) 按样本索引画 “真值 vs 预测” 曲线（每种激活一个子图）
    num_models = len(all_model_names)
    fig, axes = plt.subplots(1, num_models,
                             figsize=(4 * num_models, 4),
                             sharey=True)

    if num_models == 1:
        # 兼容只有一个子图时 axes 不是数组的情况
        axes = [axes]

    for i, name in enumerate(all_model_names):
        ax = axes[i]
        y_pred_np = all_preds[name]

        # 画真值曲线
        ax.plot(x_index,
                y_test_np[:n_show],
                marker='o',
                linestyle='-',
                label='Fc True')

        # 画预测曲线
        ax.plot(x_index,
                y_pred_np[:n_show],
                marker='x',
                linestyle='--',
                label='Fc Pred')

        ax.set_title("{}\nMSE={:.2f}".format(name, all_mse[name]))
        ax.set_xlabel("Test sample index")
        if i == 0:
            ax.set_ylabel("Cutting force Fc")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.suptitle("Comparison of Fc: Ground Truth vs Prediction\n(不同激活函数/神经元)", y=1.05)
    plt.tight_layout()
    plt.show()

    # 6.2) 可选：真值 vs 预测的散点图（所有样本合在一起对比）
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for name in all_model_names:
        y_pred_np = all_preds[name]
        ax2.scatter(y_test_np,
                    y_pred_np,
                    s=10,
                    alpha=0.6,
                    label=name + " (MSE={:.2f})".format(all_mse[name]))
    # 画理想情况 y = x 参考线
    min_val = min(y_test_np.min(), min(v.min() for v in all_preds.values()))
    max_val = max(y_test_np.max(), max(v.max() for v in all_preds.values()))
    ax2.plot([min_val, max_val],
             [min_val, max_val],
             linestyle='--',
             linewidth=1.0)

    ax2.set_xlabel("Fc True")
    ax2.set_ylabel("Fc Predicted")
    ax2.set_title("Fc 真值 vs 预测散点图（所有激活函数）")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
