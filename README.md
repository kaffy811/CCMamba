# CCMamba: Complementary Masked Siamese Mamba Network

CCMamba 是一个面向多模态语义分割（RGB + X，如热红外/深度）的研究代码仓库，聚焦于在遮挡、噪声、弱光等复杂场景下提升模型鲁棒性。该项目在 Mamba 风格主干基础上，引入互补掩码与自蒸馏训练机制，用于增强跨模态补偿能力。

> 仓库地址：https://github.com/kaffy811/CCMamba

---

## 1. 项目简介

本仓库主要用于毕业论文相关实验，实现了一个多模态语义分割训练/评估框架，支持多个公开数据集（如 MFNet、PST900、NYU、SUNRGBD、FMB）。

从代码实现看，项目核心目标是：

- 通过 **互补随机掩码（CRM）**，降低模型对单一模态的依赖；
- 通过 **教师-学生自蒸馏**，缓解掩码引起的特征分布偏移；
- 在多模态融合阶段提升对低质量特征的抑制能力。

---

## 2. 方法核心思想

### 2.1 互补随机掩码（Complementary Random Masking）

训练时随机生成二值掩码 `M`：

- RGB 分支输入 `RGB * M`
- X 模态分支输入 `X * (1 - M)`（或在部分数据集上采用同位掩码）

这样可强制模型在某一模态局部缺失时，从另一模态进行语义补偿，提升鲁棒性。

### 2.2 教师-学生自蒸馏（Self-Distillation）

- **Teacher 路径**：输入未掩码图像，生成 clean logits（无梯度）；
- **Student 路径**：输入掩码图像，生成 masked logits；
- 使用 `KLDiv + CrossEntropy` 联合优化：

\[
\mathcal{L} = \mathcal{L}_{CE} + \alpha \cdot \mathcal{L}_{KD}
\]

其中蒸馏权重会按训练进度渐进增大，并包含异常值保护机制以防梯度不稳定。

---

## 3. 仓库结构

```text
CCMamba/
├── configs/                # 各数据集配置文件（路径、类别、训练超参等）
├── dataloader/             # 数据集与 DataLoader
├── engine/                 # 训练引擎、日志、评估流程
├── models/                 # 模型主体与融合模块
├── utils/                  # 指标、可视化、工具函数
├── train.py                # 训练入口
├── eval.py                 # 评估入口
├── cal_complexity.py       # 复杂度统计脚本
├── requirements.txt        # 依赖列表
└── README.md
```

---

## 4. 环境配置

> 推荐：Linux + NVIDIA GPU + CUDA（如 RTX 3090）。

### 4.1 创建环境

```bash
conda create -n ccmamba python=3.10 -y
conda activate ccmamba
```

### 4.2 安装 PyTorch

请根据你的 CUDA 版本安装对应的 PyTorch（建议参考 PyTorch 官网命令）。例如 CUDA 11.8：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4.3 安装其余依赖

```bash
pip install -r requirements.txt
```

---

## 5. 数据准备

本仓库支持以下数据集名称（由训练/评估脚本解析）：

- `mfnet`
- `pst`
- `nyu`
- `sun`
- `fmb`

### 5.1 下载数据集

请先从各数据集官方渠道下载原始数据。

### 5.2 配置数据路径

在 `configs/config_*.py` 中配置：

- `rgb_root_folder`
- `x_root_folder`
- `gt_root_folder`
- `train_source`
- `eval_source`

请确保 `train_source` / `eval_source` 列表中的样本索引与目录结构一致。

---

## 6. 训练复现步骤

### 6.1 单卡训练（推荐先跑通）

```bash
python train.py --dataset_name mfnet --devices 0
```

可替换 `dataset_name` 为：`pst / nyu / sun / fmb`。

### 6.2 多卡训练（如需）

若你的 `Engine` 配置支持分布式，可按原项目分布式方式启动；建议先在单卡验证配置正确后再切换多卡。

### 6.3 训练输出

训练过程中会在配置文件指定目录输出：

- checkpoint 权重（按 epoch 保存/筛选）
- 日志文件
- TensorBoard 日志

---

## 7. 测试与评估复现步骤

### 7.1 使用指定权重评估

```bash
python eval.py --dataset_name mfnet --devices 0 --epochs last
```

常用参数：

- `--dataset_name`: `mfnet / pst / nyu / sun / fmb`
- `--devices`: GPU 编号
- `--epochs`: `last` 或具体 epoch（如 `100`）
- `--save_path`: 保存预测 mask 路径（可选）
- `--show_image`: 可视化显示（可选）

### 7.2 结果指标

评估脚本默认输出分割指标（含 mIoU）。

---

## 8. 常见问题（FAQ）

### Q1: 运行时报 CUDA / 算子相关错误？
优先检查：

1. PyTorch 与 CUDA 版本是否匹配；
2. GPU 驱动是否正常；
3. 是否在 CPU 或非 NVIDIA 环境下运行了仅 CUDA 支持的模块。

### Q2: 显存不足（OOM）怎么办？
可按顺序尝试：

- 减小 batch size；
- 减小输入分辨率或裁剪尺寸；
- 关闭部分增强；
- 优先单卡调通后再加大规模。

### Q3: 评估找不到权重文件？
检查配置中的 `checkpoint_dir` 与 `--epochs` 是否一致；确认训练阶段已成功保存权重。

---

## 9. 复现实验建议（论文写作友好）

为了保证结果可复现，建议在实验记录中固定并保存：

- 代码 commit id；
- 配置文件版本；
- 随机种子；
- PyTorch/CUDA/驱动版本；
- 数据划分文件（train/eval source）。

---

## 10. 致谢

本仓库用于毕业论文相关研究与实验验证，欢迎基于规范引用进行学术交流。