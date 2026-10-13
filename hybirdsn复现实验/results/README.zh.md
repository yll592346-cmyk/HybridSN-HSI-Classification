# 在 Pavia University 上的 HybridSN

一个面向教学、轻量级的 PyTorch 版 HybridSN 工作流复现，用于高光谱图像分类。包含预处理、训练、全场景预测、OA/AA/Kappa 与混淆矩阵评估、三组受控参数对比、图表，以及一份中文实验报告。

## 出处与致谢

本仓库实现了原始 HybridSN 论文中描述的“先 3D 后 2D”卷积架构。此处的实现与精简训练设置用于课程工作流复现；它们并非原作者代码，也不声称复现了论文的完整实验方案或主要结果。

- Roy, S. K. 等，“HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy for Hyperspectral Image Classification,” *IEEE Geoscience and Remote Sensing Letters*，DOI: [10.1109/LGRS.2019.2918719](https://doi.org/10.1109/LGRS.2019.2918719)。
- 原作者实现：[gokriznastic/HybridSN](https://github.com/gokriznastic/HybridSN)。
- Pavia University 数据来源：[Hyperspectral Remote Sensing Scenes, University of the Basque Country](https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes)。

复用该方法时，请引用原论文与原始实现。本仓库中的脚本是面向本次课程实验的独立、精简 PyTorch 实现。源数据集未包含在内；请从上述提供方获取并遵守其条款。

## 数据集

Pavia University 包含一幅 610 × 340 的图像（103 个光谱波段）以及一幅 610 × 340 的地面真值标签图。该场景在九个类别中共有 42,776 个有标注像素。预期的 MATLAB 文件与键名如下：

| 文件              | MATLAB 键名   |
| --------------- | ----------- |
| `PaviaU.mat`    | `paviaU`    |
| `PaviaU_gt.mat` | `paviaU_gt` |

当这两个文件缺失时，脚本会尝试将它们下载到数据目录。如果提供方无法访问，请从官方页面手动下载并放入 `data/`。

类别编号遵循数据集惯例：1 沥青（Asphalt）、2 草地（Meadows）、3 砂砾（Gravel）、4 树木（Trees）、5 涂漆金属板（Painted metal sheets）、6 裸土（Bare Soil）、7 柏油（Bitumen）、8 自锁砖（Self-Blocking Bricks）、9 阴影（Shadows）。编号 0 表示未标注。

## 环境与快速开始

已在 Windows 上使用 Python 3.8、PyTorch 1.10.1+cu102、NumPy 1.24.4、SciPy 1.10.1、scikit-learn 1.3.2 和 Matplotlib 3.7.5 测试通过；GPU 为 4 GB 显存的 GeForce GTX 1650。当 CUDA 不可用时，该实现也会自动选择 CPU。其他版本可能可用，但未用于所报告的运行结果。

请先安装与你的 CUDA 驱动兼容的 PyTorch 版本（或 CPU 版本），然后安装 `requirements.txt` 中列出的科学计算 Python 依赖。在仓库根目录下运行：

```powershell
python run_hybridsn.py --data-dir data --output-dir results --comparisons
```

快速运行的默认设置：PCA 降至 30 个成分、11 × 11 图像块、分层抽取 10% 训练像素、8 个 epoch、批大小 16、随机种子 42。`--comparisons` 会额外增加一个四 epoch 的对照实验，以及三组分别改变训练比例、PCA 维度或图像块大小的实验。缩减的 epoch 数与样本比例是为了能在普通 GPU 上快速完成。

## 已报告的运行结果

签入的结果是在上述环境下使用本地 Pavia University 数据生成的。具体指标与对比表见 `results/experiment_log.csv`；`results/report_zh.md` 包含完整的中文分析。分数来自一次随机分层像素划分。相邻的训练/测试像素可能存在重叠的图像块，因此这些分数可能偏乐观，应视为工作流演示，而非空间独立的泛化估计，也不构成对论文结果的复现声明。

## 仓库内容

- `run_hybridsn.py` —— 预处理、HybridSN、训练、评估、全场景推理、对比运行，以及报告/图表生成。
- `requirements.txt` —— 本次运行使用的科学计算 Python 依赖版本；PyTorch 请另行安装合适版本。
- `REFLECTION.md` —— 实验心得与局限性。
- `results/class_counts.csv` —— 各类别的有标注像素数量。
- `results/training_log.csv`、`results/training_loss.png` —— 每个 epoch 的损失与训练曲线。
- `results/experiment_log.csv`、`results/comparison_metrics.png` —— 精度指标与受控对比。
- `results/metrics.json`、`results/confusion_matrix.png` —— 主运行指标与混淆矩阵。
- `results/classification_map.png` —— 整幅场景的假彩色图像、地面真值与预测结果。未标注的地面真值像素仅在真值面板中显示为黑色；预测结果覆盖整幅图像。
- `results/prediction_map.npy` —— 610 × 340 的预测数组，包含全场景像素的编号 1–9。
- `results/hybridsn_paviau.pt` —— 主运行的模型状态字典。
- `results/report_zh.md` —— 生成的报告，其中包含 300 字以上的空间分析。

##
