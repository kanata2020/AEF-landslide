# Landslide Spatiotemporal Benchmark

这是一个可复现的五地区滑坡研究项目，包含两类实验：

- 15 时相 Sentinel-2 3D-CNN 空间分割模型；
- AEF 空间分割与事件月份联合预测模型。

所有实验使用同一份固定划分：2095 个训练事件、898 个验证事件，覆盖 Chimanimani、Dominica Maria、Hiroshima、Hokkaido 和 Italy。数据不复制到本目录，默认读取父项目中的现有文件。

## 项目结构

```text
landslide_benchmark/
|-- scripts/
|   |-- train_optical.py
|   |-- train_spatiotemporal.py
|   |-- evaluate.py
|   `-- evaluate_spatiotemporal.py
|-- src/landslide_benchmark/
|   |-- data.py
|   |-- models.py
|   |-- training.py
|   |-- spatiotemporal_data.py
|   |-- spatiotemporal_model.py
|   `-- spatiotemporal_training.py
|-- outputs/
|-- pyproject.toml
`-- requirements.txt
```

默认数据路径：

```text
../AEFdata/AEF_Embedding_data/
../Sen12Landslides/Sen12Landslides/s2_data/
../Sen12Landslides-main/outputs/landdetect_3dcnn/splits/dataset_split_resolved.json
```

所有路径都可以通过命令行参数覆盖。

## 安装

在项目根目录执行：

```powershell
cd landslide_benchmark
python -m pip install -e .
```

检查 GPU：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 联合模型设计

联合模型只使用 64 通道 AEF embedding，输出一个像素级滑坡 mask 和 13 类时间概率（1--12 月以及 `no-event`）。

```text
AEF [B,64,128,128]
        |
  rotate/crop/pad + NaN/Inf to zero
        |
  shared full-resolution stem (96 channels)
        |-----------------------------|
  5 residual-SE blocks       gradient scale = 0.1
        |                             |
  segmentation head          temporal downsampling branch
        |                             |
  [B,1,128,128]                    [B,13]
```

关键设计如下：

- AEF 不做标准化或数值裁剪，只执行与 mask 对齐所需的旋转、中心裁剪/补零，并将 NaN/Inf 替换为零。
- 空间路径使用 96 通道、全分辨率、5 个 Residual-SE Block，不下采样细小滑坡。
- 默认从 `../LandDetect/model/aef_s2_3dcnn_split_stable/best.pth` 初始化空间路径；前 5 个 epoch 冻结空间路径，只训练时间头。
- 时间分支只共享浅层 stem，不再读取或依赖 segmentation logits；时间梯度回传共享 stem 时缩小到 0.1，对空间 body 和 head 的梯度为零。
- 空间损失为 `0.7 × BCE + 0.3 × soft Dice`，不使用正类权重，避免大面积假阳性。
- 时间损失使用训练集类别权重。1--12 月采用循环标签平滑，相邻月份各获得 0.05 概率；`no-event` 使用严格 one-hot 标签。
- 总损失默认为 `spatial_loss + 0.25 * temporal_loss`。月份是辅助任务，避免它压过主要分割目标。
- AdamW 为共享/空间路径使用 `3e-5`，时间分支使用 `1e-4`；8 epoch warm-up 后根据验证空间损失自适应降学习率。
- 默认 300 epochs、batch size 16、CUDA 自动混合精度；空间 IoU 连续 60 epoch 不提升时早停。

这些默认值是实验起点，不应把验证集反复用于无约束调参。正式论文应预先固定搜索空间，或从训练集再划出 development set 后只对 898 个事件做一次最终评估。

## 时间标签的重要限制

当前固定划分的日期字段并不完整。按照本项目的标签定义，缺失、空白或无法解析的日期统一归入第 13 类 `no-event`：

| Split | 1--12 月标签 | no-event | 出现月份 |
|---|---:|---:|---|
| train | 1048 | 1047 | 3、5、6、9 月 |
| val | 460 | 438 | 3、5、6、9 月 |

所有 13 类都参与时间损失与时间指标。对当前固定划分的全量检查确认：训练集 1047 个、验证集 438 个 `no-event` 样本均为空 mask，没有发现标签冲突。训练统计和评估报告仍会输出 `no_event_samples_with_positive_mask`；若未来数据中的该值非零，应在解释结果前核查数据语义。

月份还与地区高度混杂：Chimanimani 的有效标签为 3 月、Italy 为 5 月、Hiroshima 为 6 月，Dominica Maria 和 Hokkaido 为 9 月。因此较高的月份准确率可能来自地区识别，而不是从年度 AEF 中恢复了事件发生时间。联合评估会同时给出按地区众数预测的基线；论文中必须披露这一限制，不能把随机事件划分下的月份准确率单独作为时序学习证据。

## 训练联合模型

```powershell
python scripts\train_spatiotemporal.py --device cuda
```

第一次运行会扫描训练集并生成：

```text
outputs/spatiotemporal_v2/training_stats.json
```

输出文件：

```text
outputs/spatiotemporal_v2/best.pth          # 0.7 IoU + 0.3 temporal macro-F1 最优
outputs/spatiotemporal_v2/best_spatial.pth  # 空间 IoU 最优
outputs/spatiotemporal_v2/last.pth          # 最近完成的 epoch，用于续训
outputs/spatiotemporal_v2/history.json
```

按 `Ctrl+C` 可以安全停止。再次执行相同命令会从 `last.pth` 的下一 epoch 继续。中断发生在 epoch 中间时，该未完成 epoch 会重新训练。

常用参数：

```powershell
# 显存不足
python scripts\train_spatiotemporal.py --batch-size 8

# 禁用混合精度
python scripts\train_spatiotemporal.py --no-amp

# 从头训练；建议同时指定新输出目录，以保留旧实验
python scripts\train_spatiotemporal.py --force-train --output-dir outputs\spatiotemporal_v2_run2

# 不加载空间路径预训练权重，从随机初始化开始
python scripts\train_spatiotemporal.py --no-spatial-init

# 实验性开启旋转/翻转增强；默认关闭
python scripts\train_spatiotemporal.py --augment

# 快速检查数据和训练链路，不用于报告结果
python scripts\train_spatiotemporal.py --epochs 1 --limit-train 16 --limit-val 8 --output-dir outputs\smoke_test
```

## 评估联合模型

```powershell
python scripts\evaluate_spatiotemporal.py
```

默认评估 `best.pth` 和固定的 898 个验证事件，报告保存到：

```text
outputs/reports/spatiotemporal_val_metrics.json
```

评估 `best_spatial.pth`：

```powershell
python scripts\evaluate_spatiotemporal.py --checkpoint outputs\spatiotemporal_v2\best_spatial.pth
```

报告包括：

- 滑坡类和非滑坡类 precision、recall、F1、IoU；
- macro average、accuracy、confusion counts；
- 五地区分别的空间和时间指标；
- 13 类 accuracy、balanced accuracy、observed-class macro-F1、Top-3 accuracy；
- 可比较日历月样本上的循环 MAE，以及 global-mode 和 region-mode 基线；
- `no-event` 与正 mask 的标签冲突计数。

## Sentinel-2 空间分割基线

训练：

```powershell
python scripts\train_optical.py --device cuda --amp
```

测试：

```powershell
python scripts\evaluate.py --model optical
```

若 Optical 评估显存不足：

```powershell
python scripts\evaluate.py --model optical --batch-size 2
```

## 跨地区迁移验证（论文四折实验）

入口为 `scripts/cross_region.py`。该实验独立于上面的五地区随机划分实验，按论文的 Cross-regional Transferability Validation 设置四折：Chimanimani、Hiroshima、Hokkaido、Dominica（数据中的名称为 `dominicamaria`）。Italy 不参与本实验。

先合并输入 split JSON 中的 train 和 val 事件作为候选清单，再按地区重新划分。每折将目标地区的全部事件留作测试；其余三个地区分别按约 70%/30% 划为源训练集和源验证集，固定种子默认为 42。AEF 联合模型与光学基线使用完全相同的事件划分。当前清单对应：

| 留出地区 | 源训练事件 | 源验证事件 | 目标测试事件 |
|---|---:|---:|---:|
| Chimanimani | 1258 | 539 | 633 |
| Hiroshima | 1096 | 470 | 864 |
| Hokkaido | 1498 | 642 | 290 |
| Dominica | 1251 | 536 | 643 |

每折遵循以下流程：

1. 仅用源训练事件重新计算 Sentinel-2/DEM 均值、标准差及 AEF 月份类别权重。
2. 光学 3D-UNet 从随机初始化训练。AEF 使用联合网络先执行源地区空间预训练（时间损失权重为零），再将该折最佳空间权重加载到新的联合网络，按原联合训练流程训练。不会加载原来的全地区 AEF 权重。
3. 两种模型均由源验证集空间 IoU 选择 checkpoint 并早停；AEF 测试使用 `aef/best_spatial.pth`，光学使用 `optical/best.pth`。
4. 固定阈值 0.5，在目标地区仅推理。跨地区流程将光学影像和 mask 同步对齐至联合模型使用的 mask 存储方向，并检查两者的参考掩膜一致。目标样本不参与训练、统计拟合、模型选择或微调。
5. 输出各地区滑坡类 Precision、Recall、F1、IoU，以及 AEF 的 13 类月份准确率（包含 `no-event`）。四折全部完成后，分别对每项地区指标取等权算术平均，不按像素数或样本数加权。

在 `landslide_benchmark` 目录下运行。若尚未安装项目，先执行上面的 `python -m pip install -e .`。

只生成并检查四折划分：

```powershell
python scripts\cross_region.py --stage prepare
```

运行完整实验（四折、两种模型、测试及汇总）：

```powershell
python scripts\cross_region.py --device cuda
```

默认每折空间预训练最多 300 epochs、联合训练最多 300 epochs、光学训练最多 80 epochs，验证 IoU 连续 60 epochs 不提升时早停。AEF batch size 为 16；光学训练为 12，测试为 4。按 `Ctrl+C` 停止后，使用完全相同的命令续跑；已完成训练的模型跳过训练，未完成的从 `last.pth` 恢复。

也可以逐折运行；四折目录相互独立：

```powershell
python scripts\cross_region.py --device cuda --folds chimanimani
python scripts\cross_region.py --device cuda --folds hiroshima
python scripts\cross_region.py --device cuda --folds hokkaido
python scripts\cross_region.py --device cuda --folds dominicamaria
```

分阶段运行或仅运行一种模型：

```powershell
python scripts\cross_region.py --stage train --device cuda --models optical
python scripts\cross_region.py --stage train --device cuda --models aef
python scripts\cross_region.py --stage evaluate --device cuda
python scripts\cross_region.py --stage summarize
```

这里 `--models aef` 始终指空间与月份联合模型。训练阶段前需已有 `--stage prepare` 生成的划分。阶段间应保持相同的 `--output-dir`、数据路径、种子和划分比例。初始训练设置保存在 `run_config.json`。续跑时允许调整 batch size 和 `--num-workers`，变更记录在 `runtime_config_history.json`，已有 checkpoint 与已完成阶段会保留。batch size 改变会影响后续优化过程，报告实验时应记录这一变化；其他训练设置改变仍需另选输出目录。

显存不足时，在首次训练前减小 batch size，例如：

```powershell
python scripts\cross_region.py --device cuda --aef-batch-size 8 --optical-batch-size 4 --eval-optical-batch-size 2 --output-dir outputs\cross_region_small_batch
```

快速检查完整链路（仅一折，每地区取两个事件，各阶段只训练一个 epoch；不作为论文结果）：

```powershell
python scripts\cross_region.py --device cuda --folds chimanimani --max-events-per-region 2 --pretrain-epochs 1 --aef-epochs 1 --optical-epochs 1 --aef-batch-size 2 --optical-batch-size 2 --eval-optical-batch-size 1 --output-dir outputs\cross_region_smoke
```

正式实验默认输出：

```text
outputs/cross_region/manifest.json
outputs/cross_region/<region>/source_split.json
outputs/cross_region/<region>/target_test.json
outputs/cross_region/<region>/optical_norm.json
outputs/cross_region/<region>/source_pretrain/best_spatial.pth
outputs/cross_region/<region>/optical/best.pth
outputs/cross_region/<region>/aef/best_spatial.pth
outputs/cross_region/<region>/<model>/test_metrics.json
outputs/cross_region/summary.json
outputs/cross_region/summary.csv
outputs/cross_region/table_rows.tex
```

汇总数值使用 0--1 比例；`table_rows.tex` 可用于填入论文对应表格，不会自动修改论文。任意一折或一种模型缺少测试结果时，不生成四折均值。更换输入清单或种子时应使用新的输出目录。该流程沿用现有数据中的日期标签，不自动重新核验事件日期。

检查实验隔离与汇总逻辑：

```powershell
python -m unittest discover -s tests -v
```

## 公平比较与论文报告

- 所有模型必须使用相同 split JSON 和相同的 898 个验证事件。
- AEF 输入不进行标准化；月份类别权重只能由训练集计算。
- 阈值若不是 0.5，必须在独立 development set 上选择，不能用最终验证集调节。
- 同时报告 overall 与 per-region 结果，不应静默删除 Italy。
- 联合模型至少同时报告 `best.pth` 与 `best_spatial.pth` 的分割结果，以显示辅助任务的真实影响。
- 建议至少运行 3 个随机种子并报告均值和标准差。
- 跨地区结论需要 leave-one-region-out 实验；当前随机事件划分不能证明对未见地区的时间泛化。

数据和 `.pth/.ckpt` 权重已由 `.gitignore` 排除。公开模型建议通过 GitHub Releases 或 Zenodo 发布，并记录数据版本、权重校验值、代码 commit 和完整命令。
