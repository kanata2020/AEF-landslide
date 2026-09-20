# AEF Landslide

[English](README.md) | [简体中文](README.zh-CN.md)

论文 **Spatiotemporal Landslide Detection Using Multimodal AlphaEarth Foundation Model Embeddings** 的研究代码，包含 AEF 空间分割与发生月份联合预测模型、Sentinel-2/DEM 3D-UNet 基线，以及四折留一地区迁移验证。

## 安装

使用 Python 3.10 或更新版本。以下命令均在本仓库根目录（包含 `pyproject.toml` 的目录）执行。建议创建独立环境，避免导入计算机上另一份同名的 `landslide_benchmark` 项目。

```bash
python -m venv .venv
```

PowerShell 中用 `.venv\Scripts\Activate.ps1` 激活；Linux/macOS 中用 `source .venv/bin/activate`。先安装适合本机 CUDA 环境的 [PyTorch](https://pytorch.org/get-started/locally/)，再安装本仓库：

```bash
python -m pip install -e .
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

示例使用单行命令和正斜杠，可用于 PowerShell 和常见 Unix shell。没有 CUDA 时使用 `--device cpu`。

## 数据与库内划分文件

从 [Sen12Landslides](https://github.com/PaulH97/Sen12Landslides) 获取原始 NetCDF 样本，并准备匹配事件的年度、64 通道 [AlphaEarth Foundations embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) GeoTIFF。仓库不包含影像和训练权重；代码读取已准备好的样本，不负责下载数据或导出 embeddings。

默认目录结构：

```text
<repository>/
  splits/
    dataset_split_resolved.json       # 原始五地区划分
    dataset_split_four_regions.json   # 默认四地区划分
  data/
    aef/<region>_AEF_<event_id>.tif
    s2/<region>_s2_<event_id>.nc
    optical_norm.json                 # 由训练样本生成
  scripts/
  src/landslide_benchmark/
  tests/
  outputs/
```

AEF 输入须匹配事件年份与样本范围。现有读取逻辑会将 AEF 数组顺时针旋转 90 度，中心裁剪或补零至 128×128，并将 NaN/Inf 替换为零。准备 GeoTIFF 时需匹配这一方向约定，并检查其与 NetCDF 掩膜的对齐。光学样本包含时序 Sentinel-2 波段和 DEM；发生月份从 NetCDF 的 `event_date` 属性提取。

两份划分均纳入 Git。`dataset_split_resolved.json` 保留原始全部事件编号、顺序以及训练/验证归属，仅把计算机专属的 S2 绝对路径替换为文件名，由 `--s2-dir` 解析。`dataset_split_four_regions.json` 仅移除 Italy，其余样本不重新划分，作为当前四地区论文的默认划分。

| 地区名称 | 原始训练事件 | 原始验证事件 |
|---|---:|---:|
| chimanimani | 443 | 190 |
| hiroshima | 621 | 243 |
| hokkaido | 200 | 90 |
| dominicamaria | 423 | 220 |
| italy | 408 | 155 |
| 原始合计 | 2095 | 898 |
| 默认四地区合计 | 1687 | 743 |

Dominica 在文件名及命令行中的名称为 `dominicamaria`。如需运行历史五地区实验，准备、训练和评估时均显式指定 `--split-json splits/dataset_split_resolved.json`，并使用独立的输出目录和归一化文件。

数据可保存在库外，向相应命令传入 `--aef-dir /path/to/aef --s2-dir /path/to/s2` 即可；带空格的路径需加引号。运行不再依赖原父项目目录。划分来源、转换方式和校验值记录于 [splits/provenance.json](splits/provenance.json)。

## AEF 联合模型：固定划分

模型输出像素级掩膜及 13 类时间预测（1—12 月和 `no-event`）。网络使用 96 通道共享 stem、五个全分辨率 Residual-SE 模块，以及传入共享 stem 时梯度缩放为 0.1 的时间分支。损失为 `0.7 * BCE + 0.3 * Dice + 0.25 * temporal_loss`；时间监督采用仅由训练集计算的类别权重和循环标签平滑。

如需先空间预训练、再联合训练，两阶段使用相同训练划分：

```bash
python scripts/train_spatiotemporal.py --device cuda --no-spatial-init --temporal-weight 0 --output-dir outputs/spatial_pretrain
python scripts/train_spatiotemporal.py --device cuda --spatial-init outputs/spatial_pretrain/best_spatial.pth
```

预训练沿用联合网络，将时间损失权重设为零，没有另设 AEF 单空间模型。联合阶段仅载入共享/空间路径权重，前五个 epoch 冻结这些模块，时间分支独立初始化。显式指定的权重文件不存在时会报错。不传 `--spatial-init` 时从随机权重开始，并且不执行初始空间冻结。

默认最多训练 300 epochs，batch size 16，共享/空间学习率 `3e-5`、时间学习率 `1e-4`，预热八个 epoch，CUDA AMP，验证空间 IoU 连续 60 epochs 不提升时早停。完整选项可通过 `--help` 查看。

```bash
python scripts/evaluate_spatiotemporal.py --device cuda --checkpoint outputs/spatiotemporal_v2/best_spatial.pth --batch-size 16 --num-workers 0
```

`best_spatial.pth` 按验证空间 IoU 选择。`best.pth` 按 `0.7 * IoU + 0.3 * temporal macro-F1` 选择，是评估脚本未传 `--checkpoint` 时的默认权重。`last.pth` 用于从下一 epoch 续训；中断的未完成 epoch 会重新运行。固定划分评估报告写入 `outputs/reports/spatiotemporal_val_metrics.json`，属于验证集结果，不是独立目标地区测试结果。

## 光学基线：固定划分

训练前，仅使用选定训练集拟合归一化统计：

```bash
python scripts/prepare_optical_norm.py
python scripts/train_optical.py --device cuda --amp --batch-size 4 --num-workers 0
python scripts/evaluate.py --model optical --device cuda --batch-size 2
```

归一化默认保存到 `data/optical_norm.json`；准备时可用 `--output`，训练和评估时可用 `--norm-json` 更改路径。缓存会拒绝与训练事件指纹不符的统计。光学训练默认仍为 80 epochs、batch size 12、学习率 `1e-3`，按验证 IoU 选择权重；上面的示例使用较小 batch size 以降低内存占用。评估默认使用 `outputs/optical/best.pth`，也可显式指定 `--checkpoint`，不会回退到其他项目的权重。

## 四折跨地区迁移验证

每折将一个完整地区留作测试，其余三个地区各自按约 70%/30% 划分为源训练集与源验证集。程序先合并输入 JSON 中的 train/val 清单，再构建这些新划分。即使输入原始五地区清单，跨地区实验也会排除 Italy。

| 留出地区 | 源训练事件 | 源验证事件 | 目标测试事件 |
|---|---:|---:|---:|
| Chimanimani | 1258 | 539 | 633 |
| Hiroshima | 1096 | 470 | 864 |
| Hokkaido | 1498 | 642 | 290 |
| Dominica | 1251 | 536 | 643 |

两种模型共用地区划分。归一化和月份类别权重仅使用源训练样本计算；每折独立进行 AEF 空间预训练。权重选择和早停只使用源验证集。目标样本仅用于最终推理，固定阈值 0.5，不做目标地区微调。此实验中，光学影像和掩膜同步对齐至联合模型的掩膜存储方向，并检查两者的参考掩膜一致。

先准备划分，再依次运行全部四折（顺序执行，不是同时训练）：

```bash
python scripts/cross_region.py --stage prepare
python scripts/cross_region.py --device cuda --aef-batch-size 8 --optical-batch-size 4 --eval-optical-batch-size 2 --num-workers 0
```

每折默认空间预训练最多 300 epochs、联合训练最多 300 epochs、光学训练最多 80 epochs，早停 patience 为 60。上述命令显式减小 batch size，没有改变 epoch 上限。使用 `--folds chimanimani` 可只运行一折，其他选项为 `hiroshima`、`hokkaido`、`dominicamaria`；使用 `--models optical` 或 `--models aef` 可只运行一种模型。这里 `aef` 始终指联合模型。

续跑时重新执行命令并保持输出目录不变，已完成阶段会跳过。允许修改 batch size 和 `--num-workers`，变化记录到 `runtime_config_history.json`。batch size 改变会影响后续优化，应纳入实验记录。其他训练设置和数据划分需保持一致，否则使用新输出目录。

如果此前使用 `--stage train` 单独训练，随后运行：

```bash
python scripts/cross_region.py --stage evaluate --device cuda --aef-batch-size 8 --eval-optical-batch-size 2 --num-workers 0
python scripts/cross_region.py --stage summarize
```

各阶段保持相同的自定义 `--output-dir`、数据路径、种子和验证比例。默认 `--stage all` 已包含训练、评估及汇总。

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

报告包含滑坡类 Precision、Recall、F1、IoU 及 AEF 的 13 类月份准确率（包含 `no-event`），汇总数值使用 0—1 比例。最终均值是四个地区指标的等权算术平均，不是按像素或样本汇总的指标。两种模型、四个地区共八份报告全部存在才会生成汇总。LaTeX 表格行仅导出，不自动修改论文。

## 小规模检查与常见问题

先运行一折小规模实验检查数据和训练链路；其输出不能作为论文结果：

```bash
python scripts/cross_region.py --device cuda --folds chimanimani --max-events-per-region 2 --pretrain-epochs 1 --aef-epochs 1 --optical-epochs 1 --aef-batch-size 2 --optical-batch-size 2 --eval-optical-batch-size 1 --num-workers 0 --output-dir outputs/cross_region_smoke
python -m unittest discover -s tests -v
```

遇到 CUDA 显存不足时，降低对应模型的 batch size。遇到 Windows 共享内存错误 1455 时，先使用 `--num-workers 0`，并检查系统内存和分页文件容量。外部数据路径加引号即可使用，不必复制影像。可用 `python -c "import landslide_benchmark; print(landslide_benchmark.__file__)"` 确认当前环境导入的是本仓库。

## 结果解释与复现记录

缺失或无效日期统一编码为 `no-event`，这一编码本身不能证明没有滑坡。应检查联合评估报告中的 `no_event_samples_with_positive_mask`。月份可能与地区高度相关，年度 embedding 本身不能证明精确发生时间。固定划分评估提供时间基线，留一地区实验则衡量不同的泛化场景。代码不会自动人工校正源数据日期。

记录划分文件、随机种子、训练参数（包括资源参数变更历史）、数据版本及所选权重。影像、权重和生成结果不纳入 Git，库内 `splits/` 则保留版本管理。影像从上游数据源获取并遵循其适用条款。本仓库不附带最终论文指标或预训练权重。
