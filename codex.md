# Codex Progress

## 2026-05-30 14:19:55 +08:00

- 创建进展文档，后续按阶段持续更新。
- 初步读取项目根目录：项目为 DOtA++ / DOtA 的官方代码发布，基于 OpenCOOD，包含 `opencood` 主体代码、`logreplay` 回放模块、`docs` 文档。
- README 指向对应论文：Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels，arXiv:2503.08421。
- 初步复现流程来自 README：安装 OpenCOOD/数据集准备，训练 label-free 初始检测器，生成伪标签，运行 MBE/box score 过滤，再用伪标签训练并评估。

## 2026-05-30 14:21:33 +08:00

- 在线确认论文来源：arXiv:2503.08421，CVPR 2025 Open Access 论文 PDF，标题与 README 一致。
- 另查到 DOtA++ 对应后续论文：DOtA++: Unsupervisely and Collaboratively Detect Objects From Multi-Agent Observations With Multi-Modal Prior Constraints，IEEE TPAMI 2026，DOI: 10.1109/TPAMI.2026.3664863，代码地址为 https://github.com/xmuqimingxia/DOtAv2。
- 代码对齐论文流程：
  - 初始无人工标签训练：`opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml` 设置 `lable_free: True`。
  - 伪标签生成：`opencood/tools/inference.py --pseudo_lable_save 0`，输出到 `pseduo_label_val/pre_box_test_full` 和 `pseduo_label_val/pre_score_test_full`。
  - MBE 初筛：`opencood/tools/MBE.py`，但脚本写死作者本地 OPV2V 路径与伪标签路径。
  - MBE 打分：`opencood/tools/box_score_for_mbe.py`，同样有 Windows/Linux 混合的硬编码路径，并依赖 `viewer.viewer`。
  - 迭代训练：`opencood/hypes_yaml/point_pillar_intermediate_fusion_dota.yaml` 设置 `iterative_training: True`，数据集会读取 `pseudo_lable_path/out_pseduo_labels_with_score_v4_{idx}.npy` 和噪声标签文件。
- 当前本机检查：
  - 当前仓库内没有 OPV2V/V2V4Real `.pcd` 数据。
  - `D:\DOtAv2-opv2v` 也是一份代码仓库，不是数据目录。
  - 当前 Python 为 3.13.5；项目文档要求 Python 3.7，且依赖 PyTorch、spconv、CUDA 扩展，需单独创建兼容环境。

## 2026-05-30 14:23:17 +08:00

- 用户确认复现工作在服务器上进行；本地只负责阅读项目、修改代码与配置、准备可推送到服务器的改动。
- 后续不在本地强行搭建 CUDA/数据训练环境；涉及训练、推理、数据路径验证和性能评估的命令应以服务器环境为准。

## 2026-05-30 14:24:08 +08:00

- 整理服务器环境配置命令：优先按 Ubuntu + Conda + Python 3.7 + PyTorch 1.10.2 + CUDA 11.3 + spconv 2.x 路线准备。
- 注意点：项目依赖较旧，`requirements.txt` 中 `numpy/numba/scipy/open3d` 未完全锁死版本，服务器安装时建议显式约束关键版本，避免 pip 拉取不兼容的新版本。

## 2026-05-30 14:34:44 +08:00

- 补全缺失文件 `opencood/hypes_yaml/yaml_utils.py`，来源对齐 OpenCOOD 原项目接口。
- 实现 `load_yaml`、`load_point_pillar_params`、`load_voxel_params`、`load_second_params`、`load_bev_params`、`save_yaml`、`save_yaml_wo_overwriting`。
- 对当前三个 PointPillar YAML 做了本地轻量验证：可正常加载，并生成 `anchor_args.W=704`、`H=200`、`D=1` 与 `grid_size=[704, 200, 1]`。

## 2026-05-30 14:35:49 +08:00

- 用户服务器安装依赖时失败：未锁版本的 `timm` 被解析到 `0.9.12`，继续拉取 `safetensors`，在 Python 3.7 + 阿里源环境下构建 backend 依赖失败。
- 处理建议：DOtA/OpenCOOD 当前代码只需要 `timm.scheduler.cosine_lr`，改为安装旧版 `timm==0.4.12`；同时建议锁定 `cython==0.29.36`，避免 Cython 3 对旧扩展编译造成额外问题。

## 2026-05-30 14:41:56 +08:00

- 用户确认服务器 OPV2V 数据位置为 `/root/autodl-tmp/opv2v`。
- 已修改三个训练配置的硬编码数据路径：
  - `point_pillar_intermediate_fusion_lable_free.yaml`: `root_dir=/root/autodl-tmp/opv2v/train`，`validate_dir=/root/autodl-tmp/opv2v/train`。
  - `point_pillar_intermediate_fusion.yaml`: `root_dir=/root/autodl-tmp/opv2v/train`，`validate_dir=/root/autodl-tmp/opv2v/validate`。
  - `point_pillar_intermediate_fusion_dota.yaml`: `root_dir=/root/autodl-tmp/opv2v/train`，`validate_dir=/root/autodl-tmp/opv2v/validate`。
- 三个配置的 `pseudo_lable_path` 最初统一改为 `/root/autodl-tmp/opv2v/out_xqm_score`，后续按用户要求改为 `/root/autodl-tmp/out`，为后续 MBE/迭代训练阶段对齐输出目录。

## 2026-05-30 14:42:43 +08:00

- 用户指定伪标签目录统一使用 `/root/autodl-tmp/out`。
- 已将三个 hypes YAML 中的 `pseudo_lable_path` 更新为 `/root/autodl-tmp/out`。
