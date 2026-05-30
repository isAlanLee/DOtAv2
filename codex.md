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

## 2026-05-30 14:45:35 +08:00

- 用户要求服务器运行训练命令时同时生成终端日志文件。
- 约定：训练终端输出保存到项目根目录 `log/` 文件夹；模型 checkpoint 和 TensorBoard 仍由原代码保存到 `opencood/logs/`。

## 2026-05-30 15:07:29 +08:00

- 本次重新阅读项目并核对当前进度。
- 当前项目结构：根目录含 `README.md`、`requirements.txt`、`setup.py`、`docs/`、`logreplay/`、`opencood/`；核心训练/推理代码集中在 `opencood/tools/`、模型在 `opencood/models/`、数据集逻辑在 `opencood/data_utils/datasets/`。
- `git status --short` 当前仅显示 `codex.md` 被修改；三个 hypes YAML 目前工作区未显示未提交改动。
- 已确认三个关键 YAML 仍保持服务器路径配置：
  - `point_pillar_intermediate_fusion_lable_free.yaml`: `root_dir=/root/autodl-tmp/opv2v/train`，`validate_dir=/root/autodl-tmp/opv2v/train`，`pseudo_lable_path=/root/autodl-tmp/out`。
  - `point_pillar_intermediate_fusion.yaml`: `root_dir=/root/autodl-tmp/opv2v/train`，`validate_dir=/root/autodl-tmp/opv2v/validate`，`pseudo_lable_path=/root/autodl-tmp/out`。
  - `point_pillar_intermediate_fusion_dota.yaml`: `root_dir=/root/autodl-tmp/opv2v/train`，`validate_dir=/root/autodl-tmp/opv2v/validate`，`iterative_training=True`，`pseudo_lable_path=/root/autodl-tmp/out`。
- 需要后续注意：
  - README 命令中写的是 `point_pillar_intermediate_fusion_label_free.yaml`，但仓库实际文件名为 `point_pillar_intermediate_fusion_lable_free.yaml`。
  - `opencood/tools/MBE.py` 和 `opencood/tools/box_score_for_mbe.py` 仍有作者本地硬编码路径，后续若要服务器跑完整 MBE 流程，需要参数化输入/输出路径。
  - `requirements.txt` 仍未锁定前面建议的 `timm==0.4.12`、`cython==0.29.36`，服务器安装时需继续留意 Python 3.7 兼容性。

## 2026-05-30 15:08:55 +08:00

- 用户希望在 V2V4Real（OPV2V format）数据集上复现。
- 已核对本项目数据读取逻辑：`BaseDataset` 根据 `root_dir`/`validate_dir` 扫描场景目录，每个场景下读取 CAV 目录中的同名 `.yaml` 和 `.pcd` 文件；这与官方 V2V4Real OPV2V format 的 `v2v4real/train`、`v2v4real/validate`、`v2v4real/test` 结构兼容。
- 建议不要覆盖当前 OPV2V 配置，而是复制三份 V2V4Real 专用 YAML，将数据目录改为服务器上的 V2V4Real 路径，例如 `/root/autodl-tmp/v2v4real/train`、`/root/autodl-tmp/v2v4real/validate`、`/root/autodl-tmp/v2v4real/test`，伪标签目录单独使用 `/root/autodl-tmp/out_v2v4real`。
- 后续完整复现仍需先参数化 `MBE.py` 和 `box_score_for_mbe.py` 的输入/输出路径，否则 MBE 阶段仍会使用作者本地硬编码路径。

## 2026-05-30 15:11:29 +08:00

- 用户提供 V2V4Real 数据目录截图：数据位于 `v2v4real/Data/` 下，包含 `train/`、`val/`、`test/` 三个子目录。
- 该结构仍可被当前数据读取逻辑使用；需要注意验证集目录名是 `val`，不是官方说明和此前建议中的 `validate`。
- V2V4Real 专用 YAML 应改为类似：`root_dir=/root/autodl-tmp/v2v4real/Data/train`，训练验证 `validate_dir=/root/autodl-tmp/v2v4real/Data/val`，最终测试时将 checkpoint 下 `config.yaml` 的 `validate_dir` 改为 `/root/autodl-tmp/v2v4real/Data/test`。

## 2026-05-30 15:13:42 +08:00

- 已生成三份 V2V4Real 专用 YAML，避免覆盖当前 OPV2V 配置：
  - `opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free_v2v4real.yaml`
  - `opencood/hypes_yaml/point_pillar_intermediate_fusion_v2v4real.yaml`
  - `opencood/hypes_yaml/point_pillar_intermediate_fusion_dota_v2v4real.yaml`
- 三份配置均使用 `root_dir=/root/autodl-tmp/v2v4real/Data/train` 和 `pseudo_lable_path=/root/autodl-tmp/out_v2v4real`。
- label-free 配置的 `validate_dir` 设置为 `/root/autodl-tmp/v2v4real/Data/train`，用于后续在训练集上生成伪标签；普通训练和 DOtA 迭代训练配置的 `validate_dir` 设置为 `/root/autodl-tmp/v2v4real/Data/val`。
- 已本地轻量加载验证三份 YAML：`yaml_utils.load_yaml` 可正常解析，派生参数为 `anchor_args.W=704`、`H=200`、`grid_size=[704, 200, 1]`。
