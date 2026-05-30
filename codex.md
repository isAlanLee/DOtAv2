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

## 2026-05-30 18:43:20 +08:00

- 用户服务器完成第一步 label-free 初始检测器训练，checkpoint 路径为 `/root/autodl-tmp/DOtAv2/opencood/tools/../logs/point_pillar_intermediate_fusion_2026_05_30_14_47_35`。
- 准备第二步：用该 checkpoint 生成训练集初始伪标签。
- 已修改 `opencood/tools/inference.py`：新增 `--pseudo_lable_dir` 参数，默认 `/root/autodl-tmp/out`；伪标签 box 保存到 `pre_box_test_full/`，分数保存到 `pre_score_test_full/`。
- 本地轻量验证：`python -m py_compile opencood/tools/inference.py` 通过。

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

## 2026-05-30 15:36:26 +08:00

- 用户请求 V2V4Real 复现第一步命令。
- 第一阶段建议先训练 label-free 初始检测器，使用 `opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free_v2v4real.yaml`。
- 命令需在服务器项目根目录执行，并将终端输出保存到项目根目录 `log/` 下，例如 `log/train_label_free_v2v4real_YYYYMMDD_HHMMSS.log`。

## 2026-05-30 15:41:03 +08:00

- 用户在服务器运行 V2V4Real label-free 训练时报错：`TypeError: only size-1 arrays can be converted to Python scalars`，位置在 `BaseDataset.calc_dist_to_ego` 计算 `lidar_pose` 距离时。
- 判断原因：V2V4Real 的 OPV2V format 中 `lidar_pose` 可能是 4x4 矩阵，而当前 DOtA/OpenCOOD 代码假设 `lidar_pose` 是 OPV2V 的 6DoF list。
- 按用户要求未修改 `train.py`；已做最小兼容改动：
  - `opencood/utils/transformation_utils.py`：让 `x1_to_x2` 同时支持 6DoF list 和矩阵 pose，并新增 `dist_two_pose`。
  - `opencood/data_utils/datasets/basedataset.py`：`calc_dist_to_ego` 改用 `dist_two_pose`。
  - `opencood/data_utils/datasets/intermediate_fusion_dataset.py`：通信距离过滤改用 `dist_two_pose`。
- 本地轻量测试通过：list/list pose 与 matrix/matrix pose 均可计算距离，`x1_to_x2` 返回 4x4 矩阵。

## 2026-05-30 15:41:49 +08:00

- 用户询问当前 V2V4Real 兼容修改是否会影响 OPV2V 整个流程。
- 评估结论：对 OPV2V 未来重新启动的训练/推理流程应保持兼容；OPV2V 的 `lidar_pose` 是 6DoF list 时，`x1_to_x2` 仍走原有 list/list 分支，距离计算 `dist_two_pose` 的 list 分支也等价于原始公式。
- 正在服务器中运行的 OPV2V 进程通常不会受磁盘代码修改影响，因为 Python 已加载的模块不会自动热更新；除非重启该训练进程才会加载新代码。
- `requirements.txt` 中锁定 `cython==0.29.36`、`timm==0.4.12` 只影响后续新环境安装，不影响当前已运行训练。

## 2026-05-30 15:45:11 +08:00

- 用户重新运行 V2V4Real label-free 训练后已能进入训练循环，但日志显示 `Loc Loss: 0.0000`，例如 epoch 0 的前 80 多个 batch 均为 0。
- 初步判断：这不是 PyTorch scheduler warning 导致；`PointPillarLoss` 中回归损失只在 `pos_equal_one > 0` 的正样本 anchor 上计算，持续为 0 通常说明当前 label-free 目标没有生成正样本。
- 当前 label-free 目标来自 `project_world_objects_lable_free`，其逻辑只保留 `object_id` 与当前 CAV id 匹配的框；V2V4Real 的车辆标注 id 可能与 CAV 文件夹 id 不一致，需在服务器用数据诊断命令确认每个 batch 的 `object_bbx_mask` 和 `pos_equal_one` 数量。

## 2026-05-30 15:48:38 +08:00

- 用户在服务器执行诊断：V2V4Real label-free 数据集长度为 7105，但 idx 0、1、2、10、50、100 均显示 `object_num=0`、`pos_anchor_num=0`。
- 已确认 `Loc Loss=0` 的直接原因是 label-free 阶段没有生成任何目标框。
- 已添加 V2V4Real 专用 fallback，不修改 `train.py`：
  - `point_pillar_intermediate_fusion_lable_free_v2v4real.yaml` 新增 `label_free_use_self_pose: True`。
  - `IntermediateFusionDataset.get_item_single_car` 在原始 label-free id 匹配无框且该 CAV 非 ego 时，调用 `generate_self_pose_label`。
  - `generate_self_pose_label` 使用通信车辆自身 `transformation_matrix` 在 ego 坐标下生成一个固定尺寸车辆框，尺寸来自 anchor 配置 `h=1.56,w=1.6,l=3.9`，z 中心使用 `-1.0`。
- 该 fallback 由 V2V4Real YAML 显式开关控制；OPV2V 配置未开启，应不进入该逻辑。
- 已本地语法检查和 YAML 解析：`python -m py_compile` 通过，`label_free_use_self_pose=True` 可正确加载。

## 2026-05-30 15:49:40 +08:00

- 用户询问 V2V4Real label-free fallback 是否符合原始论文设置。
- 结论：思想上接近论文的 preliminary label generation，即使用协作智能体内部共享的 ego-pose/ego-shape 初始化检测器；但当前 fallback 使用固定 anchor 尺寸和固定 z 中心，属于工程近似，不是严格论文复现。
- 更忠实的做法是从 V2V4Real 元数据中读取协作车自身的真实 ego-shape/尺寸；若元数据没有显式尺寸，则至少应使用论文/数据集统一车辆尺寸配置并在实验记录中说明。

## 2026-05-30 15:51:53 +08:00

- 用户尝试用 `yaml.safe_load` 查看 V2V4Real 样本 YAML，失败：`could not determine a constructor for tag:yaml.org,2002:python/object/apply:numpy.core.multiarray._reconstruct`。
- 原因：V2V4Real YAML 中包含 numpy 对象序列化，不能用 `safe_load`；应使用项目 `opencood.hypes_yaml.yaml_utils.load_yaml` 或 PyYAML 的普通 Loader 读取。

## 2026-05-30 15:54:32 +08:00

- 用户使用项目 `load_yaml` 成功读取 V2V4Real 样本：顶层字段含 `ego_speed`、`gps`、`lidar_pose`、`true_ego_pos`、`vehicles`。
- 样本 `lidar_pose`/`true_ego_pos` 为 4x4 numpy 矩阵；`vehicles` 中 CAV id `1` 包含 `location`、`angle`、`center`、`extent`、`obj_type=Car`，其中 `extent=[1.9899,1.0311,0.8490]` 可作为真实 ego-shape/半尺寸。
- 因此此前固定 anchor 尺寸 fallback 不够贴近论文；已调整为 V2V4Real 专用 `label_free_use_cav_boxes: True`：
  - 从当前 CAV 视角的 `vehicles` 中筛选 object id 属于协作 CAV id 的车辆。
  - 使用该对象 YAML 中的 `location/angle/extent` 生成框，再通过当前 CAV 到 ego 的 `transformation_matrix` 投影到 ego 坐标。
  - 固定尺寸 `label_free_use_self_pose` fallback 保留在代码中但 V2V4Real YAML 默认不启用。
- 本地 `py_compile` 和 YAML 开关解析通过：`label_free_use_cav_boxes=True`。

## 2026-05-30 18:28:14 +08:00

- 用户服务器 V2V4Real label-free 初始检测器训练完成：epoch 14 最后日志 `Loss=0.0783`、`Conf Loss=0.0488`、`Loc Loss=0.0295`。
- checkpoint 保存目录：`/root/autodl-tmp/DOtAv2/opencood/logs/point_pillar_intermediate_fusion_lable_free_v2v4real_2026_05_30_16_01_07`。
- 下一步应使用该 checkpoint 运行 `opencood/tools/inference.py --fusion_method intermediate --pseudo_lable_save 0` 生成训练集初始伪标签；由于 label-free checkpoint 的 `config.yaml` 中 `validate_dir` 指向 `/root/autodl-tmp/v2v4real/Data/train`，该步骤会遍历训练集。

## 2026-05-30 18:38:23 +08:00

- 用户已运行 V2V4Real 初始伪标签生成命令，`inference.py` 找到 7105 samples，加载 epoch 15 checkpoint，7105 iter 正常完成。
- 终端输出 AP@0.3/0.5/0.7 均为 0.00。该值不一定阻断伪标签流程，因为当前步骤主要目标是保存 `pseduo_label_val/pre_box_test_full/pre_*.npy` 和 `pseduo_label_val/pre_score_test_full/score_*.npy`；AP 可能受 label-free 初始检测器质量、V2V4Real GT 坐标/评估适配、或预测为空影响。
- 下一步需先检查伪标签文件数量和非空比例，再决定是否进入 MBE。

## 2026-05-30 18:39:52 +08:00

- 用户检查伪标签输出后发现 `box files=0`、`score files=0`。
- 定位原因：`opencood/tools/inference.py` 中 `--pseudo_lable_save` 未声明 `type=int`，命令行传入 `0` 会成为字符串 `"0"`，而保存分支判断为 `opt.pseudo_lable_save == 0`，导致分支不执行。
- 已修复 `inference.py`：
  - `parser.add_argument('--pseudo_lable_save', type=int, default=1, ...)`。
  - 当 `pseudo_lable_save == 0` 时自动创建 `pseduo_label_val/pre_box_test_full` 和 `pseduo_label_val/pre_score_test_full`。
  - 若某帧 `pred_box_tensor is None`，保存空数组，避免保存分支报错并保持帧索引连续。
- 已本地 `python -m py_compile opencood/tools/inference.py` 通过。

## 2026-05-30 15:34:42 +08:00

- 用户服务器执行 `python setup.py develop` 时再次失败：`requirements.txt` 中未锁定的 `timm` 被解析到新版，新版继续拉取 `safetensors`，而 Python 3.7 环境下拿到的 `safetensors-0.8.0rc0.tar.gz` 没有传统 `setup.py`，导致 easy_install 报错。
- 已修改 `requirements.txt`：
  - `timm` 锁定为 `timm==0.4.12`，避开新版 `timm -> safetensors` 依赖链，且满足当前代码只使用 `timm.scheduler.cosine_lr.CosineLRScheduler` 的需求。
  - `cython` 锁定为 `cython==0.29.36`，降低旧版 CUDA/C++ 扩展在 Python 3.7 环境下被 Cython 3 影响的风险。
- 服务器应重新同步该文件后执行依赖安装；若环境中已经装入不兼容新版 `timm/safetensors`，建议先卸载二者再重新安装。
