# Codex Progress

## 2026-06-06 13:34:26 +08:00
- 读取用户提到的配置位置：`opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml:58`，参数为 `postprocess.target_args.score_threshold: 0.20`。
- 搜索 `score_threshold` 和 `target_args` 的项目引用，确认 `score_threshold` 出现在 post processor 的预测后处理筛选逻辑中。
- 对比训练标签生成逻辑：`generate_label()` 使用 `pos_threshold` 和 `neg_threshold` 分配正负 anchor；未使用 `score_threshold`。
- 查看 `opencood/tools/train.py` 训练循环，loss 由 `criterion(output_dict, batch_data['ego']['label_dict'])` 计算并反向传播，未调用 `dataset.post_process()`。
- 当前结论：该参数不影响训练时的梯度和模型权重更新；主要影响推理/验证后处理阶段保留哪些预测框。

## 2026-06-06 13:36:43 +08:00
- 核对 `opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml` 当前配置：`pos_threshold: 0.6`，`neg_threshold: 0.45`。
- 查阅本地 `docs/md_files/config_tutorial.md`，OpenCOOD 配置教程中的默认 `target_args` 也为 `pos_threshold: 0.6`、`neg_threshold: 0.45`。
- 核对 PointPillars CVPR 2019 原论文：Car anchor matching 使用 positive/negative thresholds 0.6 和 0.45；Pedestrian/Cyclist 使用 0.5 和 0.35。
- 当前判断：本配置的 anchor 尺寸为车辆尺寸 `l: 3.9, w: 1.6, h: 1.56`，因此与 PointPillars 论文中的 Car/vehicle 阈值设定一致。

## 2026-06-06 13:38:15 +08:00
- 用户澄清“论文”指 DOtA 论文《Learning to Detect Objects from Multi-Agent LiDAR Scans without Manual Labels》，不是 PointPillars 原论文。
- 核对 DOtA CVPR 2025 论文正文：论文描述了采用 PointPillars/AttFuse 作为 detector/fusion 基线，以及 DOtA 的 MBE/LICL、confidence threshold 等设置；未显式给出 `pos_threshold`/`neg_threshold` 或 `0.6/0.45` 的 anchor 正负样本匹配阈值。
- 核对 DOtA 官方 GitHub `xmuqimingxia/DOtAv2` 的 `point_pillar_intermediate_fusion_lable_free.yaml` 与 `point_pillar_intermediate_fusion_dota.yaml`，两者均配置 `pos_threshold: 0.6`、`neg_threshold: 0.45`。
- 当前判断：若以 DOtA 官方代码实现为准，本地配置与官方 DOtAv2 一致；若严格以论文正文为准，论文没有单独声明这两个阈值。

## 2026-06-06 13:46:06 +08:00
- 核对 DOtA 论文 3.2 与 4.2：MBE 判别器阈值为 collision tolerance `phi_r=0.1`、alignment tolerance `phi_o=0.7`，缩放因子 `eta_e=[0.5, 0.2]`。
- 核对本地 `opencood/tools/MBE.py`：`classify_state()` 使用 `if c1 < 0.1 and c2 > 0.7`，与论文 `phi_r/phi_o` 数值一致。
- 核对本地 `opencood/tools/MBE.py`：`scale_var = [1.5, 1.2, 1.0, 0.8, 0.5]`，等价于基于 `eta_e=[0.5, 0.2]` 的放大/缩小尺度，数值与论文一致。
- 注意事项 1：当前 YAML 的 `score_threshold: 0.20` 是 MBE 前候选伪标签生成/后处理阈值，不等于论文中用于分析/展示高召回伪标签的 confidence threshold `0.01` 或 `0.1`。
- 注意事项 2：论文 ICE 公式用距离倒数平方给近距离 agent 更高权重；本地代码中 `score_d = distance_total[i] / sum(distance_total)` 使用的是距离正比权重，这在公式意义上与论文描述不完全一致。

## 2026-06-06 13:52:30 +08:00
- 根据用户要求修改 ICE 实现，文件为 `opencood/tools/MBE.py`。
- 将原先距离正比权重 `distance_total[i] / sum(distance_total)` 修改为距离平方倒数权重：`1 / max(distance^2, 1e-6)` 后归一化，使距离越近的 agent 权重越大，对齐 DOtA 论文 ICE 公式。
- 保留原有 MBE 判别器逻辑 `c1 < 0.1 and c2 > 0.7` 不变。
- 使用 `rg` 确认旧的 `sum(distance_total)` 距离正比实现已无残留；使用 `git diff --check -- opencood/tools/MBE.py` 检查未发现空白错误。

## 2026-06-06 14:30:52 +08:00
- 用户询问 `opencood/tools/box_score_for_mbe.py:534-535` 中读取的 `multi_agent_point{count}.npy` 与 `multi_agent_point_pose{count}.npy` 在哪里生成。
- 全仓搜索当前工作区：仅发现 `box_score_for_mbe.py` 对这两个缓存 npy 的读取，未发现当前活代码中有对应 `np.save` 生成逻辑。
- 搜索 git 历史发现删除前的 `opencood/tools/mvtsa.py` 中有生成代码块；该块循环每个 scenario/cav/timestamp，读取 `.pcd` 和 `.yaml`，用 `pc_2_world()` 转到世界坐标，并用 `remove_ground_points()` 去地面后保存。
- 历史生成路径为 `F:\OPV2V\OPV2V\multi_agent_point_remove_ground\multi_agent_point{count}.npy` 和 `F:\OPV2V\OPV2V\multi_agent_point_pose\multi_agent_point_pose{count}.npy`；该生成块在历史版本中已是注释状态。
- 当前 `opencood/tools/MBE.py` 中也有类似的内存构造流程，但没有保存这两个缓存 npy，只保存 MBE 筛选后的伪标签结果。

## 2026-06-06 14:32:40 +08:00
- 用户追问 `box_score_for_mbe.py` 所需的两个缓存 npy 是否与 MBE 输出的 npy 属于同一类数据。
- 对照 `opencood/tools/box_score_for_mbe.py:534-535`：`multi_agent_point{count}.npy` 是按 scenario 缓存的多车、多帧、去地面后的世界坐标点云；`multi_agent_point_pose{count}.npy` 是对应多车、多帧的 `lidar_pose`。
- 对照 `opencood/tools/MBE.py:424-426`：MBE 输出的是按 timestamp 保存的伪标签框，即 `out_pseduo_labels_v1_{num_timestamp}.npy` 和 `out_pseduo_labels_noise_v1_{num_timestamp}.npy`。
- 当前判断：二者不是同一类数据；前者是 `box_score_for_mbe.py` 计算分数时需要的点云/位姿辅助输入，后者是 MBE 筛选后的正/负伪标签框结果。

## 2026-06-06 14:34:07 +08:00
- 用户询问如果没有 `multi_agent_point{count}.npy` 和 `multi_agent_point_pose{count}.npy` 是否会影响 `box_score_for_mbe.py` 运行。
- 对照 `opencood/tools/box_score_for_mbe.py:534-535`：代码直接 `np.load()` 这两个缓存，没有 `try/except` 或替代生成路径；缺失时会在这里直接 `FileNotFoundError`。
- 对照 `opencood/tools/box_score_for_mbe.py:573-581`：后续使用 `multi_agent_point` 拼接当前时间窗口的多车点云 `dense_points_multi_frame`。
- 对照 `opencood/tools/box_score_for_mbe.py:590-598`：后续使用 `poses` 将伪标签框从 ego/local 坐标变换到世界坐标。
- 当前判断：没有这两个缓存会影响运行，且不仅是可选输入；它们是计算 `out_pseduo_labels_with_score_*` 和 `out_pseduo_labels_noise_with_score_*` 的必要输入。

## 2026-06-06 14:42:20 +08:00
- 用户询问 `box_score_for_mbe.py` 是否是必须运行的流程。
- 核对当前 YAML：`opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml` 中 `iterative_training: False`，因此当前训练配置不会进入读取 `box_score_for_mbe.py` 产物的分支。
- 核对 `opencood/data_utils/datasets/intermediate_fusion_dataset.py:160-169`：当 `iterative_training=True` 时，训练数据集硬编码读取 `out_pseduo_labels_with_score_v4_{idx}.npy` 和 `out_pseduo_labels_noise_with_score_v4_{idx}.npy`。
- 核对 `opencood/data_utils/post_processor/voxel_postprocessor.py:186-191` 与 `opencood/loss/point_pillar_loss.py:134-143`：若伪标签有第 8 列 score，则用于 `targets_score` 加权回归 loss；若没有第 8 列，则代码会 fallback 为 1。
- 当前判断：`box_score_for_mbe.py` 不是所有流程都必须运行；但若要按当前代码执行 DOtA 的迭代伪标签训练并使用 score 加权，则必须先运行它或生成等价的 `with_score` 伪标签文件。

## 2026-06-06 21:27:36 +08:00
- 根据用户要求修改 `opencood/tools/MBE.py`，在每个 scenario 构造完成 `multi_agent_point` 和 `poses` 后保存缓存，供 `box_score_for_mbe.py` 使用。
- 新增缓存保存路径：`/root/autodl-tmp/out_mbe/multi_agent_point_remove_ground/multi_agent_point{count}.npy` 与 `/root/autodl-tmp/out_mbe/multi_agent_point_pose/multi_agent_point_pose{count}.npy`。
- 同步修改 `opencood/tools/box_score_for_mbe.py`，从上述 MBE 缓存目录读取点云/pose，并从 `/root/autodl-tmp/out_mbe` 读取 MBE 输出的正/负伪标签。
- 删除 `box_score_for_mbe.py` 中不参与算分且会阻断运行的旧 `F:\OPV2V\OPV2V\gt_box\...` 读取逻辑；清理旧 Windows 路径。
- `box_score_for_mbe.py` 当前将带 score 的伪标签输出到 `/root/autodl-tmp/out_mbe/score`，后续迭代训练可将 YAML 的 `pseudo_lable_path` 指向该目录。
- 验证：旧硬编码路径搜索无匹配；`python -m py_compile opencood/tools/MBE.py opencood/tools/box_score_for_mbe.py` 通过；`git diff --check -- opencood/tools/MBE.py opencood/tools/box_score_for_mbe.py` 未发现空白错误。

## 2026-06-06 21:35:51 +08:00
- 根据用户要求新增流水线脚本 `scripts/run_readme_pipeline.py`，按 README 顺序执行：初始检测器训练、初始伪标签生成、MBE、MBE box score、DOTA 伪标签训练、最终测试。
- 脚本为每一步在 `pipeline_logs/<timestamp>/` 下保存独立 log，并维护 `pipeline_summary.log`。
- 脚本实现 fail-fast shutdown：任一步命令返回非零、关键产物缺失或 checkpoint 解析失败时，终止当前子进程、停止后续步骤并以非零状态退出；该 shutdown 不执行系统关机。
- 脚本会从训练日志自动解析 checkpoint 目录；也支持通过 `--initial-detector-dir` 或 `--final-checkpoint-dir` 传入已有 checkpoint。
- 脚本会生成临时 DOTA YAML，将 `iterative_training` 设为 `True`，并将 `pseudo_lable_path` 指向 `/root/autodl-tmp/out_mbe/score`，不直接覆盖用户原始 YAML。
- 验证：`python -m py_compile scripts/run_readme_pipeline.py` 通过；`git diff --check -- scripts/run_readme_pipeline.py` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-06 21:38:53 +08:00
- 根据用户要求检查 `scripts/run_readme_pipeline.py` 是否符合 DOtA 论文流程。
- 对照 DOtA 论文：论文概述流程为 Preliminary Label Generation、MBE label filtering、LICL 三阶段；脚本顺序为初始 detector 训练、初始伪标签生成、MBE、`box_score_for_mbe.py` 打分、DOTA 伪标签训练、最终测试，整体顺序一致。
- 对照当前初始 YAML：`opencood/hypes_yaml/point_pillar_intermediate_fusion_lable_free.yaml` 的 `score_threshold` 已为 `0.01`，符合论文中为了高召回保留初始伪标签的低阈值设置。
- 对照当前 DOTA YAML 与模型：脚本生成临时 YAML 保持 `iterative_training=True` 并指向 `/root/autodl-tmp/out_mbe/score`；模型在 `iterative_training` 时使用正/负伪标签计算 contrastive loss，符合 LICL 训练阶段。
- 剩余注意事项：最终测试是否完全符合论文评估，仍依赖最终 checkpoint 下 `config.yaml` 的 `validate_dir` 是否指向 test split；脚本目前未自动验证或修改该路径。

## 2026-06-06 21:40:41 +08:00
- 用户澄清脚本将在 Linux 服务器上运行，失败时需要使用真实 `shutdown` 命令关机。
- 修改 `scripts/run_readme_pipeline.py` 的失败处理：任一步失败时先终止当前子进程、写入 `shutdown.log`，再默认执行 `shutdown -h now`。
- 新增参数 `--shutdown-command`，可改为例如 `sudo shutdown -h now`；新增 `--no-system-shutdown` 用于调试时只停止流水线不关机。
- `--dry-run` 模式不会执行系统关机命令。
- 验证：`python -m py_compile scripts/run_readme_pipeline.py` 通过；`git diff --check -- scripts/run_readme_pipeline.py` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-06 21:47:19 +08:00
- 用户要求流水线使用单卡 GPU。
- 修改 `scripts/run_readme_pipeline.py` 默认参数：`--cuda-devices` 从 `0,1` 改为 `0`，`--nproc-per-node` 从 `2` 改为 `1`。
- 脚本训练命令仍使用 `torch.distributed.launch`，但默认以单进程单 GPU 启动；如需更换 GPU 可传 `--cuda-devices <id>`。
- 验证：`python -m py_compile scripts/run_readme_pipeline.py` 通过；`rg` 确认默认值与 `CUDA_VISIBLE_DEVICES`/`--nproc_per_node` 使用位置；`git diff --check -- scripts/run_readme_pipeline.py` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-06 21:50:33 +08:00
- 用户反馈 Python 流水线中的关机有问题，要求改用 `.sh`。
- 新增 Linux bash 流水线脚本 `scripts/run_readme_pipeline.sh`，按 README/DOtA 流程执行：初始 detector 训练、初始伪标签生成、MBE、MBE box score、DOTA 伪标签训练、最终测试。
- `.sh` 默认单卡：`CUDA_DEVICES=0`、`NPROC_PER_NODE=1`；失败时写入 `shutdown.log` 并默认执行 `shutdown -h now`。
- `.sh` 支持 `--shutdown-command "sudo shutdown -h now"`、`--no-system-shutdown`、`--dry-run`、`--skip-test`、已有 checkpoint 路径等参数。
- `.sh` 每一步单独写入 `pipeline_logs/<timestamp>/NN_step.log`，并维护 `pipeline_summary.log`。
- 验证：`rg` 检查 `.sh` 关键单卡/关机/临时 YAML 逻辑；`git diff --check -- scripts/run_readme_pipeline.sh` 未发现空白错误。本地 Windows 环境没有 `bash`，未能执行 `bash -n`，需在 Linux 服务器上运行 `bash -n scripts/run_readme_pipeline.sh` 做最终 shell 语法检查。

## 2026-06-06 21:57:49 +08:00
- 用户要求单卡训练不要使用分布式训练参数，直接 `python` 运行训练脚本。
- 修改 `scripts/run_readme_pipeline.sh`：删除 `NPROC_PER_NODE` 参数、`--nproc-per-node` CLI、`torch.distributed.launch` 和 `--use_env`，训练步骤改为 `env CUDA_VISIBLE_DEVICES=<id> python opencood/tools/train.py --hypes_yaml ...`。
- 同步修改 `scripts/run_readme_pipeline.py`：删除 `--nproc-per-node` 参数和分布式启动命令，训练命令改为直接运行 `opencood/tools/train.py`。
- 验证：`rg` 搜索 `torch.distributed`、`distributed.launch`、`nproc`、`--use_env` 在两个流水线脚本中无匹配；`python -m py_compile scripts/run_readme_pipeline.py` 通过；`git diff --check -- scripts/run_readme_pipeline.py scripts/run_readme_pipeline.sh` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-06 22:01:43 +08:00
- 用户说明服务器项目路径为 `/root/autodl-fs/DOtAv2`，日志也需要保存在该项目路径下；npy 保存在 `/root/autodl-tmp` 下的对应目录。
- 修改 `scripts/run_readme_pipeline.sh`：默认 `REPO_ROOT=/root/autodl-fs/DOtAv2`，并新增/保留 `--repo-root` 参数可覆盖；`RUN_DIR` 仍为 `$REPO_ROOT/pipeline_logs/<timestamp>`。
- 保持 npy 相关默认路径不变：`MBE_OUTPUT_DIR=/root/autodl-tmp/out_mbe`，`PSEUDO_LABEL_ROOT=/root/autodl-tmp/out_pseudo_lables`。
- 同步修改 `scripts/run_readme_pipeline.py` 默认 `--repo-root` 为 `/root/autodl-fs/DOtAv2`。
- 验证：`rg` 确认 `/root/autodl-fs` 与 `/root/autodl-tmp` 默认路径分工；`python -m py_compile scripts/run_readme_pipeline.py` 通过；`git diff --check -- scripts/run_readme_pipeline.py scripts/run_readme_pipeline.sh` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-06 22:06:32 +08:00
- 用户提供服务器运行日志：`train_initial_detector` 在 Dataset Building 阶段导入 `spconv` 失败。
- 诊断：根因是服务器环境中的 `spconv/cumm/torch/CUDA` 版本或二进制 ABI 不兼容；`sp_voxel_preprocessor.py` 先尝试 `VoxelGeneratorV2`，失败后再 fallback 到 `Point2VoxelCPU3d`，第二次导入触发 `ExternalAllocator already defined`，但根本问题仍是第一次 `spconv.core_cc` 扩展导入失败。
- 修改 `scripts/run_readme_pipeline.sh` 的 `run_step()`：执行步骤命令时临时关闭 `ERR` trap，避免命令失败被误报为 `unexpected error`；现在会明确记录 `step NN <name> failed with return code ...` 后再触发关机逻辑。
- 验证：`rg` 确认 `run_step()` 中 `set +E/set -E` 与失败路径；`git diff --check -- scripts/run_readme_pipeline.sh` 未发现空白错误。

## 2026-06-06 22:08:24 +08:00
- 用户在服务器确认环境：`torch 1.10.2+cu113`、CUDA `11.3`，同时安装了 `cumm 0.5.3`、`cumm-cu113 0.4.11`、`spconv-cu113 2.3.6`。
- 单独测试 `from spconv.utils import Point2VoxelCPU3d` 仍失败，报 `tv::Tensor` 默认参数无法注册。
- 当前判断：普通 `cumm` 与 CUDA 版 `cumm-cu113` 同时存在，极可能导致 `spconv` 绑定加载到不匹配的 `cumm/tensorview`，需要卸载冲突包并只保留与 CUDA 11.3 匹配的一组 `spconv-cu113/cumm-cu113`。

## 2026-06-06 22:10:51 +08:00
- 用户提供 `scripts/run_readme_pipeline.sh --dry-run` 日志：第 5 步 `prepare_dota_hypes` 因 dry-run 未实际生成临时 YAML，但随后仍执行 `require_file`，导致误触发 shutdown。
- 修改 `scripts/run_readme_pipeline.sh`：`GENERATED_DOTA_HYPES` 的 `require_file` 检查仅在 `DRY_RUN=0` 时执行。
- 复查 `.sh` 中其他 `require_glob/require_file`：运行产物检查均已位于 `DRY_RUN=0` 条件下；preflight 基础文件检查保留。
- 验证：`git diff --check -- scripts/run_readme_pipeline.sh` 未发现空白错误。

## 2026-06-07 12:03:58 +08:00
- 用户提供服务器 `mbe_filter` 失败日志：Open3D 读取大量 `/root/autodl-tmp/opv2v/train/.../*.pcd` 失败，随后 `ConvexHull(inter_points_scale)` 因空点集报 `ValueError: No points given`。
- 定位到 `opencood/tools/MBE.py` 的上游问题：`return_pl_frome_single_scenario()` 读取点云时使用了主进程循环末尾残留的全局 `timestamps`，导致不同 scenario 可能用错时间戳去读取不存在的 `.pcd`。
- 修改 `opencood/tools/MBE.py`：每个 scenario 内部根据当前 scenario 的 yaml 文件生成 `scenario_timestamps`，并用它读取当前 scenario 的 `.pcd/.yaml`。
- 增强 `MBE.py` 稳健性：缺失或空 `.pcd` 返回空点云并输出 warning；`remove_ground_points()`、`pc_2_world()` 和 `box_filter()` 增加空点云保护；`ConvexHull` 对少于 4 个点或 Qhull 失败时记为 0，避免崩溃。
- 增强 `classify_state()`：对 ICE/MBE 中的比例计算增加除零保护，距离权重归一化时检查权重和。
- 将 MBE 主流程的 scenario 列表从硬编码 `range(43)` 改为 `range(len(scenario_folders))`，与实际数据目录数量一致。
- 修改 `scripts/run_readme_pipeline.sh`：在 `run_step()` 执行子命令期间显式关闭 `ERR` trap，并在恢复后按返回码进入 `step failed` 关机路径，避免普通步骤失败被误报为 `unexpected error`。
- 验证：`python -m py_compile opencood/tools/MBE.py` 通过；`git diff --check -- opencood/tools/MBE.py scripts/run_readme_pipeline.sh` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示。

## 2026-06-07 12:06:40 +08:00
- 用户要求提供命令检查流水线第 02 步生成的初始伪标签，抽样 200 个。
- 提供服务器端 Python heredoc 检查命令：从 `/root/autodl-tmp/out_pseudo_lables/pre_box_test_full/pre_*.npy` 随机抽样，并检查对应 `/root/autodl-tmp/out_pseudo_lables/pre_score_test_full/score_*.npy` 是否存在、shape 是否匹配、是否包含 NaN/Inf、空标签比例和 score 分布。
- 命令会将检查结果保存到 `/root/autodl-fs/DOtAv2/pipeline_logs/check_step02_labels_<timestamp>.log`。

## 2026-06-07 12:08:00 +08:00
- 用户反馈第 02 步伪标签抽样检查结果：`pre_box_test_full` 下共有 6374 个 `pre_*.npy`，随机抽样 200 个。
- 抽样结果：总框数 13976，空标签文件 0，缺失 score 文件 0，shape 异常 0，NaN/Inf 异常 0。
- score 分布：最小值约 0.0100006，最大值约 0.974636，均值约 0.03997，中位数约 0.01395，95 分位约 0.04738，99 分位约 0.85133。
- 当前判断：第 02 步初始伪标签文件结构与数值完整性正常；score 大量集中在 0.01 附近，符合当前低阈值高召回伪标签生成设定。

## 2026-06-07 12:09:45 +08:00
- 用户询问如何在服务器上从上次断点继续训练。
- 当前断点判断：第 02 步初始伪标签已生成并抽样检查正常，因此无需重新训练初始 detector，也无需重新运行初始伪标签生成；应在同步最新 `MBE.py` 与 `run_readme_pipeline.sh` 后，从第 03 步 `mbe_filter` 继续。
- 建议续跑顺序：`MBE.py` 生成 MBE 筛选标签与点云/pose 缓存，`box_score_for_mbe.py` 生成带 score 伪标签，生成临时 DOTA YAML 指向 `/root/autodl-tmp/out_mbe/score`，最后单卡运行 `opencood/tools/train.py` 进行 DOTA 伪标签训练。

## 2026-06-07 12:13:16 +08:00
- 用户反馈服务器运行 `MBE.py` 报错：`ImportError: cannot import name 'QhullError' from 'scipy.spatial'`，同时出现 `libgomp: Invalid value for environment variable OMP_NUM_THREADS`。
- 修改 `opencood/tools/MBE.py`：`QhullError` 改为兼容导入，优先 `from scipy.spatial import QhullError`，失败时回退到老版本 SciPy 可用的 `from scipy.spatial.qhull import QhullError`。
- 当前判断：`QhullError` 是 SciPy 版本兼容问题；`OMP_NUM_THREADS` 是服务器环境变量值非法，续跑前应执行 `unset OMP_NUM_THREADS` 或设置为合法整数如 `export OMP_NUM_THREADS=1`。
- 验证：`python -m py_compile opencood/tools/MBE.py` 通过；`git diff --check -- opencood/tools/MBE.py codex.md` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示；已清理编译产生的 `opencood/tools/__pycache__`。

## 2026-06-07 18:23:33 +08:00
- 用户反馈服务器运行 `opencood/tools/box_score_for_mbe.py` 报错：`ModuleNotFoundError: No module named 'viewer'`。
- 检查 `box_score_for_mbe.py` 中 `Viewer` 使用情况：实际计算流程未使用 viewer，可视化调用均为注释代码，仅主入口初始化了 `vi = Viewer()`。
- 修改 `opencood/tools/box_score_for_mbe.py`：将 `from viewer.viewer import Viewer` 改为可选导入，缺少 viewer 包时设置 `Viewer = None`，主入口使用 `vi = Viewer() if Viewer is not None else None`。
- 当前判断：该修复不会改变 box score 计算逻辑，只避免服务器无可视化依赖时流程被阻断。
- 验证：`python -m py_compile opencood/tools/box_score_for_mbe.py` 通过；`git diff --check -- opencood/tools/box_score_for_mbe.py codex.md` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示；已清理编译产生的 `opencood/tools/__pycache__`。

## 2026-06-07 19:23:04 +08:00
- 用户要求提供从 `mbe_score` 开始的后续所有指令脚本。
- 新增 `scripts/resume_from_mbe_score.sh`，用于从第 04 步 `score_mbe_boxes` 继续执行后续流程：`box_score_for_mbe.py`、生成临时 DOTA YAML、单卡训练 `train.py`、最终 `inference.py` 测试。
- 脚本默认服务器路径：`REPO_ROOT=/root/autodl-fs/DOtAv2`，`MBE_OUTPUT_DIR=/root/autodl-tmp/out_mbe`，日志保存到 `$REPO_ROOT/pipeline_logs/resume_mbe_score_<timestamp>/`。
- 脚本默认单卡运行：`CUDA_VISIBLE_DEVICES=0`，不使用分布式训练参数；并设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`。
- 脚本默认失败后执行 `shutdown -h now`，支持 `--no-system-shutdown`、`--dry-run`、`--skip-test`、`--final-checkpoint-dir` 等参数。
- 验证：`git diff --check -- scripts/resume_from_mbe_score.sh` 未发现空白错误；`rg` 检查确认脚本包含路径、单卡、shutdown、打分、训练、测试和产物检查逻辑。

## 2026-06-08 16:49:04 +08:00
- 用户反馈最终 AP 极低：IoU 0.3/0.5 约 0.10，IoU 0.7 约 0.09，要求排查前面步骤。
- 初步代码排查：`inference.py` 保存初始伪标签时 `corner_to_center()` 默认输出 `lwh`，DOTA YAML 的 `postprocess.order` 为 `hwl`，训练集 `intermediate_fusion_dataset.py` 在迭代训练时交换第 3 和第 5 列，将 `lwh` 转为 `hwl`，该链路本身是有意匹配配置。
- 发现高风险排查方向：需要确认 MBE 保留比例、MBE score 输出数量、score 权重分布、带 score 伪标签 dtype/shape、最终 checkpoint 的 `config.yaml` 是否真的指向 `/root/autodl-tmp/out_mbe/score`。
- 新增服务器诊断脚本 `scripts/diagnose_dota_artifacts.py`，用于读取现有产物并输出 Step02、MBE、MBE score、数据集索引、伪标签尺寸和最终配置的健康检查结果。
- 验证：`python -m py_compile scripts/diagnose_dota_artifacts.py` 通过；`git diff --check -- scripts/diagnose_dota_artifacts.py` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-08 16:51:28 +08:00
- 用户确认最终模型目录为 `point_pillar_intermediate_fusion_2026_06_07_19_38_34`。
- 后续诊断命令应将 `--final-model-dir` 指向 `/root/autodl-fs/DOtAv2/opencood/logs/point_pillar_intermediate_fusion_2026_06_07_19_38_34`，用于检查该 checkpoint 的 `config.yaml` 是否使用了正确的 `iterative_training`、`pseudo_lable_path`、`validate_dir` 和后处理阈值。

## 2026-06-08 16:57:04 +08:00
- 用户粘贴 `diagnose_dota_artifacts.py` 输出：Step02、MBE、MBE score 文件数量均为 6374，数据集从 first cav yaml 统计帧数也是 6374，且缺失 `.pcd` 数为 0，说明当前文件数量、索引和点云数据完整性基本正常。
- 诊断关键异常 1：MBE 在抽样 1000 帧中只保留 1652 个正伪标签，平均每帧 1.65 个，中位保留率约 2.3%，108/1000 帧没有任何正伪标签；这会导致后续 DOTA 伪标签训练正样本极少，是 AP 极低的首要疑点。
- 诊断关键异常 2：`score_bad: {'neg_nonfinite': 1000}`，说明所有抽样的 rejected/noise scored 文件都存在 NaN/Inf，来源是 `box_score_for_mbe.py` 对空点集或退化点集求 `distance_score.mean()`。
- 诊断补充：初始伪标签和 MBE accepted 伪标签尺寸列都显示为 lwh-like，训练数据集在 `iterative_training` 时交换第 3/5 列后转成 hwl，与 YAML `postprocess.order: hwl` 的链路一致，暂不判为尺寸格式错误。
- 修改 `opencood/tools/box_score_for_mbe.py`：新增 `safe_distance_score()`，对少于 3 个点、ConvexHull 失败、空 corner、非有限 mean 统一返回有限默认 score，并将保存的 score 数组强制为 `float32`，避免生成 NaN/Inf 或 object dtype。
- 修改 `scripts/diagnose_dota_artifacts.py`：读取 checkpoint `config.yaml` 时若 `yaml.safe_load()` 遇到 numpy tag 报 `ConstructorError`，回退到 `yaml.Loader`，避免诊断在 Config Inspection 阶段中断。
- 验证：`python -m py_compile opencood/tools/box_score_for_mbe.py scripts/diagnose_dota_artifacts.py` 通过；`git diff --check -- opencood/tools/box_score_for_mbe.py scripts/diagnose_dota_artifacts.py` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示；已清理 `opencood/tools/__pycache__` 与 `scripts/__pycache__`。

## 2026-06-08 17:14:22 +08:00
- 用户重新运行诊断后确认：`score_bad: {}`，最终模型 `config.yaml` 正确设置 `iterative_training: True`、`pseudo_lable_path: /root/autodl-tmp/out_mbe/score`、`validate_dir: /root/autodl-tmp/opv2v/validate`，只剩高风险项 `MBE keeps very few boxes`。
- 当前判断：AP 低的主要前置问题不是文件缺失、score NaN 或训练配置指向错误，而是 MBE 过滤过严，导致伪标签正样本严重稀疏。
- 新增只读诊断脚本 `scripts/diagnose_mbe_thresholds.py`：抽样重算 MBE 的 `c1/c2`，并分别统计当前 inverse-distance 权重、旧 linear-distance 权重、uniform 权重下的 `pass_c1`、`pass_c2` 和最终 accepted 比例。
- 该脚本用于判断 MBE 低保留率到底是 `c1 < phi_r`、`c2 > phi_o` 哪个条件卡住，以及 ICE 权重修正是否显著降低通过率。
- 验证：`python -m py_compile scripts/diagnose_mbe_thresholds.py` 通过；`git diff --check -- scripts/diagnose_mbe_thresholds.py codex.md` 未发现空白错误；已清理 `scripts/__pycache__`。

## 2026-06-08 17:19:38 +08:00
- 用户提供 `diagnose_mbe_thresholds.py` 输出：当前 inverse 权重下 accepted 为 103/4602，约 2.24%，与已保存 MBE accepted 比例一致。
- 诊断显示 `pass_c1_only` 为 2599/4602，约 56.5%；`pass_c2_only` 仅 170/4602，约 3.7%，因此低保留率主要由 `c2 > 0.7` 条件卡住，而不是 `c1 < 0.1`。
- 对比权重：inverse accepted 2.24%，linear 1.69%，uniform 1.52%；当前论文式 inverse-distance 权重反而保留更多框，因此 ICE 权重修改不是 AP 低的主因。
- `pre_score_inverse_accepted` 中位数约 0.693，而 rejected 中位数约 0.0138，说明 MBE 主要保留高置信度候选，低置信候选几乎全部被滤掉，高召回伪标签没有转化为足够训练正样本。
- 增强 `scripts/diagnose_mbe_thresholds.py`：新增 `threshold_sweep_accept_ratio` 网格输出，用于估计不同 `phi_r/phi_o` 下的 accepted 比例，便于后续选择是否降低 `phi_o` 或调整 MBE 逻辑。
- 验证：`python -m py_compile scripts/diagnose_mbe_thresholds.py` 通过；`git diff --check -- scripts/diagnose_mbe_thresholds.py codex.md` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示；已清理 `scripts/__pycache__`。

## 2026-06-08 17:21:18 +08:00
- 用户要求将 MBE 的距离权重改回原始仓库实现 `score_d = distance_total[i] / sum(distance_total)` 进行对比实验。
- 修改 `opencood/tools/MBE.py`：`classify_state()` 中的距离权重从论文式 inverse squared distance 恢复为线性距离归一化，即 `distance_weight = distance_total` 后除以 `np.sum(distance_weight)`。
- 保留此前添加的稳定性保护：`safe_ratio()` 除零保护、空点云保护、ConvexHull 失败保护、scenario 局部时间戳读取和动态 scenario 数量。
- 当前预期：重新运行 `MBE.py` 后会覆盖 `/root/autodl-tmp/out_mbe/out_pseduo_labels*_v1_*.npy` 和点云/pose 缓存，再运行 `box_score_for_mbe.py` 重新生成 `/root/autodl-tmp/out_mbe/score`，随后需要重新训练 DOTA 模型。
- 验证：`python -m py_compile opencood/tools/MBE.py` 通过；`git diff --check -- opencood/tools/MBE.py codex.md` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示；已清理 `opencood/tools/__pycache__`。

## 2026-06-08 18:06:04 +08:00
- 用户在服务器用原始 linear 距离权重重新运行 `opencood/tools/MBE.py`，耗时约 40 分钟完成。
- 随后运行 `diagnose_dota_artifacts.py`：新的 MBE accepted 抽样 1000 帧仅 940 个，平均每帧 0.94 个，433/1000 帧 accepted 为空，保留率中位数约 1.15%，比 inverse 权重时的 1652 个、平均 1.65 个更差。
- 诊断输出同时显示 `scored_accepted_counts` 仍为 1652，说明用户重新运行 MBE 后尚未重新运行 `box_score_for_mbe.py`，因此 `/root/autodl-tmp/out_mbe/score` 仍是旧 score 产物，与新的 MBE 输出不一致。
- 当前判断：回退原始 linear 距离权重不会改善 AP 低的问题，反而进一步降低 MBE 正伪标签数量；若继续以该 MBE 输出训练，必须先重新运行 `box_score_for_mbe.py`，但更建议调整 `phi_o/c2` 过滤强度。

## 2026-06-08 18:12:29 +08:00
- 用户询问当前是否与 DOtA 论文对齐，以及 AP 低是否可能由 label-free 训练阶段出问题导致。
- 核对 `point_pillar_intermediate_fusion_lable_free.yaml`：`lable_free: True`、`iterative_training: False`、`score_threshold: 0.01`、`validate_dir: /root/autodl-tmp/opv2v/train`，符合用于在 train split 上生成高召回初始伪标签的设置。
- 核对 label-free 数据集逻辑：`generate_object_center_lable_free()` 调用 `project_world_objects_lable_free()`，后者只保留 `object_id` 属于当前 CAV ID 的车辆框，即用每个 agent 自身车辆作为免费标签，符合 DOtA 的 label-free 初始训练思路。
- 核对 DOTA 阶段：最终模型 config 已由服务器诊断确认 `iterative_training: True`、`pseudo_lable_path: /root/autodl-tmp/out_mbe/score`、`validate_dir: /root/autodl-tmp/opv2v/validate`。
- 当前差异：MBE 的 `phi_r=0.1`、`phi_o=0.7` 与论文一致；但用户要求将 `score_d` 改回原始仓库的线性距离归一化后，ICE 距离权重不再与论文公式的 inverse-distance/inverse-square-distance 描述完全一致，而是与原始仓库实现一致。
- 当前判断：已有诊断更支持“MBE 过滤过严导致正伪标签稀疏”而非 label-free 分支结构错误；但仍建议单独评估初始 label-free detector 在 validate split 上的 AP，以确认第一阶段模型质量是否过低。

## 2026-06-08 18:16:48 +08:00
- 用户检查初始 label-free checkpoint `point_pillar_intermediate_fusion_2026_06_06_22_11_36/config.yaml`：确认 `lable_free: true`、`iterative_training: false`、`root_dir/validate_dir` 均为 train、`score_threshold: 0.01`。
- 用户尝试复制 checkpoint 并用 Python `yaml.safe_dump()` 修改临时 config 为 validate split 时失败，错误为 `RepresenterError: cannot represent an object`，原因是 checkpoint config 中包含 numpy 对象 tag，`safe_dump` 无法重新序列化。
- 注意：失败发生在 `open(path, "w")` 之后，临时目录 `/root/autodl-fs/DOtAv2/tmp_eval_initial_label_free/config.yaml` 可能已被截断或写坏，后续必须先删除并重新从原始 `INIT_DIR` 复制。
- 新建议：不要 parse/dump 整个 YAML；对临时 config 使用文本替换修改 `validate_dir` 和唯一的 `score_threshold` 行，避免破坏 numpy tag。

## 2026-06-08 18:33:19 +08:00
- 用户按文本替换方式成功创建初始 label-free 模型的临时 eval config：`lable_free: true`、`iterative_training: false`、`validate_dir: /root/autodl-tmp/opv2v/validate`、`score_threshold: 0.2`。
- 用户运行初始 label-free checkpoint `point_pillar_intermediate_fusion_2026_06_06_22_11_36` 在 validate split 上的 inference：1980 samples，加载 epoch 15。
- 初始 label-free 模型 AP：IoU 0.3 为 0.13，IoU 0.5 为 0.12，IoU 0.7 为 0.11。
- 当前判断：第一阶段 label-free detector 本身 AP 已明显偏低，仅略高于最终 DOTA 模型的 0.10/0.10/0.09；后续 AP 低不是单纯最终推理配置错误，而是初始模型质量偏低与 MBE 过滤过严共同导致。

## 2026-06-08 18:34:19 +08:00
- 用户询问接下来如何排查。
- 排查优先级确定为：先评估初始 label-free 模型在 train split 上的 AP，判断是否训练集也没学起来；再检查初始训练日志和 loss 是否正常下降；最后再基于 MBE threshold sweep 调整 `phi_o/c2`，避免在初始模型质量不明时盲目重训 DOTA 阶段。

## 2026-06-08 18:58:58 +08:00
- 用户上传初始 label-free 训练日志并反馈初始模型在 train split 上 AP 为 IoU 0.3/0.5/0.7 均约 0.13。
- 训练日志显示初始训练完整跑到 epoch 14/15，return code 为 0；loss 从初始极高值下降到末尾接近 0，未见训练崩溃迹象。
- 当前解释：label-free 初始训练只使用当前 CAV/自车相关免费标签，不使用完整场景 GT，因此在用完整 GT 评估时 AP 低是可能且符合预期的；其训练目标本身稀疏且会把大量未标注真实车辆当作背景。
- 当前判断：不能仅凭 AP@0.2 低判定 label-free 分支代码错误；还需要检查低阈值 `score_threshold=0.01` 生成的初始伪标签召回/覆盖率，以及 MBE 是否过度过滤这些低置信候选。

## 2026-06-08 19:02:39 +08:00
- 进一步查看初始训练日志：训练确实跑满 15 个 epoch，最后 epoch 14 的 batch loss 多数已接近 0，说明优化过程没有明显失败；但 train/validate AP 均约 0.13，符合 label-free 稀疏监督导致完整 GT AP 偏低的可能。
- 新增只读诊断脚本 `scripts/diagnose_pseudo_recall.py`，用于将 Step02 初始伪标签 `/root/autodl-tmp/out_pseudo_lables/pre_box_test_full/pre_*.npy` 与完整 GT 对齐，统计不同 score 阈值下的预测数量、GT recall 和 AP。
- 脚本会强制使用完整 GT 逻辑：加载 hypes 后设置 `lable_free=False`、`iterative_training=False`，并按 `--split train/validate` 设置 `validate_dir`。
- 当前目的：判断低阈值 0.01 下初始伪标签是否具备足够 recall。如果 recall 尚可，则主要调 MBE；如果 recall 很低，则需要回头改 label-free 初训或初始伪标签生成。
- 验证：`python -m py_compile scripts/diagnose_pseudo_recall.py` 通过；`git diff --check -- scripts/diagnose_pseudo_recall.py codex.md` 未发现空白错误，仅有 Windows 工作区 LF/CRLF 提示；已清理 `scripts/__pycache__`。
