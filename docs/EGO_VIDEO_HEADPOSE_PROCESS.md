# Ego 视频 Headpose 标注流程

本文描述 ego 视频原始数据阶段的 headpose 标注流程。在这个阶段，
headpose 等价为头戴/第一视角相机的 camera pose。目标是用 Pi3/Pi3X
为每个视频生成 pose sidecar 文件，不在这一阶段构建下游训练用的
`.pt` 样本。

## 范围

- 输入：原始 ego 视频或已抽帧的图片目录。
- 输出：每个视频对应的 camera/head pose 标注文件。
- pose 定义：Pi3/Pi3X 输出的 OpenCV camera-to-world `4x4` 矩阵。
- 推荐模型路径：长视频优先用 Pi3XVO；短片段或已切好的窗口可以直接用 Pi3X。

这个阶段要保留足够的元信息，方便后续做坐标转换、质量筛选和数据集构建。

## 推荐输出

建议每个视频保存一个 sidecar 文件。张量数据优先用 `.npz`，同时保存一个
JSON 兼容的 metadata。最小 schema 如下：

```text
video_id: str
source_video_path: str
frame_indices: int64[N]
timestamps: float64[N]
c2w_opencv: float32[N,4,4]
valid_mask: bool[N]
conf_mean: float32[N]
conf_p10: float32[N]
conf_p50: float32[N]
pose_jump_score: float32[N]
interval: int
image_size: int[2]          # Pi3/Pi3X 实际使用的 height, width
original_image_size: int[2] # 原始视频 height, width
source_model: str           # 例如 Pi3XVO
checkpoint: str
pi3_git_commit: str
created_at: str
```

`c2w_opencv` 应保存 Pi3/Pi3X 的直接输出坐标约定。MemWorld 相关的坐标系转换
放到后续阶段处理。

## 处理流程

1. 从每个原始视频抽帧。

   记录 `frame_indices`、时间戳、原始分辨率、抽帧间隔，以及 resize/crop 策略。
   不要覆盖原始视频。

2. 长视频用 Pi3XVO 跑标注。

   Pi3XVO 会用 overlap 对齐相邻 chunk，更适合长视频连续轨迹。只有短视频或
   已经切好的短窗口，才建议直接用 Pi3X 单次推理。

3. 收集 pose 和 confidence 输出。

   将 `camera_poses` 保存为 `c2w_opencv`。对 Pi3/Pi3X 的 `conf` 先做 sigmoid，
   再计算每帧的 mean、p10、median 等统计量。这些统计量用于质量判断，不等同于
   绝对正确性。

4. 计算简单轨迹诊断指标。

   标记旋转/平移突变、近似静止或低视差片段、低 confidence 帧。第一阶段可以先
   保留标注结果，通过 `valid_mask` 或 segment 级别的质量标记让后续任务决定阈值。

5. 切分明显不连续的视频段。

   硬切、相机重置、严重损坏片段、多个无关视频拼接在一起的情况，都应该拆成独立
   segment。不要强行把视频剪辑点两侧连成同一条连续轨迹。

6. 保存 sidecar 标注。

   写文件时建议使用原子写入。原始视频、抽帧结果和 sidecar 必须能通过
   `video_id` 和 `source_video_path` 追溯。

## 质量标记

第一轮处理应优先保证覆盖率，而不是过早丢弃数据。建议记录以下标记：

| 标记 | 含义 |
| --- | --- |
| `low_confidence` | Pi3/Pi3X 的点云 confidence 偏低。 |
| `pose_jump` | 相邻帧旋转或平移出现突变。 |
| `static_or_low_parallax` | 运动太小，translation scale 可靠性较低。 |
| `scene_cut` | 视频中疑似存在硬切或相机重置。 |
| `dynamic_foreground` | 大面积动态前景主导了几何估计。 |
| `short_segment` | 片段过短，不适合稳定使用。 |

一般来说 rotation 比 translation 更可靠。translation scale 应视为近似值，只有经过
额外验证后才适合作为强监督真值。

## 批处理策略

推荐两阶段策略：

1. 全量粗标。

   先用较大的抽帧间隔跑完所有视频，例如 `4`、`5` 或 `8`，尽量覆盖更多数据并估计
   每个视频的质量。

2. 重点重跑。

   对高价值或质量较好的视频，用更小的抽帧间隔重跑。sidecar 中必须记录 interval
   和模型元信息，便于比较不同版本结果。

这样后续阶段可以根据任务需要选择 dense pose、sparse pose，或只选高质量片段。

## 下游边界

本阶段不构建 MemWorld `.pt` 样本。后续数据构建阶段可以读取 sidecar，再执行：

- OpenCV c2w 到目标坐标系的转换。
- 相对 anchor frame 的 relative pose 计算。
- long/short/dynamic context frame 选择。
- VAE latent 提取和训练样本构建。

把 Pi3 标注阶段和下游样本构建阶段拆开，可以降低排查成本；当后续采样规则变化时，
也不需要重复运行 Pi3。

## 局限

- Pi3 根据视觉几何估计相机运动；它不会估计“头部相对身体”的姿态。
- 输出坐标系相对重建场景，不应视为全局世界坐标。
- 运动模糊、弱纹理、镜面、硬切、大面积动态前景都会降低标注可靠性，需要更强过滤。
- Pi3X 的 metric translation scale 是近似值，作为真值使用前需要额外验证。
