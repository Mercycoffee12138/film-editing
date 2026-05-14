# 打斗片段视频匹配标签系统

## 系统概述

本系统为视频打斗片段拼接添加了基于大模型的8维标签分析和匹配机制。

### 核心设计原理

**分析阶段 + 匹配阶段** 的两阶段策略：
- **分析阶段（Stage 02）**: 为每个检测到的打斗片段生成8种详细标签
- **匹配阶段（Stage 04）**: 在选择候选片段时，优先使用前4种关键标签实现精确匹配

### 8种标签维度

| 维度 | 描述 | 匹配阶段使用 |
|------|------|------------|
| 1. **运动方向** | 运动矢量方向：前后、左右、上下、转身等 | ✅ 关键 |
| 2. **动作相位** | 动作阶段：蓄力→启动→摆动→腾空→击中→落地 | ✅ 关键 |
| 3. **轮廓匹配** | 主体外形一致性：身体姿态、边缘走势 | ✅ 关键 |
| 4. **镜头运动** | 摄像机运动：推/拉/摇/移及速度 | ✅ 关键 |
| 5. **构图匹配** | 主体位置、大小、重心分布 | 📋 参考 |
| 6. **色彩匹配** | 主色调、对比度、曝光、色温 | 📋 参考 |
| 7. **遮挡匹配** | 前景遮挡面积和方向 | 📋 参考 |
| 8. **纹理匹配** | 烟、水、火、布料等环境纹理 | 📋 参考 |

## 工作流程

### 1. 标签提取流程（Stage 02）

```python
检测到打斗片段
    ↓
从片段中提取3个关键帧（头、中、尾）
    ↓
使用Qwen Vision大模型分析帧画面
    ↓
获取8种标签的AI分析结果
    ↓
解析JSON响应，提取标签信息
    ↓
标签附加到FightSegmentRecord
```

**关键函数**：
- `_extract_segment_keyframes()`: 提取3个关键帧
- `_analyze_segment_labels()`: 调用大模型进行标签分析
- `build_label_analysis_prompt()`: 构建标签分析提示词

### 2. 标签匹配流程（Stage 04）

```python
为音乐高光选择候选打斗片段
    ↓
对每个候选段落计算标签匹配分数
    ↓
4种关键标签的加权平均：
    - 运动方向匹配 (30%)
    - 动作相位匹配 (30%)
    - 轮廓匹配 (25%)
    - 镜头运动匹配 (15%)
    ↓
如果标签分数 > 0.5，应用0-0.05的奖励加分
    ↓
综合考虑其他因素（故事角色、连续性等）选择最佳片段
```

**关键函数**：
- `calculate_label_match_score()`: 计算两个片段的标签相似度 (0-1)
- 集成到 `_pick_best_candidate()` 的 `weighted_score` 计算

## 数据结构

### FightSegmentRecord （扩展）

```python
@dataclass(frozen=True)
class FightSegmentRecord:
    # ... 原有字段 ...
    labels: dict[str, Any] | None = None  # 新增：标签字典
```

### SegmentLabels 结构

```json
{
  "motion_direction": {
    "primary_direction": "charge_right",
    "confidence": 0.95,
    "description": "..."
  },
  "action_phase": {
    "phases": ["charge", "strike"],
    "primary_phase": "strike",
    "confidence": 0.87,
    "description": "..."
  },
  "silhouette": {
    "shape_type": "extended",
    "similarity_score": 0.82,
    "confidence": 0.91,
    "description": "..."
  },
  "camera_motion": {
    "motion_type": "push_in",
    "speed_class": "medium",
    "confidence": 0.78,
    "description": "..."
  },
  "composition": { ... },
  "color": { ... },
  "occlusion": { ... },
  "texture": { ... }
}
```

## 使用指南

### 启用标签系统

标签系统自动启用（无需配置），当以下条件满足时：

```bash
# 设置Qwen Vision API
export ZZZ_API_KEY="your-api-key"
# 或
export DASHSCOPE_API_KEY="your-api-key"
```

### 标签匹配权重

当前的4个关键标签权重在 `segment_matching.py` 中定义：

```python
weights = {
    "motion_direction": 0.30,  # 运动方向（最重要）
    "action_phase": 0.30,      # 动作相位
    "silhouette": 0.25,        # 轮廓匹配
    "camera_motion": 0.15,     # 镜头运动
}
```

### 配置标签提取帧数

在 `stage_02_detect_fight_segments.py` 的 `_analyze_segment_labels()` 中：

```python
frame_paths = _extract_segment_keyframes(
    config,
    trimmed_path,
    duration,
    segment.start,
    segment.end,
    video_index,
    segment_index,
    frames_per_segment=3,  # 修改此参数
)
```

### 调整匹配奖励权重

在 `stage_04_match_segments.py` 的候选评分函数中：

```python
label_match_bonus = 0.0
if label_match_score > 0.5:
    label_match_bonus = (label_match_score - 0.5) * 0.1  # 修改系数
```

## 性能和成本考虑

### API调用

- 每个打斗片段产生 **1 次** AI分析调用（3张图片）
- 调用时机：Stage 02 检测阶段
- 不影响匹配阶段的性能（标签已预先计算）

### 存储

- 每个片段增加约 1-2KB 的标签数据
- 对整体输出文件大小影响很小

### 处理时间

- 标签提取：约 2-5 秒/片段（取决于网络）
- 标签匹配：< 1 毫秒（在匹配时进行）

## 扩展机制

### 添加新标签维度

1. 在 `segment_labels.py` 中添加新的 `@dataclass`
2. 更新 `SegmentLabels.to_dict()` 方法
3. 更新 `parse_label_response()` 函数
4. 在 `build_label_analysis_prompt()` 中添加新维度的说明

### 自定义匹配权重

在 `segment_matching.py` 的 `calculate_label_match_score()` 中修改权重：

```python
weights = {
    "motion_direction": 0.40,  # 增加运动方向的权重
    "action_phase": 0.25,
    "silhouette": 0.20,
    "camera_motion": 0.15,
}
```

## 故障排查

### 问题：标签为 None

**原因**：
- 未配置Qwen Vision API
- API调用超时或错误
- 视频帧提取失败

**解决**：检查日志中的错误信息，确保API配置正确

### 问题：标签与预期不符

**原因**：
- AI模型理解差异
- 光线或视角不清晰
- 提示词需要优化

**解决**：修改 `build_label_analysis_prompt()` 中的提示词

### 问题：匹配效果没有改进

**原因**：
- 标签匹配权重过低
- 阈值（0.5）设置不当
- 段落质量问题

**解决**：调整匹配奖励系数或权重

## 示例：标签分析结果

```json
{
  "motion_direction": {
    "primary_direction": "charge_right",
    "confidence": 0.95,
    "description": "武者以高速向右冲锋，身体前倾，重心转移至右腿"
  },
  "action_phase": {
    "phases": ["startup", "charge"],
    "primary_phase": "charge",
    "confidence": 0.92,
    "description": "处于蓄力和冲锋阶段的中期，动作还未到达顶点"
  },
  "silhouette": {
    "shape_type": "extended",
    "similarity_score": 0.84,
    "confidence": 0.88,
    "description": "身体充分伸展，四肢打开，与前一个镜头形态相似度高"
  },
  "camera_motion": {
    "motion_type": "follow_tracking",
    "speed_class": "medium_fast",
    "confidence": 0.79,
    "description": "摄像机以中快速度跟踪主体，保持主体在画面中央偏右"
  }
}
```

## 参考文献

- `src/cutting_pipeline/segment_labels.py` - 标签数据模型
- `src/cutting_pipeline/segment_matching.py` - 标签匹配函数
- `src/cutting_pipeline/stage_02_detect_fight_segments.py` - 标签提取集成
- `src/cutting_pipeline/stage_04_match_segments.py` - 匹配阶段集成
