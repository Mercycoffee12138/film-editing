"""
Segment labeling system for analyzing and matching fight segments.

Provides 8 matching criteria for video segment quality:
1. Motion direction: Consistency of movement vectors
2. Action phase: Same action stage (jump apex, swing, charging, etc.)
3. Silhouette: Similarity of subject outlines and edges
4. Camera motion: Similar camera movement types and speeds
5. Composition: Similar subject position, size, and balance
6. Color: Similar color tone, contrast, exposure, temperature
7. Occlusion: Similar foreground occlusion area and direction
8. Texture: Similar textures (smoke, water, fire, fabric, grass)

Strategy: 
- Use AI to label segments comprehensively during analysis phase (all 8 labels)
- Focus on first 4 labels during matching phase (critical for smooth cuts)
- Remaining 4 labels handled during analysis phase for reference
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MotionDirectionLabel:
    """Movement vector consistency: left/right/forward/backward/jump/spin etc."""
    primary_direction: str = ""  # e.g., "charge_right", "jump_vertical", "retreat"
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class ActionPhaseLabel:
    """Action stage consistency: startup, charge, strike, flight, impact, recovery"""
    phases: list[str] = field(default_factory=list)  # e.g., ["charge", "strike", "recovery"]
    primary_phase: str = ""
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class SilhouetteLabel:
    """Subject outline similarity: shape consistency and edge continuity"""
    shape_type: str = ""  # e.g., "crouched", "extended", "airborne", "collapsed"
    similarity_score: float = 0.0  # 0-1
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class CameraMotionLabel:
    """Camera movement consistency: pan/tilt/dolly/zoom and speed"""
    motion_type: str = ""  # e.g., "pan_left", "push_in", "static", "follow"
    speed_class: str = ""  # e.g., "slow", "medium", "fast"
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class CompositionLabel:
    """Subject positioning: position, size, weight distribution"""
    subject_position: str = ""  # e.g., "center", "left_third", "diagonal"
    subject_size: str = ""  # e.g., "close_up", "medium", "wide"
    balance_type: str = ""  # e.g., "symmetric", "weighted_left", "diagonal"
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class ColorLabel:
    """Color consistency: tone, contrast, exposure, temperature, saturation"""
    primary_tone: str = ""  # e.g., "warm", "cool", "neutral"
    contrast_level: str = ""  # e.g., "high", "medium", "low"
    exposure: str = ""  # e.g., "bright", "normal", "dark"
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class OcclusionLabel:
    """Foreground occlusion: area percentage and direction"""
    occlusion_area: str = ""  # e.g., "none", "small", "medium", "large"
    direction: str = ""  # e.g., "top", "left", "bottom", "right", "corners"
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class TextureLabel:
    """Environmental texture: smoke, water, fire, fabric, grass, etc."""
    textures: list[str] = field(default_factory=list)  # e.g., ["dust", "water_spray"]
    primary_texture: str = ""
    confidence: float = 0.0
    description: str = ""


@dataclass(frozen=True)
class SegmentLabels:
    """Complete label set for a fight segment."""
    motion_direction: MotionDirectionLabel = field(default_factory=MotionDirectionLabel)
    action_phase: ActionPhaseLabel = field(default_factory=ActionPhaseLabel)
    silhouette: SilhouetteLabel = field(default_factory=SilhouetteLabel)
    camera_motion: CameraMotionLabel = field(default_factory=CameraMotionLabel)
    composition: CompositionLabel = field(default_factory=CompositionLabel)
    color: ColorLabel = field(default_factory=ColorLabel)
    occlusion: OcclusionLabel = field(default_factory=OcclusionLabel)
    texture: TextureLabel = field(default_factory=TextureLabel)
    
    @property
    def top_4_confidence_average(self) -> float:
        """Average confidence of the critical first 4 labels (for matching phase)."""
        confidences = [
            self.motion_direction.confidence,
            self.action_phase.confidence,
            self.silhouette.confidence,
            self.camera_motion.confidence,
        ]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "motion_direction": {
                "primary_direction": self.motion_direction.primary_direction,
                "confidence": round(self.motion_direction.confidence, 3),
                "description": self.motion_direction.description,
            },
            "action_phase": {
                "phases": self.action_phase.phases,
                "primary_phase": self.action_phase.primary_phase,
                "confidence": round(self.action_phase.confidence, 3),
                "description": self.action_phase.description,
            },
            "silhouette": {
                "shape_type": self.silhouette.shape_type,
                "similarity_score": round(self.silhouette.similarity_score, 3),
                "confidence": round(self.silhouette.confidence, 3),
                "description": self.silhouette.description,
            },
            "camera_motion": {
                "motion_type": self.camera_motion.motion_type,
                "speed_class": self.camera_motion.speed_class,
                "confidence": round(self.camera_motion.confidence, 3),
                "description": self.camera_motion.description,
            },
            "composition": {
                "subject_position": self.composition.subject_position,
                "subject_size": self.composition.subject_size,
                "balance_type": self.composition.balance_type,
                "confidence": round(self.composition.confidence, 3),
                "description": self.composition.description,
            },
            "color": {
                "primary_tone": self.color.primary_tone,
                "contrast_level": self.color.contrast_level,
                "exposure": self.color.exposure,
                "confidence": round(self.color.confidence, 3),
                "description": self.color.description,
            },
            "occlusion": {
                "occlusion_area": self.occlusion.occlusion_area,
                "direction": self.occlusion.direction,
                "confidence": round(self.occlusion.confidence, 3),
                "description": self.occlusion.description,
            },
            "texture": {
                "textures": self.texture.textures,
                "primary_texture": self.texture.primary_texture,
                "confidence": round(self.texture.confidence, 3),
                "description": self.texture.description,
            },
        }


def build_label_analysis_prompt() -> str:
    """Build the prompt for AI to analyze and label fight segments."""
    return (
        "你需要分析这些视频画面，对打斗片段进行8个维度的标签标注，以便后续进行精确的视频匹配和拼接。\n\n"
        "请按以下8个标签维度进行分析，每个标签需要给出置信度(0-1)和描述：\n\n"
        "1. 运动方向匹配(motion_direction):\n"
        "   - 描述运动矢量的方向：向左冲/向右冲/前后推进/上下跃起/转身/撤退等\n"
        "   - 必填字段: primary_direction(主要方向), confidence(0-1), description\n\n"
        "2. 动作相位匹配(action_phase):\n"
        "   - 描述动作所处的阶段：蓄力/启动/摆动/腾空/击中/落地/恢复等\n"
        "   - 必填字段: phases(阶段列表), primary_phase(主要阶段), confidence(0-1), description\n\n"
        "3. 轮廓匹配(silhouette):\n"
        "   - 分析主体外轮廓：身体姿态的相似度、边缘走势的连贯性\n"
        "   - 形状类型: 蜷缩/伸展/腾空/倒地等\n"
        "   - 必填字段: shape_type(形状类型), similarity_score(0-1相似度), confidence(0-1), description\n\n"
        "4. 镜头运动匹配(camera_motion):\n"
        "   - 分析摄像机运动：推/拉/摇/移等，以及运动速度\n"
        "   - 运动类型: 推进/拉退/左摇/右摇/跟随/静止等\n"
        "   - 速度等级: 慢速/中速/快速\n"
        "   - 必填字段: motion_type(运动类型), speed_class(速度), confidence(0-1), description\n\n"
        "5. 构图匹配(composition):\n"
        "   - 分析主体位置、大小、重心分布的接近程度\n"
        "   - 必填字段: subject_position(位置), subject_size(大小), balance_type(重心分布), confidence(0-1), description\n\n"
        "6. 色彩匹配(color):\n"
        "   - 分析主色调、对比度、曝光、色温的接近程度\n"
        "   - 必填字段: primary_tone(主色调), contrast_level(对比度), exposure(曝光), confidence(0-1), description\n\n"
        "7. 遮挡匹配(occlusion):\n"
        "   - 分析前景遮挡的面积和方向\n"
        "   - 遮挡面积: 无/小/中/大\n"
        "   - 方向: 上/下/左/右/四角等\n"
        "   - 必填字段: occlusion_area(遮挡面积), direction(方向), confidence(0-1), description\n\n"
        "8. 纹理匹配(texture):\n"
        "   - 分析烟/水/火/布料/草地等环境纹理\n"
        "   - 必填字段: textures(纹理列表), primary_texture(主要纹理), confidence(0-1), description\n\n"
        "请返回格式严格的 JSON，不要输出markdown代码块，不要补充解释。"
        "JSON 必须包含 motion_direction, action_phase, silhouette, camera_motion, "
        "composition, color, occlusion, texture 这8个顶级键。"
    )


def parse_label_response(response_text: str) -> SegmentLabels | None:
    """Parse AI response to extract segment labels."""
    try:
        # Clean markdown if present
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        # Extract JSON
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end < start:
            return None

        data = json.loads(cleaned[start : end + 1])
        
        # Parse each label
        motion_direction = MotionDirectionLabel(
            primary_direction=str(data.get("motion_direction", {}).get("primary_direction", "")),
            confidence=float(data.get("motion_direction", {}).get("confidence", 0.0)),
            description=str(data.get("motion_direction", {}).get("description", "")),
        )
        
        action_phase_data = data.get("action_phase", {})
        action_phase = ActionPhaseLabel(
            phases=action_phase_data.get("phases", []),
            primary_phase=str(action_phase_data.get("primary_phase", "")),
            confidence=float(action_phase_data.get("confidence", 0.0)),
            description=str(action_phase_data.get("description", "")),
        )
        
        silhouette = SilhouetteLabel(
            shape_type=str(data.get("silhouette", {}).get("shape_type", "")),
            similarity_score=float(data.get("silhouette", {}).get("similarity_score", 0.0)),
            confidence=float(data.get("silhouette", {}).get("confidence", 0.0)),
            description=str(data.get("silhouette", {}).get("description", "")),
        )
        
        camera_motion = CameraMotionLabel(
            motion_type=str(data.get("camera_motion", {}).get("motion_type", "")),
            speed_class=str(data.get("camera_motion", {}).get("speed_class", "")),
            confidence=float(data.get("camera_motion", {}).get("confidence", 0.0)),
            description=str(data.get("camera_motion", {}).get("description", "")),
        )
        
        composition = CompositionLabel(
            subject_position=str(data.get("composition", {}).get("subject_position", "")),
            subject_size=str(data.get("composition", {}).get("subject_size", "")),
            balance_type=str(data.get("composition", {}).get("balance_type", "")),
            confidence=float(data.get("composition", {}).get("confidence", 0.0)),
            description=str(data.get("composition", {}).get("description", "")),
        )
        
        color = ColorLabel(
            primary_tone=str(data.get("color", {}).get("primary_tone", "")),
            contrast_level=str(data.get("color", {}).get("contrast_level", "")),
            exposure=str(data.get("color", {}).get("exposure", "")),
            confidence=float(data.get("color", {}).get("confidence", 0.0)),
            description=str(data.get("color", {}).get("description", "")),
        )
        
        occlusion = OcclusionLabel(
            occlusion_area=str(data.get("occlusion", {}).get("occlusion_area", "")),
            direction=str(data.get("occlusion", {}).get("direction", "")),
            confidence=float(data.get("occlusion", {}).get("confidence", 0.0)),
            description=str(data.get("occlusion", {}).get("description", "")),
        )
        
        texture_data = data.get("texture", {})
        texture = TextureLabel(
            textures=texture_data.get("textures", []),
            primary_texture=str(texture_data.get("primary_texture", "")),
            confidence=float(texture_data.get("confidence", 0.0)),
            description=str(texture_data.get("description", "")),
        )
        
        return SegmentLabels(
            motion_direction=motion_direction,
            action_phase=action_phase,
            silhouette=silhouette,
            camera_motion=camera_motion,
            composition=composition,
            color=color,
            occlusion=occlusion,
            texture=texture,
        )
    except (json.JSONDecodeError, ValueError, KeyError, TypeError):
        return None
