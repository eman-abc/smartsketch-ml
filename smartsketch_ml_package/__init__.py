"""
SmartSketch.AI - ML Package
Forensic-grade AI face generation system

Main components:
- ForensicPromptValidator: Validates and enhances prompts
- FaceGenerator: Generates faces using SDXL + LoRA
- FaceScorer: Scores quality using CLIP
- SmartSketchPipeline: Complete end-to-end pipeline
"""

__version__ = "1.0.0"
__author__ = "Muqaddas Anees, Muqadas Zahra, Eman Chaudhary"
__institution__ = "NUST SEECS"

from .validator import ForensicPromptValidator, validate_prompt
from .generator import FaceGenerator, generate_face
from .scorer import FaceScorer, score_image
from .pipeline import SmartSketchPipeline, generate_forensic_sketch

__all__ = [
    'ForensicPromptValidator',
    'FaceGenerator',
    'FaceScorer',
    'SmartSketchPipeline',
    'validate_prompt',
    'generate_face',
    'score_image',
    'generate_forensic_sketch',
]
