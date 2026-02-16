"""
SmartSketch.AI - ML Package (Updated)
"""

__version__ = "1.1.0"  # Updated version
__author__ = "Muqaddas Anees, Muqadas Zahra, Eman Chaudhary"
__institution__ = "NUST SEECS"

from .validator import ForensicPromptValidator, validate_prompt
from .generator import FaceGenerator, generate_face
from .scorer import FaceScorer, score_image
from .sketch_converter import MemoryEfficientSketchConverter, convert_to_sketch  # NEW
from .pipeline import SmartSketchPipeline

__all__ = [
    'ForensicPromptValidator',
    'FaceGenerator',
    'FaceScorer',
    'MemoryEfficientSketchConverter',  # NEW
    'SmartSketchPipeline',
    'validate_prompt',
    'generate_face',
    'score_image',
    'convert_to_sketch',  # NEW
    # 'generate_forensic_sketch',
]