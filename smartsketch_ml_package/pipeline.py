"""
SmartSketch.AI - Complete Pipeline
Integrates validation, generation, and scoring
"""

import random
from datetime import datetime
from typing import Dict, Optional
from PIL import Image

from .validator import ForensicPromptValidator
from .generator import FaceGenerator
from .scorer import FaceScorer


class SmartSketchPipeline:
    """
    Complete SmartSketch pipeline
    
    Workflow:
    1. Validate & enhance prompt (LLM)
    2. Generate face (SDXL + LoRA)
    3. Score quality (CLIP)
    4. Return with full metadata
    """
    
    def __init__(
        self,
        validator: ForensicPromptValidator,
        generator: FaceGenerator,
        scorer: FaceScorer
    ):
        """
        Initialize pipeline with components
        
        Args:
            validator: ForensicPromptValidator instance
            generator: FaceGenerator instance
            scorer: FaceScorer instance
        """
        self.validator = validator
        self.generator = generator
        self.scorer = scorer
        
        print("=" * 60)
        print("🚀 SmartSketch Pipeline Initialized")
        print("=" * 60)
        print("✅ LLM Validator: Ready")
        print("✅ Image Generator: Ready")
        print("✅ Scorer: Ready")
        print("=" * 60)
    
    def generate_sketch(
        self,
        prompt: str,
        case_type: str = "criminal",
        age: Optional[int] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30
    ) -> Dict:
        """
        Main pipeline: Validate → Generate → Score
        
        Args:
            prompt: User's text description
            case_type: "criminal" or "missing"
            age: Person's age (required for criminal cases)
            seed: Random seed for reproducibility
            num_inference_steps: Generation quality (20-50)
        
        Returns:
            Dictionary with:
            - success: bool
            - generation_id: str
            - image: PIL.Image (if successful)
            - scores: dict
            - metadata: dict
            - error: str (if failed)
        """
        
        # Generate unique ID
        generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        timestamp = datetime.now().isoformat()
        
        # ============================================
        # STEP 1: VALIDATE & ENHANCE WITH LLM
        # ============================================
        
        is_valid, enhanced_prompt, validation_meta = self.validator.validate_and_enhance(
            prompt=prompt,
            case_type=case_type,
            age=age
        )
        
        if not is_valid:
            return {
                "success": False,
                "generation_id": generation_id,
                "error": validation_meta['reason'],
                "timestamp": timestamp,
                "original_prompt": prompt,
                "metadata": {
                    "validation": validation_meta
                }
            }
        
        # ============================================
        # STEP 2: GENERATE IMAGE
        # ============================================
        
        try:
            if seed is None:
                seed = random.randint(0, 999999)
            
            generated_image = self.generator.generate_face(
                prompt=enhanced_prompt,
                seed=seed,
                num_inference_steps=num_inference_steps
            )
            
        except Exception as e:
            return {
                "success": False,
                "generation_id": generation_id,
                "error": f"Generation failed: {str(e)}",
                "timestamp": timestamp,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt
            }
        
        # ============================================
        # STEP 3: SCORE THE OUTPUT
        # ============================================
        
        try:
            scores = self.scorer.score_generation(
                image=generated_image,
                prompt=enhanced_prompt
            )
        except Exception as e:
            # Non-fatal: continue with default scores
            scores = {
                "clip_score": 0.5,
                "identity_score": None,
                "combined_score": 50.0,
                "interpretation": f"Scoring failed: {str(e)}"
            }
        
        # ============================================
        # STEP 4: COMPILE RESULTS
        # ============================================
        
        result = {
            "success": True,
            "generation_id": generation_id,
            "image": generated_image,
            "scores": scores,
            "metadata": {
                "timestamp": timestamp,
                "case_type": case_type,
                "age": age,
                "seed": seed,
                "original_prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "validation": validation_meta,
                "num_inference_steps": num_inference_steps,
                "model_version": "sdxl-lora-v1.0"
            }
        }
        
        return result
    
    @classmethod
    def from_pretrained(
        cls,
        lora_path: Optional[str] = None,
        validator_model: str = "Qwen/Qwen2.5-3B-Instruct",
        sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        lora_strength: float = 0.3,
        device: str = "cuda"
    ):
        """
        Convenience method to initialize complete pipeline
        
        Args:
            lora_path: Path to LoRA weights
            validator_model: HuggingFace model ID for validator
            sdxl_model: Path to SDXL base model
            lora_strength: LoRA influence (0.0-1.0)
            device: 'cuda' or 'cpu'
        
        Returns:
            SmartSketchPipeline instance
        """
        validator = ForensicPromptValidator(validator_model, device=device)
        generator = FaceGenerator(sdxl_model, lora_path, lora_strength, device=device)
        scorer = FaceScorer(device=device)
        
        return cls(validator, generator, scorer)


# Convenience function for quick usage
def generate_forensic_sketch(
    prompt: str,
    case_type: str = "criminal",
    age: Optional[int] = None,
    seed: Optional[int] = None,
    pipeline: Optional[SmartSketchPipeline] = None,
    **kwargs
) -> Dict:
    """
    Quick generation function
    
    Args:
        prompt: Text description
        case_type: "criminal" or "missing"
        age: Person's age
        seed: Random seed for reproducibility
        pipeline: Optional pre-initialized pipeline (for performance)
        **kwargs: Additional generation parameters
    
    Returns:
        Result dictionary
    """
    if pipeline is None:
        pipeline = SmartSketchPipeline.from_pretrained()
    
    return pipeline.generate_sketch(
        prompt=prompt,
        case_type=case_type,
        age=age,
        seed=seed,
        **kwargs
    )
