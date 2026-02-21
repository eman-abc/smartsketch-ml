"""
SmartSketch.AI - Complete Pipeline (Updated with Sketch Support)
"""

import random
from datetime import datetime
from typing import Dict, Optional, Literal
from PIL import Image

from .validator import ForensicPromptValidator
from .generator import FaceGenerator
from .scorer import FaceScorer
from .sketch_converter import MemoryEfficientSketchConverter, convert_to_sketch    # NEW


class SmartSketchPipeline:
    """
    Complete SmartSketch pipeline with sketch generation
    """
    
    def __init__(
        self,
        validator: ForensicPromptValidator,
        generator: FaceGenerator,
        scorer: FaceScorer,
        sketch_converter: Optional[MemoryEfficientSketchConverter] = None # <-- FIXED    
        ):
        """
        Initialize pipeline
        
        Args:
            validator: ForensicPromptValidator instance
            generator: FaceGenerator instance
            scorer: FaceScorer instance
            sketch_converter: SketchConverter instance (optional)
        """
        self.validator = validator
        self.generator = generator
        self.scorer = scorer
        self.sketch_converter = sketch_converter
        
        print("=" * 60)
        print("🚀 SmartSketch Pipeline Initialized")
        print("=" * 60)
        print("✅ LLM Validator: Ready")
        print("✅ Image Generator: Ready")
        print("✅ Scorer: Ready")
        if sketch_converter:
            print("✅ Sketch Converter: Ready")
        else:
            print("⚠️  Sketch Converter: Not loaded (photos only)")
        print("=" * 60)
    
    def generate_sketch(
        self,
        prompt: str,
        case_type: str = "criminal",
        age: Optional[int] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        output_type: Literal["photo", "sketch"] = "photo",
        sketch_style: Literal["pencil", "charcoal", "forensic"] = "forensic",
        sketch_method: Literal["simple", "controlnet"] = "controlnet"
    ) -> Dict:
        """
        Main pipeline with sketch support
        
        Args:
            prompt: User's text description
            case_type: "criminal" or "missing"
            age: Person's age
            seed: Random seed
            num_inference_steps: Generation quality
            output_type: "photo" or "sketch"
            sketch_style: Style if output_type="sketch"
            sketch_method: "simple" (fast) or "controlnet" (quality)
        
        Returns:
            Result dictionary
        """
        
        # Generate unique ID
        generation_id = f"gen_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        timestamp = datetime.now().isoformat()
        
        # ============================================
        # STEP 1: VALIDATE & ENHANCE
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
                "timestamp": timestamp
            }
        
        # ============================================
        # STEP 2: GENERATE PHOTOREALISTIC FACE
        # ============================================
        
        try:
            if seed is None:
                seed = random.randint(0, 999999)
            
            photo_image = self.generator.generate_face(
                prompt=enhanced_prompt,
                seed=seed,
                num_inference_steps=num_inference_steps
            )
            
        except Exception as e:
            return {
                "success": False,
                "generation_id": generation_id,
                "error": f"Generation failed: {str(e)}",
                "timestamp": timestamp
            }
        
        # ============================================
        # STEP 3: CONVERT TO SKETCH (if requested)
        # ============================================
        
        final_image = photo_image
        conversion_metadata = {}
        
        if output_type == "sketch":
            if self.sketch_converter is None:
                return {
                    "success": False,
                    "generation_id": generation_id,
                    "error": "Sketch converter not initialized",
                    "timestamp": timestamp
                }
            
            try:
                print(f"\n🎨 Converting to {sketch_style} sketch...")
                
                sketch_image = self.sketch_converter.convert(
                    image=photo_image,
                    method=sketch_method,
                    style=sketch_style,
                    seed=seed
                )
                
                final_image = sketch_image
                conversion_metadata = {
                    "sketch_style": sketch_style,
                    "sketch_method": sketch_method,
                    "converted_from_photo": True
                }
                
                print("✅ Sketch conversion complete!")
                
            except Exception as e:
                return {
                    "success": False,
                    "generation_id": generation_id,
                    "error": f"Sketch conversion failed: {str(e)}",
                    "timestamp": timestamp,
                    "photo_image": photo_image  # Return photo as fallback
                }
        
        # ============================================
        # STEP 4: SCORE THE OUTPUT
        # ============================================
        
        try:
            # Score the photo (not sketch, for consistency)
            scores = self.scorer.score_generation(
                image=photo_image,
                prompt=enhanced_prompt
            )
        except Exception as e:
            scores = {
                "clip_score": 0.5,
                "combined_score": 50.0,
                "interpretation": f"Scoring failed: {str(e)}"
            }
        
        # ============================================
        # STEP 5: COMPILE RESULTS
        # ============================================
        
        result = {
            "success": True,
            "generation_id": generation_id,
            "image": final_image,  # Sketch or photo
            "photo_image": photo_image,  # Always include photo
            "output_type": output_type,
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
                "model_version": "sdxl-lora-v1.0",
                **conversion_metadata
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
        device: str = "cuda",
        enable_sketch: bool = True
    ):
        """
        Initialize complete pipeline
        
        Args:
            lora_path: Path to LoRA weights
            validator_model: Validator model ID
            sdxl_model: SDXL model path
            lora_strength: LoRA strength
            device: 'cuda' or 'cpu'
            enable_sketch: Load sketch converter
        
        Returns:
            SmartSketchPipeline instance
        """
        validator = ForensicPromptValidator(validator_model, device=device)
        generator = FaceGenerator(sdxl_model, lora_path, lora_strength, device=device)
        scorer = FaceScorer(device=device)
        
        sketch_converter = None
        if enable_sketch:
            try:
                # FIXED: Use the correct class name AND pass the generator's pipeline to share memory!
                sketch_converter = MemoryEfficientSketchConverter(
                    base_pipeline=generator.pipe, 
                    device=device
                )
            except Exception as e:
                print(f"⚠️  Could not load sketch converter: {e}")
                print("   Pipeline will work for photos only")
        
        return cls(validator, generator, scorer, sketch_converter)