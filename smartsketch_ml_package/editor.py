# ============================================
# CELL 1: Create editor.py
# ============================================

import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
from typing import Optional, Dict
import random
from datetime import datetime


class FaceEditor:
    """
    Edit faces using SDXL img2img
    
    Features:
    - Modify facial features (glasses, hair, beard, etc.)
    - Preserve identity
    - Iterative refinement
    """
    
    def __init__(
        self,
        base_pipeline=None,
        device: str = "cuda"
    ):
        """
        Initialize face editor
        
        Args:
            base_pipeline: Existing SDXL pipeline to reuse (saves memory!)
            device: cuda or cpu
        """
        print("✏️ Loading Face Editor...")
        
        self.device = device
        
        # Reuse existing SDXL components (saves memory!)
        if base_pipeline:
            print("  - Reusing SDXL components from generator...")
            self.pipe = StableDiffusionXLImg2ImgPipeline(
                vae=base_pipeline.vae,
                text_encoder=base_pipeline.text_encoder,
                text_encoder_2=base_pipeline.text_encoder_2,
                tokenizer=base_pipeline.tokenizer,
                tokenizer_2=base_pipeline.tokenizer_2,
                unet=base_pipeline.unet,
                scheduler=base_pipeline.scheduler
            )
        else:
            print("  - Loading new SDXL img2img pipeline...")
            self.pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
        
        self.pipe.to(device)
        
        # Edit type presets (optimized strengths)
        self.edit_presets = {
            'glasses': {'strength': 0.60, 'guidance': 7.5},
            'beard': {'strength': 0.70, 'guidance': 7.5},
            'hair': {'strength': 0.65, 'guidance': 7.0},
            'hair_color': {'strength': 0.55, 'guidance': 7.0},
            'age': {'strength': 0.75, 'guidance': 8.0},
            'expression': {'strength': 0.50, 'guidance': 7.0},
            'accessories': {'strength': 0.60, 'guidance': 7.5},
            'default': {'strength': 0.70, 'guidance': 7.5}
        }
        
        print("✅ Face Editor ready!")
    
    def detect_edit_type(self, edit_prompt: str) -> str:
        """Auto-detect edit type from prompt"""
        prompt_lower = edit_prompt.lower()
        
        if any(word in prompt_lower for word in ['glasses', 'spectacles', 'eyeglasses']):
            return 'glasses'
        elif any(word in prompt_lower for word in ['beard', 'mustache', 'facial hair']):
            return 'beard'
        elif 'color' in prompt_lower and 'hair' in prompt_lower:
            return 'hair_color'
        elif any(word in prompt_lower for word in ['hair', 'hairstyle', 'haircut']):
            return 'hair'
        elif any(word in prompt_lower for word in ['older', 'younger', 'age']):
            return 'age'
        elif any(word in prompt_lower for word in ['smile', 'expression', 'frown', 'serious']):
            return 'expression'
        elif any(word in prompt_lower for word in ['hat', 'cap', 'jewelry', 'earring']):
            return 'accessories'
        else:
            return 'default'
    
    def compute_identity_preservation(
        self,
        original_image: Image.Image,
        edited_image: Image.Image
    ) -> float:
        """
        Measure how well identity was preserved using SSIM
        
        Returns:
            Score 0-1 (higher = better preservation)
        """
        try:
            import cv2
            import numpy as np
            from skimage.metrics import structural_similarity as ssim
            
            # Convert to grayscale numpy arrays
            orig_gray = np.array(original_image.convert('L'))
            edit_gray = np.array(edited_image.convert('L'))
            
            # Ensure same size
            if orig_gray.shape != edit_gray.shape:
                edit_gray = cv2.resize(edit_gray, (orig_gray.shape[1], orig_gray.shape[0]))
            
            # Compute SSIM
            score = ssim(orig_gray, edit_gray)
            
            return float(score)
            
        except Exception as e:
            print(f"⚠️  Identity scoring failed: {e}")
            return 0.0
    
    def edit_face(
        self,
        original_image: Image.Image,
        edit_prompt: str,
        strength: Optional[float] = None,
        guidance_scale: Optional[float] = None,
        num_inference_steps: int = 30,
        seed: Optional[int] = None
    ) -> Dict:
        """
        Edit a face
        
        Args:
            original_image: PIL Image of original face
            edit_prompt: What to change (e.g., "add round glasses")
            strength: How much to change (0.0-1.0, auto if None)
            guidance_scale: Prompt adherence (auto if None)
            num_inference_steps: Quality (20-50, default 30)
            seed: Random seed for reproducibility
        
        Returns:
            Dictionary with:
            - success: bool
            - edit_id: str
            - original_image: PIL Image
            - edited_image: PIL Image
            - identity_score: float (0-1)
            - metadata: dict
        """
        
        # Generate edit ID
        edit_id = f"edit_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        timestamp = datetime.now().isoformat()
        
        print(f"\\n{'='*60}")
        print(f"✏️  EDITING FACE")
        print(f"{'='*60}")
        print(f"🆔 Edit ID: {edit_id}")
        print(f"📝 Edit: {edit_prompt}")
        
        # Auto-detect edit type
        edit_type = self.detect_edit_type(edit_prompt)
        preset = self.edit_presets.get(edit_type, self.edit_presets['default'])
        
        # Use preset or provided values
        if strength is None:
            strength = preset['strength']
        if guidance_scale is None:
            guidance_scale = preset['guidance']
        
        print(f"🎯 Edit Type: {edit_type}")
        print(f"⚙️  Strength: {strength:.2f} (0=no change, 1=full change)")
        print(f"⚙️  Guidance: {guidance_scale}")
        
        # Resize if needed
        if original_image.size != (512, 512):
            print("📐 Resizing to 512x512...")
            original_image = original_image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Build full prompt
        full_prompt = f"professional forensic photograph, {edit_prompt}, realistic, photorealistic, natural skin tones, detailed facial features, high quality"
        
        negative_prompt = (
            "low quality, blurry, distorted, deformed, disfigured, "
            "bad anatomy, extra limbs, poorly drawn face, mutation, "
            "anime, cartoon, 3d render, duplicate, multiple people"
        )
        
        # Set up generator for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"🎨 Applying edit...")
        
        try:
            # Generate edited image
            edited_image = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=original_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
            
            print("✅ Edit complete!")
            
            # Compute identity preservation
            print("👤 Computing identity preservation...")
            identity_score = self.compute_identity_preservation(
                original_image,
                edited_image
            )
            print(f"   Identity preserved: {identity_score:.1%}")
            
            # Determine if identity well preserved
            identity_preserved = identity_score >= 0.75
            
            return {
                'success': True,
                'edit_id': edit_id,
                'original_image': original_image,
                'edited_image': edited_image,
                'identity_score': identity_score,
                'identity_preserved': identity_preserved,
                'edit_prompt': edit_prompt,
                'metadata': {
                    'timestamp': timestamp,
                    'edit_id': edit_id,
                    'edit_prompt': edit_prompt,
                    'edit_type': edit_type,
                    'strength': strength,
                    'guidance_scale': guidance_scale,
                    'num_inference_steps': num_inference_steps,
                    'seed': seed,
                    'identity_score': identity_score,
                    'identity_preserved': identity_preserved
                }
            }
            
        except Exception as e:
            print(f"❌ Edit failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'edit_id': edit_id,
                'timestamp': timestamp
            }
    
    def batch_edit(
        self,
        original_image: Image.Image,
        edit_prompts: list,
        **kwargs
    ) -> list:
        """
        Apply multiple edits to the same image
        
        Args:
            original_image: PIL Image
            edit_prompts: List of edit descriptions
            **kwargs: Additional parameters for edit_face
        
        Returns:
            List of edit result dictionaries
        """
        print(f"\\n{'='*60}")
        print(f"🔄 BATCH EDITING: {len(edit_prompts)} edits")
        print(f"{'='*60}")
        
        results = []
        
        for i, prompt in enumerate(edit_prompts, 1):
            print(f"\\n[{i}/{len(edit_prompts)}] {prompt}")
            
            result = self.edit_face(
                original_image=original_image,
                edit_prompt=prompt,
                **kwargs
            )
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        print(f"\\n{'='*60}")
        print(f"✅ Batch complete: {successful}/{len(edit_prompts)} successful")
        print(f"{'='*60}")
        
        return results


# Convenience function
def edit_face(
    original_image: Image.Image,
    edit_prompt: str,
    editor: Optional[FaceEditor] = None,
    **kwargs
) -> Dict:
    """
    Quick edit function
    
    Args:
        original_image: PIL Image
        edit_prompt: What to change
        editor: Optional pre-initialized editor
        **kwargs: Additional parameters
    
    Returns:
        Edit result dictionary
    """
    if editor is None:
        editor = FaceEditor()
    
    return editor.edit_face(original_image, edit_prompt, **kwargs)

