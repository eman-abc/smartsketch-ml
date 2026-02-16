"""
SmartSketch.AI - Sketch Converter Module
Converts photorealistic faces to forensic sketches using ControlNet
"""

import torch
import cv2
import numpy as np
from PIL import Image
from typing import Optional, Literal
from controlnet_aux import CannyDetector
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline


class SketchConverter:
    """
    Converts photorealistic faces to pencil sketches
    
    Uses ControlNet with Canny edge detection for high-quality conversion
    """
    
    def __init__(
        self,
        controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0",
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: str = "cuda"
    ):
        """
        Initialize sketch converter
        
        Args:
            controlnet_model: ControlNet model for edge detection
            base_model: Base SDXL model
            device: 'cuda' or 'cpu'
        """
        print("🎨 Loading Sketch Converter...")
        
        self.device = device
        
        # Load Canny edge detector
        print("  - Loading Canny detector...")
        self.canny_detector = CannyDetector()
        
        # Load ControlNet
        print("  - Loading ControlNet...")
        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        
        # Load SDXL with ControlNet
        print("  - Loading SDXL pipeline...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            base_model,
            controlnet=self.controlnet,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            variant="fp16" if device == "cuda" else None
        )
        
        # Move to device
        self.pipe.to(device)
        
        print("✅ Sketch Converter ready!")
    
    def photo_to_sketch_simple(
        self,
        image: Image.Image,
        style: Literal["light", "medium", "dark"] = "medium"
    ) -> Image.Image:
        """
        Simple sketch conversion using OpenCV (fallback method)
        
        Args:
            image: Input PIL Image
            style: Sketch darkness ("light", "medium", "dark")
        
        Returns:
            Sketch as PIL Image
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Invert the image
        inverted = 255 - gray
        
        # Blur based on style
        blur_sizes = {
            "light": (11, 11),
            "medium": (21, 21),
            "dark": (31, 31)
        }
        blur_size = blur_sizes.get(style, (21, 21))
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inverted, blur_size, 0)
        
        # Invert blurred image
        inverted_blur = 255 - blurred
        
        # Create pencil sketch
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        
        # Enhance contrast
        sketch = cv2.normalize(sketch, None, 0, 255, cv2.NORM_MINMAX)
        
        return Image.fromarray(sketch)
    
    def photo_to_sketch_controlnet(
        self,
        image: Image.Image,
        style: Literal["pencil", "charcoal", "forensic"] = "forensic",
        detail_level: float = 0.8,
        num_inference_steps: int = 30,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        High-quality sketch conversion using ControlNet
        
        Args:
            image: Input PIL Image (photorealistic face)
            style: Sketch style
                - "pencil": Light pencil sketch
                - "charcoal": Dark charcoal sketch  
                - "forensic": Professional forensic sketch
            detail_level: How much detail to preserve (0.0-1.0)
            num_inference_steps: Quality (20-50, higher=better)
            seed: Random seed for reproducibility
        
        Returns:
            Sketch as PIL Image
        """
        
        # Resize to standard size if needed
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.Resampling.LANCZOS)
        
        # Detect edges using Canny
        print("🔍 Detecting edges...")
        canny_image = self.canny_detector(image)
        
        # Style-specific prompts
        style_prompts = {
            "pencil": "pencil sketch, hand-drawn portrait, light shading, detailed lines, graphite drawing, sketch on paper",
            "charcoal": "charcoal sketch, dark shading, dramatic portrait, heavy lines, artistic sketch, carbon drawing",
            "forensic": "forensic sketch, police sketch, professional portrait drawing, detailed facial features, law enforcement sketch, witness description drawing"
        }
        
        prompt = style_prompts.get(style, style_prompts["forensic"])
        
        # Negative prompt to avoid unwanted styles
        negative_prompt = (
            "photo, photograph, photorealistic, realistic, color, colored, "
            "painting, oil painting, watercolor, digital art, 3d render, "
            "anime, cartoon, low quality, blurry, distorted"
        )
        
        # Set up generator for reproducibility
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None
        
        print(f"🎨 Converting to {style} sketch...")
        
        # Generate sketch
        sketch = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=canny_image,
            num_inference_steps=num_inference_steps,
            controlnet_conditioning_scale=detail_level,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        print("✅ Sketch generated!")
        
        return sketch
    
    def convert(
        self,
        image: Image.Image,
        method: Literal["simple", "controlnet"] = "controlnet",
        **kwargs
    ) -> Image.Image:
        """
        Main conversion method
        
        Args:
            image: Input PIL Image
            method: "simple" (fast) or "controlnet" (high quality)
            **kwargs: Additional arguments for chosen method
        
        Returns:
            Sketch as PIL Image
        """
        if method == "simple":
            return self.photo_to_sketch_simple(image, **kwargs)
        elif method == "controlnet":
            return self.photo_to_sketch_controlnet(image, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")


# Convenience function
def convert_to_sketch(
    image: Image.Image,
    method: str = "controlnet",
    style: str = "forensic",
    converter: Optional[SketchConverter] = None,
    **kwargs
) -> Image.Image:
    """
    Quick sketch conversion
    
    Args:
        image: PIL Image to convert
        method: "simple" or "controlnet"
        style: Sketch style
        converter: Optional pre-initialized converter
        **kwargs: Additional arguments
    
    Returns:
        Sketch as PIL Image
    """
    if converter is None:
        converter = SketchConverter()
    
    return converter.convert(image, method=method, style=style, **kwargs)