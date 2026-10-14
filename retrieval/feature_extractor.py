import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import os
import json
from pathlib import Path


@dataclass
class EmbeddingResult:
    """Container for embedding extraction results."""
    embedding: np.ndarray
    embedding_type: str  # 'image', 'text', 'crop'
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            'embedding': self.embedding.tolist(),
            'embedding_type': self.embedding_type,
            'metadata': self.metadata
        }


class FeatureExtractor:
    """
    Feature extractor using CLIP for image and text embeddings.
    
    Uses the same CLIP model as the evaluator for consistency.
    Supports offline inference after initial model download.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.normalize = normalize
        
        self.model = None
        self.processor = None
        self._loaded = False
        self.embedding_dim = 512  # CLIP ViT-B/32 embedding dimension
    
    def load(self) -> bool:
        """Load the CLIP model and processor."""
        if self._loaded:
            return True
        
        try:
            from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection, CLIPTokenizer
            
            print(f"[FeatureExtractor] Loading CLIP model {self.model_name} on {self.device}...")
            
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.vision_model = CLIPVisionModelWithProjection.from_pretrained(self.model_name).to(self.device)
            self.text_model = CLIPTextModelWithProjection.from_pretrained(self.model_name).to(self.device)
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
            
            self.vision_model.eval()
            self.text_model.eval()
            
            # Get embedding dimension from model config
            self.embedding_dim = self.vision_model.config.projection_dim
            
            self._loaded = True
            print(f"[FeatureExtractor] Model loaded successfully (embedding_dim={self.embedding_dim})")
            return True
            
        except ImportError:
            print(f"[FeatureExtractor] ERROR: transformers not installed. Run: pip install transformers")
            return False
        except Exception as e:
            print(f"[FeatureExtractor] ERROR: Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        return self._loaded and self.vision_model is not None
    
    def _preprocess_image(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """Convert input to PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert('RGB')
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    @torch.no_grad()
    def extract_image_embedding(self, image: Union[Image.Image, np.ndarray, str]) -> EmbeddingResult:
        """
        Extract CLIP image embedding.
        
        Args:
            image: PIL Image, numpy array (HWC RGB), or image path
            
        Returns:
            EmbeddingResult with normalized embedding
        """
        if not self.is_loaded():
            if not self.load():
                raise RuntimeError("Failed to load feature extractor")
        
        pil_image = self._preprocess_image(image)
        
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        outputs = self.vision_model(**inputs)
        embedding = outputs.image_embeds.cpu().numpy().flatten()
        
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return EmbeddingResult(
            embedding=embedding.astype(np.float32),
            embedding_type='image',
            metadata={'input_size': pil_image.size}
        )
    
    @torch.no_grad()
    def extract_text_embedding(self, text: str) -> EmbeddingResult:
        """
        Extract CLIP text embedding.
        
        Args:
            text: Text query
            
        Returns:
            EmbeddingResult with normalized embedding
        """
        if not self.is_loaded():
            if not self.load():
                raise RuntimeError("Failed to load feature extractor")
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.text_model(**inputs)
        embedding = outputs.text_embeds.cpu().numpy().flatten()
        
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return EmbeddingResult(
            embedding=embedding.astype(np.float32),
            embedding_type='text',
            metadata={'text': text}
        )
    
    @torch.no_grad()
    def extract_batch_image_embeddings(self, images: List[Union[Image.Image, np.ndarray, str]]) -> List[EmbeddingResult]:
        """Extract embeddings for a batch of images."""
        if not self.is_loaded():
            if not self.load():
                raise RuntimeError("Failed to load feature extractor")
        
        pil_images = [self._preprocess_image(img) for img in images]
        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        outputs = self.vision_model(**inputs)
        embeddings = outputs.image_embeds.cpu().numpy()
        
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            embeddings = embeddings / norms
        
        results = []
        for i, emb in enumerate(embeddings):
            results.append(EmbeddingResult(
                embedding=emb.astype(np.float32),
                embedding_type='image',
                metadata={'input_size': pil_images[i].size}
            ))
        
        return results
    
    @torch.no_grad()
    def extract_crop_embedding(self, image: Image.Image, bbox: Tuple[float, float, float, float]) -> EmbeddingResult:
        """Extract embedding for a cropped region of an image."""
        x1, y1, x2, y2 = map(int, bbox)
        crop = image.crop((x1, y1, x2, y2))
        return self.extract_image_embedding(crop)


def create_feature_extractor(model_name: str = "openai/clip-vit-base-patch32", **kwargs) -> FeatureExtractor:
    """Factory function to create a feature extractor."""
    return FeatureExtractor(model_name=model_name, **kwargs)


if __name__ == "__main__":
    print("Testing Feature Extractor...")
    
    extractor = create_feature_extractor()
    
    if not extractor.load():
        print("Failed to load model. Skipping test.")
    else:
        test_img = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
        
        img_result = extractor.extract_image_embedding(test_img)
        print(f"Image embedding shape: {img_result.embedding.shape}")
        print(f"Image embedding norm: {np.linalg.norm(img_result.embedding):.6f}")
        
        text_result = extractor.extract_text_embedding("a person walking on the street")
        print(f"Text embedding shape: {text_result.embedding.shape}")
        print(f"Text embedding norm: {np.linalg.norm(text_result.embedding):.6f}")
        
        # Test similarity
        sim = np.dot(img_result.embedding, text_result.embedding)
        print(f"Image-Text similarity: {sim:.4f}")
    
    print("Feature Extractor test complete!")