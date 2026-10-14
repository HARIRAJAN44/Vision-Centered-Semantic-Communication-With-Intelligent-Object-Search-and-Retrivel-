from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from PIL import Image
import numpy as np
from datetime import datetime

from .feature_extractor import FeatureExtractor, EmbeddingResult
from .faiss_store import FAISSStore, Record, create_faiss_store


@dataclass
class SearchResult:
    """Search result with metadata."""
    record: Record
    similarity: float
    
    def to_dict(self) -> Dict:
        return {
            'record': self.record.to_dict(),
            'similarity': self.similarity
        }


@dataclass
class SearchQuery:
    """Structured search query."""
    text: Optional[str] = None
    image: Optional[Union[Image.Image, np.ndarray, str]] = None
    object_class: Optional[str] = None
    top_k: int = 10
    min_confidence: Optional[float] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    embedding_type: Optional[str] = None
    source: Optional[str] = None
    tags: Optional[List[str]] = None
    
    def filter_fn(self) -> callable:
        """Generate filter function from query parameters."""
        def filter_func(record: Record) -> bool:
            if self.object_class and record.object_class != self.object_class:
                return False
            if self.min_confidence and record.confidence is not None:
                if record.confidence < self.min_confidence:
                    return False
            if self.embedding_type and record.embedding_type != self.embedding_type:
                return False
            if self.source and record.source != self.source:
                return False
            if self.tags:
                if not all(tag in record.tags for tag in self.tags):
                    return False
            if self.start_time:
                if record.timestamp < self.start_time:
                    return False
            if self.end_time:
                if record.timestamp > self.end_time:
                    return False
            return True
        return filter_func


class SearchEngine:
    """
    High-level search interface combining FAISS store and feature extractor.
    """
    
    def __init__(
        self,
        store: FAISSStore,
        feature_extractor: FeatureExtractor
    ):
        self.store = store
        self.feature_extractor = feature_extractor
    
    def search(self, query: SearchQuery) -> List[SearchResult]:
        """Execute a structured search query."""
        filter_fn = query.filter_fn()
        
        if query.text:
            results = self.store.search_by_text(
                query.text,
                self.feature_extractor,
                top_k=query.top_k,
                filter_fn=filter_fn
            )
        elif query.image is not None:
            results = self.store.search_by_image(
                query.image,
                self.feature_extractor,
                top_k=query.top_k,
                filter_fn=filter_fn
            )
        elif query.object_class:
            records = self.store.search_by_class(query.object_class, top_k=query.top_k)
            results = [(r, 1.0) for r in records]
            # Apply additional filters
            filtered = []
            for record, score in results:
                if filter_fn(record):
                    filtered.append((record, score))
            results = filtered
        else:
            # No valid query
            return []
        
        return [SearchResult(record=r, similarity=s) for r, s in results]
    
    def search_text(self, text: str, top_k: int = 10, **filters) -> List[SearchResult]:
        """Search by natural language text."""
        query = SearchQuery(text=text, top_k=top_k, **filters)
        return self.search(query)
    
    def search_image(self, image: Union[Image.Image, np.ndarray, str], top_k: int = 10, **filters) -> List[SearchResult]:
        """Search by image similarity."""
        query = SearchQuery(image=image, top_k=top_k, **filters)
        return self.search(query)
    
    def search_class(self, object_class: str, top_k: int = 10, **filters) -> List[SearchResult]:
        """Search by object class."""
        query = SearchQuery(object_class=object_class, top_k=top_k, **filters)
        return self.search(query)
    
    def add_record(
        self,
        record_id: str,
        embedding: np.ndarray,
        image_path: Optional[str] = None,
        object_class: Optional[str] = None,
        confidence: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        embedding_type: str = 'image',
        source: str = 'vcsc',
        channel_snr_db: Optional[float] = None,
        channel_type: Optional[str] = None,
        quantization_mode: Optional[str] = None,
        reconstruction_path: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Add a record to the index."""
        record = Record(
            record_id=record_id,
            embedding=embedding,
            image_path=image_path,
            object_class=object_class,
            confidence=confidence,
            bbox=bbox,
            embedding_type=embedding_type,
            source=source,
            channel_snr_db=channel_snr_db,
            channel_type=channel_type,
            quantization_mode=quantization_mode,
            reconstruction_path=reconstruction_path,
            tags=tags or []
        )
        return self.store.add(record)
    
    def add_image_record(
        self,
        image: Union[Image.Image, np.ndarray, str],
        record_id: Optional[str] = None,
        object_class: Optional[str] = None,
        confidence: Optional[float] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        image_path: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Add an image record by extracting its embedding."""
        if record_id is None:
            record_id = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        result = self.feature_extractor.extract_image_embedding(image)
        
        return self.add_record(
            record_id=record_id,
            embedding=result.embedding,
            image_path=image_path,
            object_class=object_class,
            confidence=confidence,
            bbox=bbox,
            **kwargs
        )
    
    def add_crop_record(
        self,
        image: Image.Image,
        bbox: Tuple[float, float, float, float],
        record_id: Optional[str] = None,
        object_class: Optional[str] = None,
        confidence: Optional[float] = None,
        **kwargs
    ) -> bool:
        """Add a record from an image crop."""
        if record_id is None:
            record_id = f"crop_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        result = self.feature_extractor.extract_crop_embedding(image, bbox)
        
        return self.add_record(
            record_id=record_id,
            embedding=result.embedding,
            object_class=object_class,
            confidence=confidence,
            bbox=bbox,
            embedding_type='crop',
            **kwargs
        )
    
    def add_reconstruction_record(
        self,
        image: Union[Image.Image, np.ndarray, str],
        record_id: Optional[str] = None,
        original_record_id: Optional[str] = None,
        snr_db: Optional[float] = None,
        channel_type: Optional[str] = None,
        quantization_mode: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Add a reconstruction record linked to original."""
        if record_id is None:
            record_id = f"recon_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        result = self.feature_extractor.extract_image_embedding(image)
        
        tags = kwargs.get('tags', [])
        if original_record_id:
            tags.append(f"original:{original_record_id}")
        
        return self.add_record(
            record_id=record_id,
            embedding=result.embedding,
            object_class=kwargs.get('object_class'),
            confidence=kwargs.get('confidence'),
            embedding_type='reconstruction',
            source=kwargs.get('source', 'vcsc'),
            channel_snr_db=snr_db,
            channel_type=channel_type,
            quantization_mode=quantization_mode,
            reconstruction_path=kwargs.get('reconstruction_path'),
            tags=tags
        )
    
    def get_stats(self) -> Dict:
        """Get search engine statistics."""
        return self.store.get_stats()
    
    def save(self):
        """Save the index."""
        self.store.save()


def create_search_engine(
    index_path: str = "faiss_index",
    model_name: str = "openai/clip-vit-base-patch32",
    embedding_dim: int = 512,
    **kwargs
) -> SearchEngine:
    """Factory function to create a search engine."""
    store = create_faiss_store(index_path=index_path, embedding_dim=embedding_dim)
    extractor = FeatureExtractor(model_name=model_name)
    extractor.load()
    return SearchEngine(store=store, feature_extractor=extractor)


if __name__ == "__main__":
    print("Testing Search Engine...")
    
    # Create search engine with test index
    store = create_faiss_store("test_search_index", embedding_dim=512)
    extractor = FeatureExtractor()
    extractor.load()
    
    engine = SearchEngine(store=store, feature_extractor=extractor)
    
    # Add test images
    for i in range(5):
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        engine.add_image_record(
            image=img,
            record_id=f"test_img_{i}",
            object_class='person' if i % 2 == 0 else 'car',
            confidence=0.9 - i * 0.1,
            source='vcsc'
        )
    
    print(f"Added images. Stats: {engine.get_stats()}")
    
    # Search by text
    results = engine.search_text("a person", top_k=3)
    print(f"Text search 'a person': {len(results)} results")
    for r in results:
        print(f"  {r.record.record_id}: sim={r.similarity:.4f}, class={r.record.object_class}")
    
    # Search by image
    query_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
    results = engine.search_image(query_img, top_k=3)
    print(f"Image search: {len(results)} results")
    for r in results:
        print(f"  {r.record.record_id}: sim={r.similarity:.4f}")
    
    # Search by class
    results = engine.search_class('car', top_k=10)
    print(f"Class search 'car': {len(results)} results")
    
    # Save
    engine.save()
    
    # Reload
    engine2 = create_search_engine("test_search_index")
    results = engine2.search_text("a person", top_k=3)
    print(f"After reload: {len(results)} results")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_search_index")
    
    print("Search Engine test complete!")