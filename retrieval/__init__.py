"""
Retrieval Module for VCSC

Provides FAISS-based vector search with CLIP embeddings for intelligent object/image search.
"""

from .feature_extractor import FeatureExtractor, EmbeddingResult, create_feature_extractor
from .faiss_store import FAISSStore, Record, create_faiss_store
from .search import SearchEngine, SearchQuery, SearchResult, create_search_engine

__all__ = [
    'FeatureExtractor',
    'EmbeddingResult',
    'create_feature_extractor',
    'FAISSStore',
    'Record',
    'create_faiss_store',
    'SearchEngine',
    'SearchQuery',
    'SearchResult',
    'create_search_engine'
]