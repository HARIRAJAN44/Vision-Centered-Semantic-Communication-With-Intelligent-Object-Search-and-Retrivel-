import faiss
import numpy as np
import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
from PIL import Image
import threading


@dataclass
class Record:
    """Searchable record in the FAISS index."""
    record_id: str
    embedding: np.ndarray
    image_path: Optional[str] = None
    object_class: Optional[str] = None
    confidence: Optional[float] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    embedding_type: str = 'image'  # 'image', 'crop', 'reconstruction'
    source: str = 'vcsc'  # 'vcsc', 'jpeg', 'huffman', 'itisc', 'upload'
    channel_snr_db: Optional[float] = None
    channel_type: Optional[str] = None
    quantization_mode: Optional[str] = None
    reconstruction_path: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization (embedding handled separately)."""
        d = asdict(self)
        # Don't serialize embedding to JSON
        d.pop('embedding', None)
        # Convert numpy types to Python types
        if d['confidence'] is not None:
            d['confidence'] = float(d['confidence'])
        if d['bbox'] is not None:
            d['bbox'] = tuple(float(x) for x in d['bbox'])
        if d['channel_snr_db'] is not None:
            d['channel_snr_db'] = float(d['channel_snr_db'])
        return d
    
    @classmethod
    def from_dict(cls, data: Dict, embedding: np.ndarray) -> 'Record':
        """Create Record from dictionary and embedding array."""
        return cls(
            record_id=data['record_id'],
            embedding=embedding,
            image_path=data.get('image_path'),
            object_class=data.get('object_class'),
            confidence=data.get('confidence'),
            bbox=tuple(data['bbox']) if data.get('bbox') else None,
            timestamp=data.get('timestamp', datetime.now().isoformat()),
            embedding_type=data.get('embedding_type', 'image'),
            source=data.get('source', 'vcsc'),
            channel_snr_db=data.get('channel_snr_db'),
            channel_type=data.get('channel_type'),
            quantization_mode=data.get('quantization_mode'),
            reconstruction_path=data.get('reconstruction_path'),
            tags=data.get('tags', [])
        )


class FAISSStore:
    """
    Persistent FAISS vector store with metadata.
    
    Features:
    - Exact search with IndexFlatIP (cosine similarity via normalized vectors)
    - Persistent index and metadata storage
    - Duplicate prevention
    - Thread-safe operations
    - Index rebuild and validation
    """
    
    def __init__(
        self,
        index_path: str = "faiss_index",
        embedding_dim: int = 512,
        use_gpu: bool = False
    ):
        self.index_path = Path(index_path)
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu and faiss.get_num_gpus() > 0
        
        self.index = None
        self.metadata: Dict[int, Record] = {}  # faiss_id -> Record
        self.id_to_faiss_id: Dict[str, int] = {}  # record_id -> faiss_id
        self.faiss_id_to_id: Dict[int, str] = {}  # faiss_id -> record_id
        self.next_faiss_id = 0
        self._lock = threading.RLock()
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Create or load FAISS index."""
        index_file = self.index_path / "index.faiss"
        meta_file = self.index_path / "metadata.json"
        mapping_file = self.index_path / "mapping.pkl"
        
        self.index_path.mkdir(parents=True, exist_ok=True)
        
        if index_file.exists() and meta_file.exists() and mapping_file.exists():
            self._load_index(index_file, meta_file, mapping_file)
        else:
            # Create new exact search index (Inner Product for cosine similarity)
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            if self.use_gpu:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            print(f"[FAISSStore] Created new index at {self.index_path}")
    
    def _load_index(self, index_file: Path, meta_file: Path, mapping_file: Path):
        """Load existing index and metadata."""
        try:
            self.index = faiss.read_index(str(index_file))
            if self.use_gpu and not hasattr(self.index, 'device'):
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            
            with open(meta_file, 'r') as f:
                meta_data = json.load(f)
            
            with open(mapping_file, 'rb') as f:
                mapping_data = pickle.load(f)
            
            self.metadata = {int(k): Record.from_dict(v, mapping_data['embeddings'][int(k)]) 
                           for k, v in meta_data.items()}
            self.id_to_faiss_id = mapping_data['id_to_faiss_id']
            self.faiss_id_to_id = {int(k): v for k, v in mapping_data['faiss_id_to_id'].items()}
            self.next_faiss_id = mapping_data.get('next_faiss_id', max(self.faiss_id_to_id.keys()) + 1 if self.faiss_id_to_id else 0)
            
            print(f"[FAISSStore] Loaded index with {self.index.ntotal} vectors")
            
        except Exception as e:
            print(f"[FAISSStore] ERROR loading index: {e}. Creating new index.")
            self._initialize_index()
    
    def save(self):
        """Save index and metadata to disk."""
        with self._lock:
            index_file = self.index_path / "index.faiss"
            meta_file = self.index_path / "metadata.json"
            mapping_file = self.index_path / "mapping.pkl"
            
            # Save FAISS index
            if self.use_gpu and hasattr(self.index, 'device'):
                cpu_index = faiss.index_gpu_to_cpu(self.index)
                faiss.write_index(cpu_index, str(index_file))
            else:
                faiss.write_index(self.index, str(index_file))
            
            # Save metadata
            meta_data = {str(k): v.to_dict() for k, v in self.metadata.items()}
            with open(meta_file, 'w') as f:
                json.dump(meta_data, f, indent=2)
            
            # Save embeddings and mappings
            embeddings = {k: v.embedding for k, v in self.metadata.items()}
            mapping_data = {
                'embeddings': embeddings,
                'id_to_faiss_id': self.id_to_faiss_id,
                'faiss_id_to_id': {str(k): v for k, v in self.faiss_id_to_id.items()},
                'next_faiss_id': self.next_faiss_id
            }
            with open(mapping_file, 'wb') as f:
                pickle.dump(mapping_data, f)
            
            print(f"[FAISSStore] Saved index with {self.index.ntotal} vectors")
    
    def add(self, record: Record) -> bool:
        """
        Add a record to the index.
        
        Returns:
            True if added, False if duplicate record_id exists
        """
        with self._lock:
            if record.record_id in self.id_to_faiss_id:
                print(f"[FAISSStore] Duplicate record_id: {record.record_id}")
                return False
            
            # Normalize embedding for cosine similarity
            embedding = record.embedding.astype(np.float32)
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            norm[norm == 0] = 1
            embedding = embedding / norm
            
            # Add to FAISS index
            faiss_id = self.next_faiss_id
            self.index.add(embedding)
            
            # Update mappings
            self.metadata[faiss_id] = record
            self.id_to_faiss_id[record.record_id] = faiss_id
            self.faiss_id_to_id[faiss_id] = record.record_id
            self.next_faiss_id += 1
            
            return True
    
    def add_batch(self, records: List[Record]) -> int:
        """Add multiple records. Returns number of records added."""
        added = 0
        for record in records:
            if self.add(record):
                added += 1
        return added
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[Record, float]]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            filter_fn: Optional function(record) -> bool to filter results
            
        Returns:
            List of (Record, similarity_score) tuples
        """
        with self._lock:
            if self.index.ntotal == 0:
                return []
            
            # Normalize query
            query = query_embedding.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(query, axis=1, keepdims=True)
            if norm > 0:
                query = query / norm
            
            # Search
            k = min(top_k * 3, self.index.ntotal)  # Search more to account for filtering
            scores, indices = self.index.search(query, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                
                if idx in self.metadata:
                    record = self.metadata[idx]
                    
                    if filter_fn is None or filter_fn(record):
                        results.append((record, float(score)))
                        
                        if len(results) >= top_k:
                            break
            
            return results
    
    def search_by_text(
        self,
        text: str,
        feature_extractor,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[Record, float]]:
        """Search using a text query."""
        text_result = feature_extractor.extract_text_embedding(text)
        return self.search(text_result.embedding, top_k, filter_fn)
    
    def search_by_image(
        self,
        image: Union[Image.Image, np.ndarray, str],
        feature_extractor,
        top_k: int = 10,
        filter_fn: Optional[callable] = None
    ) -> List[Tuple[Record, float]]:
        """Search using an image query."""
        img_result = feature_extractor.extract_image_embedding(image)
        return self.search(img_result.embedding, top_k, filter_fn)
    
    def search_by_class(
        self,
        object_class: str,
        top_k: int = 10
    ) -> List[Record]:
        """Search records by object class."""
        with self._lock:
            results = []
            for record in self.metadata.values():
                if record.object_class == object_class:
                    results.append(record)
                    if len(results) >= top_k:
                        break
            return results
    
    def get(self, record_id: str) -> Optional[Record]:
        """Get a record by ID."""
        with self._lock:
            faiss_id = self.id_to_faiss_id.get(record_id)
            if faiss_id is not None:
                return self.metadata.get(faiss_id)
            return None
    
    def delete(self, record_id: str) -> bool:
        """
        Delete a record from the index.
        
        Note: FAISS doesn't support efficient deletion. This marks the record
        as deleted and rebuilds the index if needed.
        """
        with self._lock:
            faiss_id = self.id_to_faiss_id.pop(record_id, None)
            if faiss_id is None:
                return False
            
            self.metadata.pop(faiss_id, None)
            self.faiss_id_to_id.pop(faiss_id, None)
            
            # Rebuild index (simple but not efficient for large indices)
            self._rebuild_index()
            return True
    
    def _rebuild_index(self):
        """Rebuild FAISS index from current metadata."""
        if self.use_gpu and hasattr(self.index, 'device'):
            cpu_index = faiss.index_gpu_to_cpu(self.index)
        else:
            cpu_index = self.index
        
        # Create new index
        new_index = faiss.IndexFlatIP(self.embedding_dim)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            new_index = faiss.index_cpu_to_gpu(res, 0, new_index)
        
        # Re-add all embeddings
        new_metadata = {}
        new_id_to_faiss_id = {}
        new_faiss_id_to_id = {}
        next_faiss_id = 0
        
        for record in self.metadata.values():
            embedding = record.embedding.astype(np.float32).reshape(1, -1)
            norm = np.linalg.norm(embedding, axis=1, keepdims=True)
            norm[norm == 0] = 1
            embedding = embedding / norm
            
            new_index.add(embedding)
            new_metadata[next_faiss_id] = record
            new_id_to_faiss_id[record.record_id] = next_faiss_id
            new_faiss_id_to_id[next_faiss_id] = record.record_id
            next_faiss_id += 1
        
        self.index = new_index
        self.metadata = new_metadata
        self.id_to_faiss_id = new_id_to_faiss_id
        self.faiss_id_to_id = new_faiss_id_to_id
        self.next_faiss_id = next_faiss_id
        
        print(f"[FAISSStore] Rebuilt index with {self.index.ntotal} vectors")
    
    def validate(self) -> bool:
        """Validate index consistency."""
        with self._lock:
            if self.index.ntotal != len(self.metadata):
                print(f"[FAISSStore] WARNING: Index size ({self.index.ntotal}) != metadata size ({len(self.metadata)})")
                return False
            
            # Check mappings
            for record_id, faiss_id in self.id_to_faiss_id.items():
                if faiss_id not in self.metadata:
                    print(f"[FAISSStore] WARNING: Missing metadata for faiss_id {faiss_id}")
                    return False
                if self.metadata[faiss_id].record_id != record_id:
                    print(f"[FAISSStore] WARNING: ID mismatch for faiss_id {faiss_id}")
                    return False
            
            return True
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        with self._lock:
            class_counts = {}
            source_counts = {}
            type_counts = {}
            
            for record in self.metadata.values():
                if record.object_class:
                    class_counts[record.object_class] = class_counts.get(record.object_class, 0) + 1
                source_counts[record.source] = source_counts.get(record.source, 0) + 1
                type_counts[record.embedding_type] = type_counts.get(record.embedding_type, 0) + 1
            
            return {
                'total_vectors': self.index.ntotal,
                'embedding_dim': self.embedding_dim,
                'class_distribution': class_counts,
                'source_distribution': source_counts,
                'type_distribution': type_counts,
                'index_path': str(self.index_path)
            }
    
    def __len__(self) -> int:
        return self.index.ntotal if self.index else 0


def create_faiss_store(index_path: str = "faiss_index", **kwargs) -> FAISSStore:
    """Factory function to create a FAISS store."""
    return FAISSStore(index_path=index_path, **kwargs)


if __name__ == "__main__":
    print("Testing FAISS Store...")
    
    # Create store
    store = create_faiss_store("test_faiss_index", embedding_dim=512)
    
    # Create test records
    for i in range(5):
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        
        record = Record(
            record_id=f"test_{i}",
            embedding=embedding,
            object_class='person' if i % 2 == 0 else 'car',
            confidence=0.9 - i * 0.1,
            source='vcsc',
            tags=['test']
        )
        store.add(record)
    
    print(f"Added records. Total: {len(store)}")
    
    # Search
    query = np.random.randn(512).astype(np.float32)
    query = query / np.linalg.norm(query)
    
    results = store.search(query, top_k=3)
    print(f"Search results: {len(results)}")
    for record, score in results:
        print(f"  {record.record_id}: {score:.4f} (class={record.object_class})")
    
    # Test class search
    person_results = store.search_by_class('person', top_k=10)
    print(f"Person records: {len(person_results)}")
    
    # Save and reload
    store.save()
    
    store2 = create_faiss_store("test_faiss_index", embedding_dim=512)
    print(f"Reloaded. Total: {len(store2)}")
    
    results2 = store2.search(query, top_k=3)
    print(f"Search after reload: {len(results2)}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_faiss_index")
    
    print("FAISS Store test complete!")