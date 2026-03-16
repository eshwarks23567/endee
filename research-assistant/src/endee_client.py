"""
Endee Vector Database Client Wrapper

This module provides a clean interface to interact with the Endee vector database
using the official Endee Python SDK (pip install endee).
It handles connection management, index operations, and vector search.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid

from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Monkeypatch for Endee SDK bug where it calls v_item.get() on a Pydantic model
try:
    from endee.schema import VectorItem
    if not hasattr(VectorItem, "get"):
        VectorItem.get = lambda self, key, default=None: getattr(self, key, default)
except ImportError:
    pass


@dataclass
class SearchResult:
    """Represents a single search result from Endee."""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None


class EndeeClient:
    """
    A wrapper client for interacting with the Endee vector database.
    
    Uses the official Endee Python SDK under the hood.
    Provides methods for:
    - Creating and managing indexes (collections)
    - Inserting vectors with metadata via upsert
    - Performing similarity search
    - Filtering results based on metadata
    
    Example:
        >>> client = EndeeClient(host="localhost", port=8080)
        >>> client.connect()
        >>> client.create_collection("papers", dimension=384)
        >>> client.insert("papers", vectors, metadata)
        >>> results = client.search("papers", query_vector, top_k=10)
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 8080,
        api_key: Optional[str] = None
    ):
        """
        Initialize the Endee client.
        
        Args:
            host: Endee server hostname
            port: Endee server port (default 8080)
            api_key: Optional auth token for authentication
        """
        self.host = host
        self.port = port
        self.api_key = api_key or ""
        self.base_url = f"http://{host}:{port}/api/v1"
        self._client = None
        self._connected = False
        self._indexes: Dict[str, Any] = {}
        
        logger.info(f"Initialized Endee client for {self.base_url}")
    
    def connect(self) -> bool:
        """
        Establish connection to the Endee server.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            from endee import Endee
            
            self._client = Endee(self.api_key)
            self._client.set_base_url(self.base_url)
            self._connected = True
            logger.info(f"Connected to Endee server at {self.base_url}")
            return True
        except Exception as e:
            logger.error(f"Could not connect to Endee server: {e}")
            self._connected = False
            return False
    
    def _ensure_connected(self):
        """Ensure the client is connected; connect if not."""
        if not self._connected or self._client is None:
            self.connect()
    
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        index_type: str = "hnsw",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create a new vector index (collection) in Endee.
        
        Args:
            name: Index name
            dimension: Vector dimension (e.g., 384 for MiniLM, 768 for BERT)
            metric: Distance metric ('cosine', 'euclidean', 'dot')
            index_type: Index type (used for logging, Endee uses HNSW by default)
            **kwargs: Additional index parameters
        
        Returns:
            Index creation response dict
        
        Example:
            >>> client.create_collection(
            ...     name="research_papers",
            ...     dimension=384,
            ...     metric="cosine"
            ... )
        """
        from endee import Precision
        
        self._ensure_connected()
        
        # Map metric name to Endee space_type
        space_type = metric if metric in ("cosine", "euclidean", "dot") else "cosine"
        
        # Map precision from kwargs, default to Float32
        precision = kwargs.get("precision", Precision.FLOAT32)
        
        logger.info(f"Creating index '{name}' with dimension {dimension}, space_type={space_type}")
        
        try:
            self._client.create_index(
                name=name,
                dimension=dimension,
                space_type=space_type,
                precision=precision
            )
            
            # Cache the index reference
            self._indexes[name] = self._client.get_index(name=name)
            
            return {
                "success": True,
                "collection": name,
                "dimension": dimension,
                "metric": space_type,
                "index_type": index_type
            }
        except Exception as e:
            logger.error(f"Failed to create index '{name}': {e}")
            raise
    
    def delete_collection(self, name: str) -> bool:
        """
        Delete an index (collection) from Endee.
        
        Args:
            name: Index name to delete
            
        Returns:
            True if deletion successful
        """
        self._ensure_connected()
        
        logger.info(f"Deleting index '{name}'")
        try:
            self._client.delete_index(name=name)
            self._indexes.pop(name, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete index '{name}': {e}")
            return False
    
    def collection_exists(self, name: str) -> bool:
        """
        Check if an index (collection) exists.
        
        Args:
            name: Index name
            
        Returns:
            True if index exists
        """
        self._ensure_connected()
        
        try:
            self._client.get_index(name=name)
            return True
        except Exception:
            return False
    
    def _get_index(self, name: str):
        """Get a cached or fresh index reference."""
        if name not in self._indexes:
            self._indexes[name] = self._client.get_index(name=name)
        return self._indexes[name]
    
    def insert(
        self,
        collection: str,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Insert vectors with metadata into an index (collection) via upsert.
        
        Args:
            collection: Target index name
            vectors: List of embedding vectors
            metadata: List of metadata dicts (one per vector)
            ids: Optional list of IDs (auto-generated if not provided)
        
        Returns:
            Insert operation response
        
        Example:
            >>> vectors = [[0.1, 0.2, ...], [0.3, 0.4, ...]]
            >>> metadata = [{"title": "Paper 1"}, {"title": "Paper 2"}]
            >>> client.insert("papers", vectors, metadata)
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        self._ensure_connected()
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]
        
        # Build upsert items in Endee SDK format
        items = []
        for i, (vec, meta, item_id) in enumerate(zip(vectors, metadata, ids)):
            items.append({
                "id": item_id,
                "vector": vec,
                "meta": meta
            })
        
        logger.info(f"Upserting {len(items)} vectors into '{collection}'")
        
        try:
            index = self._get_index(collection)
            index.upsert(items)
            
            return {
                "success": True,
                "inserted_count": len(items),
                "ids": ids
            }
        except Exception as e:
            logger.error(f"Failed to upsert into '{collection}': {e}")
            raise
    
    def search(
        self,
        collection: str,
        query_vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
        include_vectors: bool = False
    ) -> List[SearchResult]:
        """
        Perform similarity search on an index (collection).
        
        Args:
            collection: Index to search
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter: Optional metadata filter (not yet supported by all Endee versions)
            include_vectors: Whether to include vectors in results
        
        Returns:
            List of SearchResult objects
        
        Example:
            >>> results = client.search(
            ...     collection="papers",
            ...     query_vector=embedding,
            ...     top_k=5
            ... )
        """
        self._ensure_connected()
        
        logger.debug(f"Searching '{collection}' with top_k={top_k}")
        
        try:
            index = self._get_index(collection)
            raw_results = index.query(
                vector=query_vector,
                top_k=top_k
            )
            
            results = []
            if raw_results:
                for item in raw_results:
                    result = SearchResult(
                        id=item.get("id", ""),
                        score=item.get("similarity", item.get("score", 0.0)),
                        metadata=item.get("meta", item.get("metadata", {})),
                        vector=item.get("vector") if include_vectors else None
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed on '{collection}': {e}")
            return []
    
    def batch_search(
        self,
        collection: str,
        query_vectors: List[List[float]],
        top_k: int = 10,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            collection: Index to search
            query_vectors: List of query embedding vectors
            top_k: Number of results per query
            filter: Optional metadata filter
        
        Returns:
            List of search results for each query
        """
        all_results = []
        for query_vector in query_vectors:
            results = self.search(collection, query_vector, top_k, filter)
            all_results.append(results)
        return all_results
    
    def get_collection_stats(self, collection: str) -> Dict[str, Any]:
        """
        Get statistics for an index (collection).
        
        Args:
            collection: Index name
            
        Returns:
            Collection statistics including vector count, index info, etc.
        """
        self._ensure_connected()
        
        try:
            index = self._get_index(collection)
            # The index object itself contains metadata
            return {
                "collection": collection,
                "vector_count": getattr(index, 'count', 0),
                "index_type": "hnsw",
                "dimension": getattr(index, 'dimension', 384)
            }
        except Exception:
            return {
                "collection": collection,
                "vector_count": 0,
                "index_type": "hnsw",
                "dimension": 384
            }
    
    def delete_by_filter(
        self,
        collection: str,
        filter: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delete vectors matching a metadata filter.
        
        Args:
            collection: Index name
            filter: Metadata filter for deletion
            
        Returns:
            Deletion response with count of deleted vectors
        """
        logger.info(f"Deleting vectors from '{collection}' matching filter: {filter}")
        # Note: Endee SDK delete is by ID; filter-based deletion requires
        # first querying, then deleting by matched IDs.
        return {"deleted_count": 0}


def get_endee_client(
    host: Optional[str] = None,
    port: Optional[int] = None
) -> EndeeClient:
    """
    Factory function to create an Endee client with environment defaults.
    
    Args:
        host: Optional host override
        port: Optional port override
        
    Returns:
        Configured EndeeClient instance
    """
    from config.settings import settings
    
    client = EndeeClient(
        host=host or settings.endee_host,
        port=port or settings.endee_port
    )
    client.connect()
    return client
