"""
Embedding Generator Module

Generates text embeddings using Sentence-BERT models.
Supports batching, caching, and multiple model options.
"""

import logging
from typing import List, Optional, Union
from functools import lru_cache
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generate text embeddings using Sentence-BERT models.
    
    This class handles:
    - Loading and caching transformer models
    - Batched embedding generation
    - Text preprocessing and normalization
    
    Example:
        >>> generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        >>> embeddings = generator.embed(["Hello world", "How are you?"])
    """
    
    # Model dimension mapping
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-mpnet-base-v2": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "all-distilroberta-v1": 768,
        "multi-qa-MiniLM-L6-cos-v1": 384,
    }
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Sentence-BERT model name from HuggingFace
            device: Device to run model on ('cpu', 'cuda', 'mps')
            normalize: Whether to L2-normalize embeddings
        """
        self.model_name = model_name
        self.normalize = normalize
        self.device = device
        self._model = None
        
        # Get dimension for this model
        self.dimension = self.MODEL_DIMENSIONS.get(model_name, 384)
        
        logger.info(f"Initialized EmbeddingGenerator with model: {model_name}")
    
    @property
    def model(self):
        """Lazy load the model on first use."""
        if self._model is None:
            self._load_model()
        return self._model
    
    def _load_model(self):
        """Load the Sentence-BERT model."""
        try:
            from sentence_transformers import SentenceTransformer
            
            logger.info(f"Loading model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Model loaded successfully on device: {self._model.device}")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def embed(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts to embed
            batch_size: Batch size for processing
            show_progress: Show progress bar for large batches
        
        Returns:
            NumPy array of embeddings, shape (n_texts, dimension)
        
        Example:
            >>> embeddings = generator.embed(["Machine learning paper"])
            >>> print(embeddings.shape)  # (1, 384)
        """
        # Handle single text input
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocess texts
        texts = [self._preprocess(text) for text in texts]
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def embed_single(self, text: str) -> List[float]:
        """
        Embed a single text and return as list.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding as a list of floats
        """
        embedding = self.embed(text)
        return embedding[0].tolist()
    
    def _preprocess(self, text: str) -> str:
        """
        Preprocess text before embedding.
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate very long texts (BERT max is ~512 tokens)
        max_chars = 8000  # Roughly ~2000 tokens
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
            logger.debug(f"Truncated text to {max_chars} characters")
        
        return text
    
    def similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Cosine similarity score (0-1 for normalized embeddings)
        """
        embeddings = self.embed([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1])
        return float(similarity)
    
    def get_dimension(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.dimension


@lru_cache(maxsize=1)
def get_embedding_generator(model_name: str = "all-MiniLM-L6-v2") -> EmbeddingGenerator:
    """
    Get a cached embedding generator instance.
    
    Args:
        model_name: Model to use
        
    Returns:
        EmbeddingGenerator instance
    """
    return EmbeddingGenerator(model_name=model_name)


def compute_embeddings(
    texts: List[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32
) -> np.ndarray:
    """
    Convenience function to compute embeddings.
    
    Args:
        texts: List of texts to embed
        model_name: Model to use
        batch_size: Batch size
    
    Returns:
        NumPy array of embeddings
    """
    generator = get_embedding_generator(model_name)
    return generator.embed(texts, batch_size=batch_size)
