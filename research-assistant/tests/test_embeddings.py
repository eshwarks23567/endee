"""
Tests for the embedding generator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.embedding_generator import EmbeddingGenerator


class TestEmbeddingGenerator:
    """Test cases for EmbeddingGenerator."""
    
    def test_initialization(self):
        """Test generator initializes with correct defaults."""
        generator = EmbeddingGenerator()
        
        assert generator.model_name == "all-MiniLM-L6-v2"
        assert generator.normalize is True
        assert generator.dimension == 384
    
    def test_custom_model(self):
        """Test generator with custom model."""
        generator = EmbeddingGenerator(model_name="all-mpnet-base-v2")
        
        assert generator.model_name == "all-mpnet-base-v2"
        assert generator.dimension == 768
    
    def test_get_dimension(self):
        """Test dimension retrieval."""
        generator = EmbeddingGenerator(model_name="all-MiniLM-L6-v2")
        assert generator.get_dimension() == 384
        
        generator2 = EmbeddingGenerator(model_name="all-mpnet-base-v2")
        assert generator2.get_dimension() == 768
    
    def test_preprocess_text(self):
        """Test text preprocessing."""
        generator = EmbeddingGenerator()
        
        # Test whitespace normalization
        text = "This   has   extra    spaces"
        preprocessed = generator._preprocess(text)
        assert preprocessed == "This has extra spaces"
        
        # Test newline handling
        text = "Line 1\n\nLine 2"
        preprocessed = generator._preprocess(text)
        assert preprocessed == "Line 1 Line 2"
    
    def test_preprocess_truncation(self):
        """Test long text truncation."""
        generator = EmbeddingGenerator()
        
        long_text = "A" * 10000
        preprocessed = generator._preprocess(long_text)
        
        assert len(preprocessed) <= 8003  # 8000 + "..."
        assert preprocessed.endswith("...")
    
    @patch('src.embedding_generator.EmbeddingGenerator._load_model')
    def test_embed_single_text(self, mock_load):
        """Test embedding a single text."""
        generator = EmbeddingGenerator()
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        generator._model = mock_model
        
        result = generator.embed("Test text")
        
        assert result.shape == (1, 384)
        mock_model.encode.assert_called_once()
    
    @patch('src.embedding_generator.EmbeddingGenerator._load_model')
    def test_embed_multiple_texts(self, mock_load):
        """Test embedding multiple texts."""
        generator = EmbeddingGenerator()
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        generator._model = mock_model
        
        result = generator.embed(["Text 1", "Text 2"])
        
        assert result.shape == (2, 384)
    
    @patch('src.embedding_generator.EmbeddingGenerator._load_model')
    def test_embed_single_returns_list(self, mock_load):
        """Test embed_single returns a flat list."""
        generator = EmbeddingGenerator()
        
        # Mock the model
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384])
        generator._model = mock_model
        
        result = generator.embed_single("Test text")
        
        assert isinstance(result, list)
        assert len(result) == 384
    
    @patch('src.embedding_generator.EmbeddingGenerator._load_model')
    def test_similarity(self, mock_load):
        """Test similarity computation."""
        generator = EmbeddingGenerator()
        
        # Mock normalized embeddings (dot product = cosine similarity)
        mock_model = MagicMock()
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        mock_model.encode.return_value = np.array([vec1, vec2])
        generator._model = mock_model
        
        similarity = generator.similarity("Text 1", "Text 2")
        
        assert 0 <= similarity <= 1


class TestModelDimensions:
    """Test model dimension mappings."""
    
    def test_all_models_have_dimensions(self):
        """Test that all listed models have dimensions."""
        for model_name in EmbeddingGenerator.MODEL_DIMENSIONS:
            generator = EmbeddingGenerator(model_name=model_name)
            assert generator.dimension > 0
    
    def test_unknown_model_default_dimension(self):
        """Test unknown model gets default dimension."""
        generator = EmbeddingGenerator(model_name="unknown-model")
        assert generator.dimension == 384


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
