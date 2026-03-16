"""
Tests for the Endee client wrapper.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.endee_client import EndeeClient, SearchResult


class TestEndeeClient:
    """Test cases for EndeeClient."""
    
    def test_client_initialization(self):
        """Test client initializes with correct defaults."""
        client = EndeeClient()
        
        assert client.host == "localhost"
        assert client.port == 8080
        assert client.base_url == "http://localhost:8080/api/v1"
    
    def test_client_custom_config(self):
        """Test client with custom configuration."""
        client = EndeeClient(
            host="endee-server",
            port=9000,
            api_key="test-key"
        )
        
        assert client.host == "endee-server"
        assert client.port == 9000
        assert client.base_url == "http://endee-server:9000/api/v1"
        assert client.api_key == "test-key"
    
    @patch('src.endee_client.EndeeClient._ensure_connected')
    def test_create_collection(self, mock_connect):
        """Test collection creation calls Endee SDK."""
        client = EndeeClient()
        
        # Mock the internal client
        mock_endee = MagicMock()
        mock_index = MagicMock()
        mock_endee.get_index.return_value = mock_index
        client._client = mock_endee
        client._connected = True
        
        with patch('endee.Precision') as mock_precision:
            mock_precision.FLOAT32 = "float32"
            result = client.create_collection(
                name="test_collection",
                dimension=384,
                metric="cosine",
                index_type="hnsw"
            )
        
        assert result["success"] is True
        assert result["collection"] == "test_collection"
        assert result["dimension"] == 384
        mock_endee.create_index.assert_called_once()
    
    @patch('src.endee_client.EndeeClient._ensure_connected')
    def test_insert_vectors(self, mock_connect):
        """Test vector insertion via upsert."""
        client = EndeeClient()
        
        # Mock the Endee index
        mock_index = MagicMock()
        client._client = MagicMock()
        client._connected = True
        client._indexes["test_collection"] = mock_index
        
        vectors = [[0.1] * 384, [0.2] * 384]
        metadata = [{"title": "Paper 1"}, {"title": "Paper 2"}]
        
        result = client.insert(
            collection="test_collection",
            vectors=vectors,
            metadata=metadata
        )
        
        assert result["success"] is True
        assert result["inserted_count"] == 2
        assert len(result["ids"]) == 2
        mock_index.upsert.assert_called_once()
    
    def test_insert_mismatched_lengths(self):
        """Test that mismatched vectors/metadata raises error."""
        client = EndeeClient()
        
        vectors = [[0.1] * 384]
        metadata = [{"title": "Paper 1"}, {"title": "Paper 2"}]
        
        with pytest.raises(ValueError):
            client.insert(
                collection="test_collection",
                vectors=vectors,
                metadata=metadata
            )
    
    @patch('src.endee_client.EndeeClient._ensure_connected')
    def test_search(self, mock_connect):
        """Test similarity search."""
        client = EndeeClient()
        
        # Mock the Endee index
        mock_index = MagicMock()
        mock_index.query.return_value = [
            {"id": "doc1", "similarity": 0.95, "meta": {"title": "Paper 1"}},
            {"id": "doc2", "similarity": 0.85, "meta": {"title": "Paper 2"}}
        ]
        client._client = MagicMock()
        client._connected = True
        client._indexes["test_collection"] = mock_index
        
        results = client.search(
            collection="test_collection",
            query_vector=[0.1] * 384,
            top_k=5
        )
        
        assert len(results) == 2
        assert results[0].id == "doc1"
        assert results[0].score == 0.95
        assert results[0].metadata["title"] == "Paper 1"
        mock_index.query.assert_called_once()
    
    @patch('src.endee_client.EndeeClient._ensure_connected')
    def test_collection_exists_true(self, mock_connect):
        """Test collection_exists returns True when index exists."""
        client = EndeeClient()
        mock_endee = MagicMock()
        mock_endee.get_index.return_value = MagicMock()
        client._client = mock_endee
        client._connected = True
        
        assert client.collection_exists("test_collection") is True
    
    @patch('src.endee_client.EndeeClient._ensure_connected')
    def test_collection_exists_false(self, mock_connect):
        """Test collection_exists returns False when index doesn't exist."""
        client = EndeeClient()
        mock_endee = MagicMock()
        mock_endee.get_index.side_effect = Exception("Not found")
        client._client = mock_endee
        client._connected = True
        
        assert client.collection_exists("nonexistent") is False
    
    def test_search_result_dataclass(self):
        """Test SearchResult dataclass."""
        result = SearchResult(
            id="test-id",
            score=0.95,
            metadata={"title": "Test Paper"}
        )
        
        assert result.id == "test-id"
        assert result.score == 0.95
        assert result.metadata["title"] == "Test Paper"
        assert result.vector is None
    
    @patch('src.endee_client.EndeeClient._ensure_connected')
    def test_collection_stats(self, mock_connect):
        """Test getting collection statistics."""
        client = EndeeClient()
        mock_index = MagicMock()
        mock_index.count = 100
        mock_index.dimension = 384
        client._client = MagicMock()
        client._client.get_index.return_value = mock_index
        client._connected = True
        
        stats = client.get_collection_stats("test_collection")
        
        assert "collection" in stats
        assert "vector_count" in stats
        assert "index_type" in stats


class TestSearchResult:
    """Test cases for SearchResult dataclass."""
    
    def test_search_result_with_vector(self):
        """Test SearchResult with vector included."""
        result = SearchResult(
            id="test-id",
            score=0.85,
            metadata={"title": "Paper"},
            vector=[0.1, 0.2, 0.3]
        )
        
        assert result.vector == [0.1, 0.2, 0.3]
    
    def test_search_result_default_vector(self):
        """Test SearchResult without vector."""
        result = SearchResult(
            id="test-id",
            score=0.85,
            metadata={}
        )
        
        assert result.vector is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
