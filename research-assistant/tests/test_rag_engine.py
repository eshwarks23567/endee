"""
Tests for the RAG engine.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rag_engine import RAGEngine, RAGResponse


class TestRAGResponse:
    """Test cases for RAGResponse dataclass."""
    
    def test_rag_response_creation(self):
        """Test creating a RAG response."""
        response = RAGResponse(
            question="What is NLP?",
            answer="NLP is Natural Language Processing.",
            sources=[{"title": "NLP Paper"}],
            confidence=0.85,
            tokens_used=100
        )
        
        assert response.question == "What is NLP?"
        assert response.answer == "NLP is Natural Language Processing."
        assert len(response.sources) == 1
        assert response.confidence == 0.85
        assert response.tokens_used == 100
    
    def test_rag_response_to_dict(self):
        """Test converting RAG response to dictionary."""
        response = RAGResponse(
            question="Test?",
            answer="Answer",
            sources=[],
            confidence=0.5
        )
        
        result = response.to_dict()
        
        assert isinstance(result, dict)
        assert "question" in result
        assert "answer" in result
        assert "sources" in result
        assert "confidence" in result
    
    def test_rag_response_default_tokens(self):
        """Test default tokens_used value."""
        response = RAGResponse(
            question="Test?",
            answer="Answer",
            sources=[],
            confidence=0.5
        )
        
        assert response.tokens_used == 0


class TestRAGEngine:
    """Test cases for RAGEngine."""
    
    @patch.dict(os.environ, {"GEMINI_API_KEY": ""})
    def test_engine_without_api_key(self):
        """Test engine raises error without API key."""
        engine = RAGEngine(gemini_api_key=None)
        
        with pytest.raises(ValueError, match="Gemini API key not set"):
            _ = engine.model
    
    def test_engine_with_api_key(self):
        """Test engine initialization with API key."""
        engine = RAGEngine(gemini_api_key="test-key")
        
        assert engine.api_key == "test-key"
        assert engine.model_name == "gemini-2.0-flash"
    
    def test_build_context(self):
        """Test context building from papers."""
        engine = RAGEngine(gemini_api_key="test-key")
        
        # Create mock papers
        paper1 = Mock()
        paper1.title = "Paper 1"
        paper1.authors = ["Author A"]
        paper1.arxiv_id = "2024.1234"
        paper1.abstract = "Abstract of paper 1"
        paper1.chunk_text = None
        
        paper2 = Mock()
        paper2.title = "Paper 2"
        paper2.authors = ["Author B", "Author C"]
        paper2.arxiv_id = "2024.5678"
        paper2.abstract = "Abstract of paper 2"
        paper2.chunk_text = None
        
        context = engine._build_context([paper1, paper2])
        
        assert "Paper 1" in context
        assert "Paper 2" in context
        assert "Author A" in context
        assert "Abstract of paper 1" in context
    
    def test_estimate_confidence_high(self):
        """Test confidence estimation with high scores."""
        engine = RAGEngine(gemini_api_key="test-key")
        
        papers = [Mock(score=0.95), Mock(score=0.90), Mock(score=0.85)]
        answer = "This is a confident answer."
        
        confidence = engine._estimate_confidence(papers, answer)
        
        assert confidence > 0.8
    
    def test_estimate_confidence_with_uncertainty(self):
        """Test confidence estimation with uncertain answer."""
        engine = RAGEngine(gemini_api_key="test-key")
        
        papers = [Mock(score=0.95)]
        answer = "I couldn't find enough information to fully answer this."
        
        confidence = engine._estimate_confidence(papers, answer)
        
        assert confidence < 0.95 * 0.8  # Reduced due to uncertainty
    
    def test_estimate_confidence_empty_papers(self):
        """Test confidence with no papers."""
        engine = RAGEngine(gemini_api_key="test-key")
        
        confidence = engine._estimate_confidence([], "Some answer")
        
        assert confidence == 0.0
    
    def test_ask_no_papers_found(self):
        """Test asking when no papers are found."""
        mock_search = Mock()
        mock_search.find_papers.return_value = []
        
        engine = RAGEngine(
            semantic_search=mock_search,
            gemini_api_key="test-key"
        )
        
        response = engine.ask("What is something?")
        
        assert "couldn't find" in response.answer.lower()
        assert response.confidence == 0.0
        assert len(response.sources) == 0


class TestRAGEngineIntegration:
    """Integration-style tests for RAG engine."""
    
    @patch('google.generativeai.GenerativeModel')
    @patch('google.generativeai.configure')
    def test_ask_with_papers(self, mock_genai_configure, mock_genai_model_class):
        """Test asking with papers available."""
        # Mock search
        mock_paper = Mock()
        mock_paper.title = "Test Paper"
        mock_paper.authors = ["Author"]
        mock_paper.arxiv_id = "1234"
        mock_paper.abstract = "Test abstract"
        mock_paper.chunk_text = None
        mock_paper.score = 0.9
        mock_paper.to_dict.return_value = {"title": "Test Paper"}
        
        mock_search = Mock()
        mock_search.find_papers.return_value = [mock_paper]
        
        # Mock Gemini model
        mock_response = Mock()
        mock_response.text = "Generated answer"
        
        mock_model = Mock()
        mock_model.generate_content.return_value = mock_response
        mock_genai_model_class.return_value = mock_model
        
        # Test
        engine = RAGEngine(
            semantic_search=mock_search,
            gemini_api_key="test-key"
        )
        
        response = engine.ask("What is this about?")
        
        assert response.answer == "Generated answer"
        assert len(response.sources) == 1
        assert response.tokens_used > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
