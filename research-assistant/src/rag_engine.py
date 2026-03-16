"""
RAG (Retrieval Augmented Generation) Engine

Combines semantic search with Google Gemini LLM to provide intelligent Q&A
grounded in research papers.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import os

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Represents a RAG-generated response."""
    question: str
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    tokens_used: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used
        }


class RAGEngine:
    """
    RAG Engine for research paper Q&A using Google Gemini.
    
    This engine:
    1. Retrieves relevant papers using semantic search
    2. Builds a context from the retrieved papers
    3. Uses Gemini LLM to generate answers grounded in the context
    
    Example:
        >>> rag = RAGEngine()
        >>> response = rag.ask("What are the main approaches to improve transformer efficiency?")
        >>> print(response.answer)
        >>> print(response.sources)
    """
    
    # System prompt for the RAG assistant
    SYSTEM_PROMPT = """You are a research assistant specializing in AI and machine learning papers. 
Your task is to answer questions based on the provided research paper abstracts.

Guidelines:
1. Only use information from the provided context
2. If the context doesn't contain enough information, say so
3. Cite specific papers when making claims
4. Be concise but thorough
5. If asked about something not in the context, acknowledge the limitation

Format your response clearly with key points and citations."""

    def __init__(
        self,
        semantic_search=None,
        gemini_api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        max_context_papers: int = 5
    ):
        """
        Initialize the RAG engine.
        
        Args:
            semantic_search: SemanticSearch instance
            gemini_api_key: Google Gemini API key (or from env)
            model: Gemini model to use
            max_context_papers: Maximum papers to include in context
        """
        from .semantic_search import SemanticSearch
        
        self.search = semantic_search or SemanticSearch()
        self.model_name = model
        self.max_context_papers = max_context_papers
        
        # Get API key
        self.api_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self._model = None
    
    @property
    def model(self):
        """Lazy load the Gemini model."""
        if self._model is None:
            if not self.api_key:
                raise ValueError("Gemini API key not set. Set GEMINI_API_KEY environment variable.")
            
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._model = genai.GenerativeModel(self.model_name)
                logger.info(f"Initialized Gemini model: {self.model_name}")
            except ImportError:
                logger.error("google-generativeai not installed. Run: pip install google-generativeai")
                raise
        return self._model
    
    def ask(
        self,
        question: str,
        num_papers: int = 5,
        category_filter: Optional[str] = None
    ) -> RAGResponse:
        """
        Ask a question and get an answer grounded in research papers.
        
        Args:
            question: Natural language question
            num_papers: Number of papers to retrieve for context
            category_filter: Optional category filter
        
        Returns:
            RAGResponse with answer and sources
        """
        logger.info(f"RAG query: '{question}'")
        
        # Retrieve relevant papers
        papers = self.search.find_papers(
            query=question,
            top_k=min(num_papers, self.max_context_papers),
            category_filter=category_filter
        )
        
        if not papers:
            return RAGResponse(
                question=question,
                answer="I couldn't find any relevant papers in the database to answer this question. "
                       "Please try indexing some papers first using the data pipeline.",
                sources=[],
                confidence=0.0
            )
        
        # Build context from papers
        context = self._build_context(papers)
        
        # Generate answer using Gemini
        try:
            answer, tokens = self._generate_answer(question, context)
            confidence = self._estimate_confidence(papers, answer)
            
            return RAGResponse(
                question=question,
                answer=answer,
                sources=[p.to_dict() for p in papers],
                confidence=confidence,
                tokens_used=tokens
            )
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return RAGResponse(
                question=question,
                answer=f"Error generating answer: {str(e)}. Please check your Gemini API key.",
                sources=[p.to_dict() for p in papers],
                confidence=0.0
            )
    
    def _build_context(self, papers: List) -> str:
        """Build context string from retrieved papers."""
        context_parts = []
        
        for i, paper in enumerate(papers, 1):
            paper_context = f"""
[Paper {i}]
Title: {paper.title}
Authors: {', '.join(paper.authors) if paper.authors else 'Unknown'}
arXiv ID: {paper.arxiv_id or 'N/A'}

Abstract: {paper.abstract or paper.chunk_text}
"""
            context_parts.append(paper_context.strip())
        
        return "\n\n---\n\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> tuple:
        """Generate answer using Google Gemini."""
        user_prompt = f"""{self.SYSTEM_PROMPT}

Based on the following research paper abstracts, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based only on the information in the provided papers. 
Cite specific papers by their title when making claims."""

        response = self.model.generate_content(
            user_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000,
            }
        )
        
        answer = response.text
        # Gemini doesn't provide exact token counts in the same way, estimate
        tokens = len(user_prompt.split()) + len(answer.split())
        
        return answer, tokens
    
    def _estimate_confidence(self, papers: List, answer: str) -> float:
        """Estimate confidence based on search scores and answer quality."""
        if not papers:
            return 0.0
        
        # Average top paper scores
        avg_score = sum(p.score for p in papers[:3]) / min(len(papers), 3)
        
        # Check if answer acknowledges limitations
        uncertainty_phrases = [
            "i couldn't find",
            "not enough information",
            "unclear",
            "not mentioned",
            "no information"
        ]
        has_uncertainty = any(phrase in answer.lower() for phrase in uncertainty_phrases)
        
        confidence = avg_score
        if has_uncertainty:
            confidence *= 0.7
        
        return min(confidence, 1.0)
    
    def stream_ask(
        self,
        question: str,
        num_papers: int = 5
    ):
        """
        Stream the RAG response (for real-time UI).
        
        Args:
            question: Natural language question
            num_papers: Number of papers for context
        
        Yields:
            Response chunks as they're generated
        """
        # Retrieve papers
        papers = self.search.find_papers(query=question, top_k=num_papers)
        
        if not papers:
            yield "I couldn't find any relevant papers in the database."
            return
        
        context = self._build_context(papers)
        
        user_prompt = f"""{self.SYSTEM_PROMPT}

Based on the following research paper abstracts, please answer the question.

CONTEXT:
{context}

QUESTION: {question}

Please provide a comprehensive answer based only on the information in the provided papers."""

        # Stream response from Gemini
        response = self.model.generate_content(
            user_prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1000,
            },
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    
    def summarize_papers(
        self,
        topic: str,
        num_papers: int = 5
    ) -> str:
        """
        Generate a summary of papers on a topic.
        
        Args:
            topic: Topic to summarize
            num_papers: Number of papers to include
        
        Returns:
            Summary text
        """
        papers = self.search.find_papers(query=topic, top_k=num_papers)
        
        if not papers:
            return "No papers found on this topic."
        
        context = self._build_context(papers)
        
        prompt = f"""Summarize the key findings and themes from these research papers on "{topic}":

{context}

Provide a structured summary covering:
1. Main approaches/methods used
2. Key findings
3. Common themes
4. Gaps or future directions mentioned"""

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 1500,
            }
        )
        
        return response.text


def create_rag_engine(
    gemini_api_key: Optional[str] = None,
    model: str = "gemini-2.0-flash"
) -> RAGEngine:
    """
    Factory function to create a RAG engine.
    
    Args:
        gemini_api_key: Optional API key override
        model: Gemini model to use
    
    Returns:
        Configured RAGEngine instance
    """
    return RAGEngine(gemini_api_key=gemini_api_key, model=model)
