"""
Semantic Search Module

Provides semantic search functionality over research papers using Endee.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a semantic search result."""
    title: str
    abstract: str
    authors: List[str]
    score: float
    arxiv_id: Optional[str] = None
    categories: List[str] = None
    published: Optional[str] = None
    chunk_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "score": self.score,
            "arxiv_id": self.arxiv_id,
            "categories": self.categories,
            "published": self.published
        }


class SemanticSearch:
    """
    Semantic search over research papers using Endee vector database.
    
    Example:
        >>> search = SemanticSearch()
        >>> results = search.find_papers("transformer attention mechanisms", top_k=5)
        >>> for r in results:
        ...     print(f"{r.title} (score: {r.score:.3f})")
    """
    
    def __init__(
        self,
        endee_client=None,
        embedding_generator=None,
        collection_name: str = "research_papers"
    ):
        """
        Initialize semantic search.
        
        Args:
            endee_client: Endee client instance
            embedding_generator: Embedding generator instance
            collection_name: Collection to search
        """
        from .endee_client import get_endee_client
        from .embedding_generator import get_embedding_generator
        
        self.endee = endee_client or get_endee_client()
        self.embedder = embedding_generator or get_embedding_generator()
        self.collection_name = collection_name
    
    def find_papers(
        self,
        query: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        Find papers similar to the query using semantic search.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            category_filter: Optional category filter (e.g., 'cs.AI')
            min_score: Minimum similarity score threshold
        
        Returns:
            List of SearchResult objects sorted by relevance
        """
        logger.info(f"Searching for: '{query}' (top_k={top_k})")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_single(query)
        
        # Build filter
        filter_dict = None
        if category_filter:
            filter_dict = {"categories": category_filter}
        
        # Search Endee
        raw_results = self.endee.search(
            collection=self.collection_name,
            query_vector=query_embedding,
            top_k=top_k * 2,  # Fetch extra for deduplication
            filter=filter_dict
        )
        
        # Process and deduplicate results
        seen_titles = set()
        results = []
        
        for r in raw_results:
            title = r.metadata.get("title", "")
            
            # Skip duplicates (same paper, different chunks)
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            # Skip low scores
            if r.score < min_score:
                continue
            
            result = SearchResult(
                title=title,
                abstract=r.metadata.get("abstract", r.metadata.get("text", "")),
                authors=r.metadata.get("authors", []),
                score=r.score,
                arxiv_id=r.metadata.get("arxiv_id"),
                categories=r.metadata.get("categories", []),
                published=r.metadata.get("published"),
                chunk_text=r.metadata.get("text")
            )
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        logger.info(f"Found {len(results)} unique papers")
        return results
    
    def find_similar_to_paper(
        self,
        paper_id: str,
        top_k: int = 5,
        exclude_self: bool = True
    ) -> List[SearchResult]:
        """
        Find papers similar to a given paper.
        
        Args:
            paper_id: ID of the source paper
            top_k: Number of similar papers to return
            exclude_self: Whether to exclude the source paper
        
        Returns:
            List of similar papers
        """
        # First, get the paper's embedding (using its text)
        # In a real implementation, you'd retrieve the stored embedding
        # For now, we'll search by paper title
        
        # This would need the paper's text or stored embedding
        # Placeholder implementation
        logger.warning("find_similar_to_paper requires stored embeddings - returning empty")
        return []
    
    def hybrid_search(
        self,
        query: str,
        keywords: List[str],
        top_k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """
        Combine semantic search with keyword filtering.
        
        Args:
            query: Semantic search query
            keywords: Keywords that must appear in results
            top_k: Number of results
            semantic_weight: Weight for semantic vs keyword matching (0-1)
        
        Returns:
            Hybrid search results
        """
        # Get semantic results
        semantic_results = self.find_papers(query, top_k=top_k * 2)
        
        # Filter by keywords
        filtered_results = []
        for result in semantic_results:
            text_lower = (result.title + " " + result.abstract).lower()
            matches_keywords = all(kw.lower() in text_lower for kw in keywords)
            
            if matches_keywords:
                # Boost score for keyword matches
                result.score = (result.score * semantic_weight) + ((1 - semantic_weight) * 1.0)
                filtered_results.append(result)
        
        # Sort by adjusted score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_results[:top_k]
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the search collection."""
        return self.endee.get_collection_stats(self.collection_name)
