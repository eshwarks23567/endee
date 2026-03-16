"""
Data Pipeline Module

Fetches research papers from arXiv, processes them, and indexes into Endee.
Supports batch processing, chunking, and incremental updates.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional
import hashlib
import json
import os

logger = logging.getLogger(__name__)


@dataclass
class Paper:
    """Represents a research paper with metadata."""
    id: str
    title: str
    abstract: str
    authors: List[str]
    categories: List[str]
    published: datetime
    updated: Optional[datetime] = None
    pdf_url: Optional[str] = None
    arxiv_id: Optional[str] = None
    chunks: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert paper to dictionary format."""
        return {
            "id": self.id,
            "title": self.title,
            "abstract": self.abstract,
            "authors": self.authors,
            "categories": self.categories,
            "published": self.published.isoformat() if self.published else None,
            "updated": self.updated.isoformat() if self.updated else None,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id
        }


class ArxivFetcher:
    """
    Fetches papers from the arXiv API.
    
    Example:
        >>> fetcher = ArxivFetcher()
        >>> papers = fetcher.fetch(category="cs.AI", max_results=100)
    """
    
    ARXIV_CATEGORIES = {
        "cs.AI": "Artificial Intelligence",
        "cs.LG": "Machine Learning",
        "cs.CL": "Computation and Language (NLP)",
        "cs.CV": "Computer Vision",
        "cs.IR": "Information Retrieval",
        "stat.ML": "Machine Learning (Statistics)",
    }
    
    def __init__(self, delay_between_requests: float = 3.0):
        """
        Initialize arXiv fetcher.
        
        Args:
            delay_between_requests: Delay in seconds between API calls (be nice to arXiv)
        """
        self.delay = delay_between_requests
    
    def fetch(
        self,
        category: str = "cs.AI",
        max_results: int = 100,
        start_date: Optional[str] = None,
        query: Optional[str] = None
    ) -> List[Paper]:
        """
        Fetch papers from arXiv.
        
        Args:
            category: arXiv category (e.g., 'cs.AI', 'cs.LG')
            max_results: Maximum number of papers to fetch
            start_date: Only fetch papers after this date (YYYY-MM-DD)
            query: Additional search query
        
        Returns:
            List of Paper objects
        """
        try:
            import arxiv
        except ImportError:
            logger.error("arxiv package not installed. Running: pip install arxiv")
            raise
        
        # Build search query
        search_query = f"cat:{category}"
        if query:
            search_query = f"{search_query} AND {query}"
        
        logger.info(f"Fetching up to {max_results} papers from arXiv category: {category}")
        
        # Search arXiv
        search = arxiv.Search(
            query=search_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        papers = []
        client = arxiv.Client()
        
        for result in client.results(search):
            paper = Paper(
                id=self._generate_id(result.entry_id),
                title=result.title,
                abstract=result.summary,
                authors=[author.name for author in result.authors],
                categories=result.categories,
                published=result.published,
                updated=result.updated,
                pdf_url=result.pdf_url,
                arxiv_id=result.entry_id.split("/")[-1]
            )
            papers.append(paper)
            
            # Rate limiting
            time.sleep(0.1)
        
        logger.info(f"Fetched {len(papers)} papers from arXiv")
        return papers
    
    def _generate_id(self, arxiv_url: str) -> str:
        """Generate a unique ID from arXiv URL."""
        return hashlib.md5(arxiv_url.encode()).hexdigest()[:16]


class TextChunker:
    """
    Split long texts into smaller chunks for embedding.
    
    Uses sliding window approach to preserve context across chunk boundaries.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (discard smaller)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of text chunks
        """
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk end position
            end = start + self.chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending
                for sep in ['. ', '.\n', '? ', '! ']:
                    last_sep = text[start:end].rfind(sep)
                    if last_sep > self.min_chunk_size:
                        end = start + last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_paper(self, paper: Paper) -> List[Dict[str, Any]]:
        """
        Create chunks from a paper with metadata.
        
        Args:
            paper: Paper object
        
        Returns:
            List of chunk dictionaries with text and metadata
        """
        # Combine title and abstract
        full_text = f"{paper.title}\n\n{paper.abstract}"
        text_chunks = self.chunk_text(full_text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "paper_id": paper.id,
                "title": paper.title,
                "authors": paper.authors,
                "categories": paper.categories,
                "published": paper.published.isoformat() if paper.published else None,
                "arxiv_id": paper.arxiv_id
            })
        
        return chunks


class DataPipeline:
    """
    End-to-end pipeline for fetching, processing, and indexing papers.
    
    Example:
        >>> pipeline = DataPipeline()
        >>> pipeline.run(category="cs.AI", max_papers=100)
    """
    
    def __init__(
        self,
        endee_client=None,
        embedding_generator=None,
        collection_name: str = "research_papers"
    ):
        """
        Initialize the data pipeline.
        
        Args:
            endee_client: Endee client instance
            embedding_generator: Embedding generator instance
            collection_name: Target collection name
        """
        from .endee_client import get_endee_client
        from .embedding_generator import get_embedding_generator
        
        self.endee = endee_client or get_endee_client()
        self.embedder = embedding_generator or get_embedding_generator()
        self.collection_name = collection_name
        self.chunker = TextChunker()
        self.fetcher = ArxivFetcher()
    
    def setup_collection(self) -> None:
        """Create the collection if it doesn't exist."""
        if not self.endee.collection_exists(self.collection_name):
            try:
                self.endee.create_collection(
                    name=self.collection_name,
                    dimension=self.embedder.get_dimension(),
                    metric="cosine",
                    index_type="hnsw"
                )
                logger.info(f"Created collection: {self.collection_name}")
            except Exception as e:
                logger.warning(f"Collection setup: {e} (may already exist)")
    
    def run(
        self,
        category: str = "cs.AI",
        max_papers: int = 100,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Run the full data pipeline.
        
        Args:
            category: arXiv category to fetch
            max_papers: Maximum papers to fetch
            batch_size: Batch size for embedding and indexing
        
        Returns:
            Pipeline statistics
        """
        start_time = time.time()
        stats = {
            "papers_fetched": 0,
            "chunks_created": 0,
            "vectors_indexed": 0,
            "errors": 0
        }
        
        logger.info(f"Starting data pipeline for category: {category}")
        
        # Setup collection
        self.setup_collection()
        
        # Fetch papers
        papers = self.fetcher.fetch(category=category, max_results=max_papers)
        stats["papers_fetched"] = len(papers)
        
        # Process in batches
        all_chunks = []
        for paper in papers:
            chunks = self.chunker.chunk_paper(paper)
            all_chunks.extend(chunks)
        
        stats["chunks_created"] = len(all_chunks)
        
        # Generate embeddings and index
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            
            # Generate embeddings
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.embedder.embed(texts)
            
            # Index in Endee
            vectors = embeddings.tolist()
            metadata = [{k: v for k, v in chunk.items() if k != "text"} for chunk in batch]
            
            # Add text to metadata for retrieval
            for j, chunk in enumerate(batch):
                metadata[j]["text"] = chunk["text"]
            
            result = self.endee.insert(
                collection=self.collection_name,
                vectors=vectors,
                metadata=metadata
            )
            
            stats["vectors_indexed"] += result.get("inserted_count", len(batch)) if isinstance(result, dict) else len(batch)
            logger.info(f"Indexed batch {i // batch_size + 1}: {len(batch)} vectors")
        
        elapsed = time.time() - start_time
        stats["elapsed_seconds"] = elapsed
        
        logger.info(f"Pipeline complete: {stats}")
        return stats
    
    def index_papers_from_file(
        self,
        file_path: str,
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Index papers from a JSON file.
        
        Args:
            file_path: Path to JSON file with papers
            batch_size: Batch size for processing
        
        Returns:
            Indexing statistics
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
        
        papers = []
        for p in papers_data:
            paper = Paper(
                id=p.get("id", hashlib.md5(p["title"].encode()).hexdigest()[:16]),
                title=p["title"],
                abstract=p["abstract"],
                authors=p.get("authors", []),
                categories=p.get("categories", []),
                published=datetime.fromisoformat(p["published"]) if p.get("published") else datetime.now(),
                arxiv_id=p.get("arxiv_id")
            )
            papers.append(paper)
        
        # Process same as run() method
        self.setup_collection()
        
        all_chunks = []
        for paper in papers:
            chunks = self.chunker.chunk_paper(paper)
            all_chunks.extend(chunks)
        
        stats = {"papers": len(papers), "chunks": len(all_chunks), "indexed": 0}
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            texts = [chunk["text"] for chunk in batch]
            embeddings = self.embedder.embed(texts)
            
            vectors = embeddings.tolist()
            metadata = batch
            
            self.endee.insert(
                collection=self.collection_name,
                vectors=vectors,
                metadata=metadata
            )
            stats["indexed"] += len(batch)
        
        return stats


def main():
    """CLI entry point for the data pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Endee Research Assistant Data Pipeline")
    parser.add_argument("--category", type=str, default="cs.AI", help="arXiv category")
    parser.add_argument("--max-papers", type=int, default=100, help="Maximum papers to fetch")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--file", type=str, help="Index from JSON file instead of arXiv")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    pipeline = DataPipeline()
    
    if args.file:
        stats = pipeline.index_papers_from_file(args.file, batch_size=args.batch_size)
    else:
        stats = pipeline.run(
            category=args.category,
            max_papers=args.max_papers,
            batch_size=args.batch_size
        )
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Papers fetched: {stats.get('papers_fetched', stats.get('papers', 0))}")
    print(f"   Chunks created: {stats.get('chunks_created', stats.get('chunks', 0))}")
    print(f"   Vectors indexed: {stats.get('vectors_indexed', stats.get('indexed', 0))}")


if __name__ == "__main__":
    main()
