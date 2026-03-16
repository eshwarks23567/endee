"""
Endee Research Assistant - Streamlit Web Application

A beautiful, interactive UI for semantic search and RAG-powered Q&A
on research papers using Endee vector database.
"""

import streamlit as st
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Endee Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    
    .result-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
    }
    
    .result-meta {
        font-size: 0.85rem;
        color: #666;
        margin-bottom: 0.75rem;
    }
    
    .result-abstract {
        font-size: 0.95rem;
        color: #444;
        line-height: 1.6;
    }
    
    .score-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Stats cards */
    .stats-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        flex: 1;
        background: white;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .stat-label {
        font-size: 0.85rem;
        color: #666;
        margin-top: 0.25rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Input styling */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
    }
    
    /* Answer box styling */
    .answer-box {
        background: #f8f9fa;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 4px solid #28a745;
    }
    
    .answer-title {
        font-weight: 600;
        color: #28a745;
        margin-bottom: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'indexed_count' not in st.session_state:
        st.session_state.indexed_count = 0


def render_header():
    """Render the application header."""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">🔬 Endee Research Assistant</div>
        <div class="header-subtitle">
            Semantic search and AI-powered Q&A for research papers
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with configuration options."""
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # Connection status
        st.markdown("### 📡 Endee Connection")
        endee_host = st.text_input("Host", value="localhost")
        endee_port = st.number_input("Port", value=8080, min_value=1, max_value=65535)
        
        if st.button("Test Connection", use_container_width=True):
            try:
                from src.endee_client import EndeeClient
                client = EndeeClient(host=endee_host, port=int(endee_port))
                if client.connect():
                    st.success("✅ Connected to Endee!")
                else:
                    st.warning("⚠️ Could not verify connection")
            except Exception as e:
                st.error(f"❌ Connection failed: {e}")
        
        st.divider()
        
        # Model settings
        st.markdown("### 🤖 Model Settings")
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-MiniLM-L6-v2"
            ],
            index=0
        )
        
        llm_model = st.selectbox(
            "LLM Model (for RAG)",
            options=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0
        )
        
        st.divider()
        
        # Data pipeline
        st.markdown("### 📥 Index Papers")
        category = st.selectbox(
            "arXiv Category",
            options=["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR"],
            index=0
        )
        max_papers = st.slider("Max Papers", min_value=10, max_value=500, value=50)
        
        if st.button("🚀 Index Papers", use_container_width=True):
            st.info("Starting indexing pipeline...")
            try:
                from src.data_pipeline import DataPipeline
                pipeline = DataPipeline()
                with st.spinner(f"Fetching and indexing {max_papers} papers from {category}..."):
                    stats = pipeline.run(category=category, max_papers=max_papers)
                st.success(f"✅ Indexed {stats.get('vectors_indexed', 0)} vectors!")
                st.session_state.indexed_count = stats.get('vectors_indexed', 0)
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")
        
        st.divider()
        
        # About section
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Endee Research Assistant** uses semantic 
        search and RAG to help you explore research papers.
        
        Built with:
        - 🗄️ [Endee](https://endee.io) Vector DB
        - 🤗 Sentence-BERT Embeddings
        - 🤖 Google Gemini
        - 🎈 Streamlit
        """)
        
        return {
            "endee_host": endee_host,
            "endee_port": int(endee_port),
            "embedding_model": embedding_model,
            "llm_model": llm_model
        }


def render_search_tab(config: dict):
    """Render the semantic search tab."""
    st.markdown("### 🔍 Semantic Paper Search")
    st.markdown("Find papers by meaning, not just keywords. Try queries like *'methods to improve transformer efficiency'*")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            label_visibility="collapsed"
        )
    with col2:
        top_k = st.selectbox("Results", options=[5, 10, 15, 20], index=1, label_visibility="collapsed")
    
    if st.button("🔍 Search", use_container_width=True) or query:
        if query:
            with st.spinner("Searching..."):
                try:
                    from src.semantic_search import SemanticSearch
                    search = SemanticSearch()
                    results = search.find_papers(query=query, top_k=top_k)
                    
                    st.session_state.search_history.append({
                        "query": query,
                        "timestamp": datetime.now().isoformat(),
                        "results_count": len(results)
                    })
                    
                    if results:
                        st.markdown(f"**Found {len(results)} relevant papers:**")
                        for i, paper in enumerate(results, 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="result-card">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div class="result-title">{i}. {paper.title}</div>
                                        <span class="score-badge">Score: {paper.score:.3f}</span>
                                    </div>
                                    <div class="result-meta">
                                        👤 {', '.join(paper.authors[:3]) if paper.authors else 'Unknown'} 
                                        {'...' if paper.authors and len(paper.authors) > 3 else ''} 
                                        | 🏷️ {', '.join(paper.categories[:2]) if paper.categories else 'N/A'}
                                        {f" | 📅 {paper.published[:10]}" if paper.published else ""}
                                    </div>
                                    <div class="result-abstract">{paper.abstract[:400]}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if paper.arxiv_id:
                                    st.markdown(f"[📄 View on arXiv](https://arxiv.org/abs/{paper.arxiv_id})")
                    else:
                        st.info("No papers found. Try indexing some papers first using the sidebar.")
                        
                except Exception as e:
                    st.error(f"Search error: {e}")
                    st.info("Make sure Endee is running and papers are indexed.")


def render_rag_tab(config: dict):
    """Render the RAG Q&A tab."""
    st.markdown("### 🤖 AI-Powered Research Q&A")
    st.markdown("Ask questions and get answers grounded in research papers.")
    
    # Check for API key
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        st.warning("⚠️ Gemini API key not set. Add `GEMINI_API_KEY` to your `.env` file to use RAG features.")
        gemini_key = st.text_input("Or enter API key here:", type="password")
    
    question = st.text_area(
        "Your Question",
        placeholder="e.g., What are the main approaches to reduce the memory footprint of large language models?",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        num_papers = st.slider("Papers to consider", min_value=3, max_value=10, value=5)
    
    if st.button("🧠 Get Answer", use_container_width=True):
        if not question:
            st.warning("Please enter a question.")
        elif not gemini_key:
            st.error("Gemini API key required for RAG.")
        else:
            with st.spinner("Researching and generating answer..."):
                try:
                    from src.rag_engine import RAGEngine
                    rag = RAGEngine(gemini_api_key=gemini_key, model=config.get("llm_model", "gemini-2.0-flash"))
                    response = rag.ask(question=question, num_papers=num_papers)
                    
                    # Display answer
                    st.markdown("""
                    <div class="answer-box">
                        <div class="answer-title">📝 Answer</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(response.answer)
                    
                    # Display confidence
                    confidence_color = "green" if response.confidence > 0.7 else "orange" if response.confidence > 0.4 else "red"
                    st.markdown(f"**Confidence:** :{confidence_color}[{response.confidence:.0%}]")
                    
                    # Display sources
                    if response.sources:
                        with st.expander(f"📚 Sources ({len(response.sources)} papers)", expanded=False):
                            for source in response.sources:
                                st.markdown(f"- **{source['title']}** - {', '.join(source.get('authors', [])[:2])}")
                    
                    # Token usage
                    if response.tokens_used:
                        st.caption(f"Tokens used: {response.tokens_used}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


def render_stats_tab():
    """Render collection statistics."""
    st.markdown("### 📊 Collection Statistics")
    
    try:
        from src.endee_client import get_endee_client
        client = get_endee_client()
        stats = client.get_collection_stats("research_papers")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("📄 Total Vectors", stats.get("vector_count", 0))
        with col2:
            st.metric("📐 Dimension", stats.get("dimension", 384))
        with col3:
            st.metric("🗂️ Index Type", stats.get("index_type", "HNSW").upper())
        
    except Exception as e:
        st.info("No statistics available. Index some papers first.")
    
    # Search history
    if st.session_state.search_history:
        st.markdown("### 📜 Recent Searches")
        for item in st.session_state.search_history[-5:][::-1]:
            st.markdown(f"- `{item['query']}` ({item['results_count']} results)")


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Search", "🤖 Ask AI", "📊 Stats"])
    
    with tab1:
        render_search_tab(config)
    
    with tab2:
        render_rag_tab(config)
    
    with tab3:
        render_stats_tab()
    
    # Footer
    st.divider()
    st.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 0.85rem;">
            Built with ❤️ using <a href="https://endee.io">Endee</a> Vector Database
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
