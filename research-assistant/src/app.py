"""
Endee Research Assistant - Streamlit Web Application

A clean, structured interface for semantic search and RAG-powered analysis
on research papers using the Endee vector database.
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
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a premium, intentional design system
st.markdown("""
<style>
    /* Typography System - Inter (sans-serif) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        color: #0f172a; /* Slate 900 */
    }

    /* 8-Point Spacing Rhythm System */
    .main {
        padding: 48px 40px; 
    }
    
    /* Header styling - Clean, grounded, hierarchy-driven */
    .header-container {
        padding: 0 0 40px 0;
        margin-bottom: 40px;
        border-bottom: 1px solid #e2e8f0; /* Slate 200 */
    }
    
    .header-title {
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #0f172a;
        margin-bottom: 8px;
    }
    
    .header-subtitle {
        font-size: 16px;
        font-weight: 400;
        color: #64748b; /* Slate 500 */
        line-height: 1.5;
    }
    
    /* Card Component System - shared radius, padding, and border */
    .result-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
        margin-bottom: 16px;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    
    .result-card:hover {
        border-color: #cbd5e1; /* Slate 300 */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05); /* Tailwind shadow-md */
    }
    
    .result-title {
        font-size: 18px;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 8px;
        line-height: 1.4;
    }
    
    .result-meta {
        font-size: 14px;
        color: #64748b;
        margin-bottom: 16px;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }
    
    .result-abstract {
        font-size: 14px;
        color: #334155; /* Slate 700 */
        line-height: 1.6;
    }
    
    .score-badge {
        display: inline-block;
        background: #f1f5f9; /* Slate 100 */
        color: #475569; /* Slate 600 */
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        font-family: ui-monospace, SFMono-Regular, monospace;
        border: 1px solid #e2e8f0;
    }
    
    /* Stats Layout System */
    .stats-container {
        display: flex;
        gap: 24px;
        margin-bottom: 32px;
    }
    
    .stat-card {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 24px;
    }
    
    .stat-value {
        font-size: 32px;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.02em;
    }
    
    .stat-label {
        font-size: 14px;
        font-weight: 500;
        color: #64748b;
        margin-top: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Streamlit Overrides for Consistent UI Forms */
    
    /* Primary Buttons */
    .stButton>button {
        background-color: #0f172a !important; /* Slate 900 */
        color: #ffffff !important;
        border: 1px solid #0f172a !important;
        padding: 8px 16px !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        min-height: 40px !important;
        transition: background-color 0.15s ease !important;
        box-shadow: none !important;
    }
    
    .stButton>button:hover {
        background-color: #334155 !important; /* Slate 700 */
        border-color: #334155 !important;
        transform: none !important; /* Remove jumpy hover */
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        border-radius: 6px !important;
        border: 1px solid #cbd5e1 !important; /* Slate 300 */
        padding: 10px 12px !important;
        font-size: 14px !important;
        background-color: #ffffff !important;
        color: #0f172a !important;
        box-shadow: none !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #3b82f6 !important; /* Blue 500 */
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* Answer Analysis Box */
    .answer-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #0f172a;
        border-radius: 8px;
        padding: 24px;
        margin-top: 24px;
    }
    
    .answer-title {
        font-size: 12px;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 16px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Sidebar */
    .sidebar .sidebar-content {
        background: #f8fafc; /* Slate 50 */
        border-right: 1px solid #e2e8f0;
    }
    
    /* Remove streamlit branding padding if possible */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    if 'indexed_count' not in st.session_state:
        st.session_state.indexed_count = 0


def render_header():
    """Render the application header following strict typography rules."""
    st.markdown("""
    <div class="header-container">
        <div class="header-title">Endee Research Assistant</div>
        <div class="header-subtitle">
            Semantic search and document analysis powered by vector retrieval.
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with crisp configuration groupings."""
    with st.sidebar:
        st.markdown("<h3 style='font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 16px;'>Configuration</h3>", unsafe_allow_html=True)
        
        # Connection status
        st.markdown("<div style='font-weight: 500; font-size: 14px; margin-bottom: 8px;'>Endee Connection</div>", unsafe_allow_html=True)
        endee_host = st.text_input("Host", value="localhost", label_visibility="collapsed")
        endee_port = st.number_input("Port", value=8080, min_value=1, max_value=65535, label_visibility="collapsed")
        
        if st.button("Verify Connection", use_container_width=True):
            try:
                from src.endee_client import EndeeClient
                client = EndeeClient(host=endee_host, port=int(endee_port))
                if client.connect():
                    st.success("Connection verified.")
                else:
                    st.warning("Connection could not be established.")
            except Exception as e:
                st.error(f"Connection failed: {e}")
        
        st.divider()
        
        # Model settings
        st.markdown("<div style='font-weight: 500; font-size: 14px; margin-bottom: 8px;'>Model Settings</div>", unsafe_allow_html=True)
        embedding_model = st.selectbox(
            "Embedding Model",
            options=[
                "all-MiniLM-L6-v2",
                "all-mpnet-base-v2",
                "paraphrase-MiniLM-L6-v2"
            ],
            index=0,
            label_visibility="collapsed"
        )
        
        llm_model = st.selectbox(
            "LLM Model for Generation",
            options=["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
            index=0,
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Data pipeline
        st.markdown("<div style='font-weight: 500; font-size: 14px; margin-bottom: 8px;'>Index Data Segment</div>", unsafe_allow_html=True)
        category = st.selectbox(
            "arXiv Segment",
            options=["cs.AI", "cs.LG", "cs.CL", "cs.CV", "cs.IR"],
            index=0,
            label_visibility="collapsed"
        )
        max_papers = st.slider("Document Limit", min_value=10, max_value=500, value=50)
        
        if st.button("Index Documents", use_container_width=True):
            try:
                from src.data_pipeline import DataPipeline
                pipeline = DataPipeline()
                with st.spinner(f"Ingesting top {max_papers} documents from {category}..."):
                    stats = pipeline.run(category=category, max_papers=max_papers)
                st.success(f"Successfully indexed {stats.get('vectors_indexed', 0)} vectors.")
                st.session_state.indexed_count = stats.get('vectors_indexed', 0)
            except Exception as e:
                st.error(f"Process failed: {e}")
        
        st.divider()
        
        # About section
        st.markdown("<h3 style='font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em; color: #64748b; margin-bottom: 16px;'>System Information</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 13px; color: #475569; line-height: 1.5;">
        Architecture relies on Endee for vector storage, Sentence-BERT for embedding generation, and Google Gemini for synthesis.
        </div>
        """, unsafe_allow_html=True)
        
        return {
            "endee_host": endee_host,
            "endee_port": int(endee_port),
            "embedding_model": embedding_model,
            "llm_model": llm_model
        }


def render_search_tab(config: dict):
    """Render the semantic search interface."""
    st.markdown("<div style='font-weight: 600; font-size: 20px; margin-bottom: 8px;'>Semantic Retrieval</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #64748b; font-size: 14px; margin-bottom: 24px;'>Retrieve documents by conceptual similarity rather than exact keyword matches.</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., methods to improve attention mechanism efficiency",
            label_visibility="collapsed"
        )
    with col2:
        top_k = st.selectbox("Results Count", options=[5, 10, 15, 20], index=1, label_visibility="collapsed")
    
    if st.button("Retrieve Documents", use_container_width=False) or query:
        if query:
            with st.spinner("Processing retrieval..."):
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
                        st.markdown(f"<div style='font-size: 14px; font-weight: 500; margin: 24px 0 16px 0;'>Retrieved {len(results)} relevant documents</div>", unsafe_allow_html=True)
                        for i, paper in enumerate(results, 1):
                            with st.container():
                                st.markdown(f"""
                                <div class="result-card">
                                    <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
                                        <div class="result-title">{paper.title}</div>
                                        <span class="score-badge">Match: {paper.score:.4f}</span>
                                    </div>
                                    <div class="result-meta">
                                        {', '.join(paper.authors[:3]) if paper.authors else 'Unknown'} 
                                        {'...' if paper.authors and len(paper.authors) > 3 else ''} 
                                        &bull; {', '.join(paper.categories[:2]) if paper.categories else 'Uncategorized'}
                                        {f" &bull; {paper.published[:10]}" if paper.published else ""}
                                    </div>
                                    <div class="result-abstract">{paper.abstract[:400]}...</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                if paper.arxiv_id:
                                    st.markdown(f"<a href='https://arxiv.org/abs/{paper.arxiv_id}' style='font-size: 13px; color: #3b82f6; text-decoration: none; font-weight: 500;'>View Source Document &rarr;</a>", unsafe_allow_html=True)
                                    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
                    else:
                        st.info("No matching documents found in the current index.")
                        
                except Exception as e:
                    st.error(f"Retrieval error: {e}")


def render_rag_tab(config: dict):
    """Render the document analysis tab."""
    st.markdown("<div style='font-weight: 600; font-size: 20px; margin-bottom: 8px;'>Document Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #64748b; font-size: 14px; margin-bottom: 24px;'>Synthesize answers grounded strictly in the indexed literature corpus.</div>", unsafe_allow_html=True)
    
    # Check for API key
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if not gemini_key:
        st.warning("System requires a Gemini API key. Configure GEMINI_API_KEY in the environment variables to proceed.")
        return
    
    question = st.text_area(
        "Analysis Prompt",
        placeholder="Enter your specific question regarding the literature...",
        height=120,
        label_visibility="collapsed"
    )
    
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("<div style='font-size: 14px; font-weight: 500; margin-bottom: 8px;'>Context Depth</div>", unsafe_allow_html=True)
        num_papers = st.slider("Source Documents", min_value=3, max_value=10, value=5, label_visibility="collapsed")
    
    if st.button("Synthesize Analysis", use_container_width=False):
        if not question:
            st.warning("Prompt requires input.")
        else:
            with st.spinner("Analyzing corpus and generating synthesis..."):
                try:
                    from src.rag_engine import RAGEngine
                    rag = RAGEngine(gemini_api_key=gemini_key, model=config.get("llm_model", "gemini-2.0-flash"))
                    response = rag.ask(question=question, num_papers=num_papers)
                    
                    # Display answer
                    st.markdown("""
                    <div class="answer-box">
                        <div class="answer-title">Synthesis Result</div>
                        <div style="font-size: 15px; line-height: 1.6; color: #1e293b;">
                    """, unsafe_allow_html=True)
                    st.markdown(response.answer)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                    
                    # Evaluation Metrics
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown("<div style='font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;'>Confidence Metric</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='font-size: 16px; font-weight: 500; color: #0f172a;'>{response.confidence:.1%}</div>", unsafe_allow_html=True)
                    
                    with col_b:
                        if response.tokens_used:
                            st.markdown("<div style='font-size: 12px; font-weight: 600; color: #64748b; text-transform: uppercase; margin-bottom: 4px;'>Compute Usage</div>", unsafe_allow_html=True)
                            st.markdown(f"<div style='font-size: 16px; font-weight: 500; color: #0f172a;'>{response.tokens_used} tokens</div>", unsafe_allow_html=True)
                    
                    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
                    
                    # Display sources
                    if response.sources:
                        st.markdown("<div style='font-size: 14px; font-weight: 600; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; margin-bottom: 16px;'>Referenced Documents</div>", unsafe_allow_html=True)
                        for source in response.sources:
                            st.markdown(f"""
                            <div style="font-size: 13px; color: #334155; margin-bottom: 8px;">
                                <span style="font-weight: 500; color: #0f172a;">{source['title']}</span> 
                                <span style="color: #94a3b8;">&nbsp;|&nbsp;</span> 
                                {', '.join(source.get('authors', [])[:2])}
                            </div>
                            """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Analysis process encountered an error: {e}")


def render_stats_tab():
    """Render database integrity and corpus statistics."""
    st.markdown("<div style='font-weight: 600; font-size: 20px; margin-bottom: 8px;'>Corpus Statistics</div>", unsafe_allow_html=True)
    st.markdown("<div style='color: #64748b; font-size: 14px; margin-bottom: 32px;'>Current state of the indexed vector database.</div>", unsafe_allow_html=True)
    
    try:
        from src.endee_client import get_endee_client
        client = get_endee_client()
        stats = client.get_collection_stats("research_papers")
        
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-card">
                <div class="stat-value">{stats.get("vector_count", 0):,}</div>
                <div class="stat-label">Total Vectors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats.get("dimension", 384)}</div>
                <div class="stat-label">Vector Dimension</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{str(stats.get("index_type", "HNSW")).upper()}</div>
                <div class="stat-label">Index Type</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.info("System unable to retrieve index statistics. Ensure Endee is running and the index exists.")
    
    # Search history log
    if st.session_state.search_history:
        st.markdown("<div style='font-weight: 600; font-size: 16px; margin: 32px 0 16px 0;'>Retrieval Log</div>", unsafe_allow_html=True)
        for item in st.session_state.search_history[-10:][::-1]:
            st.markdown(f"""
            <div style="display: flex; justify-content: space-between; padding: 12px; border-bottom: 1px solid #f1f5f9; font-size: 13px;">
                <div style="font-family: ui-monospace, monospace; color: #334155;">"{item['query']}"</div>
                <div style="color: #64748b;">{item['results_count']} results</div>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    init_session_state()
    render_header()
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Clean up streamlit spacing for tabs
    st.markdown("""
        <style>
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 48px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 0px 0px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            font-weight: 600 !important;
            color: #0f172a !important;
            border-bottom-color: #0f172a !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Main content tabs (removed emojis)
    tab1, tab2, tab3 = st.tabs(["Retrieval", "Analysis", "Metrics"])
    
    with tab1:
        render_search_tab(config)
    
    with tab2:
        render_rag_tab(config)
    
    with tab3:
        render_stats_tab()
    
    # Footer
    st.markdown("<div style='height: 48px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="border-top: 1px solid #e2e8f0; padding-top: 24px; text-align: left; color: #94a3b8; font-size: 12px; letter-spacing: 0.02em;">
            Endee Information Retrieval System &bull; Version 1.0.0
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
