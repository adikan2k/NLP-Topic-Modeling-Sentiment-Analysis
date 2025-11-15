"""
Streamlit Dashboard for John Lewis Christmas Ad NLP Analysis.
Interactive visualization and exploration of analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pickle
import json
from pathlib import Path
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import AnalysisPipeline
from src.config import DASHBOARD_CONFIG, OUTPUT_DIR, DATA_DIR

# Set page config
st.set_page_config(
    page_title=DASHBOARD_CONFIG["title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state=DASHBOARD_CONFIG["initial_sidebar_state"]
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=DASHBOARD_CONFIG["cache_ttl"])
def load_pipeline_results():
    """Load cached pipeline results or run pipeline if needed."""
    cache_path = OUTPUT_DIR / "cache" / "dashboard_data.pkl"
    
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading cached data: {e}")
    
    # Run pipeline if no cache exists
    with st.spinner("Running analysis pipeline..."):
        pipeline = AnalysisPipeline()
        results = pipeline.run_full_pipeline()
        
        # Cache results
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
    
    return results

def create_sentiment_gauge(sentiment_score):
    """Create a gauge chart for sentiment score."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average Sentiment Score"},
        delta = {'reference': 0},
        gauge = {
            'axis': {'range': [-1, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, -0.33], 'color': "lightcoral"},
                {'range': [-0.33, 0.33], 'color': "lightgray"},
                {'range': [0.33, 1], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 0
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_topic_wordcloud(topic_words, title):
    """Create word cloud for topic visualization."""
    if not topic_words:
        return None
    
    # Convert topic words to frequency dictionary
    word_freq = {}
    for word in topic_words:
        if isinstance(word, tuple):
            word_freq[word[0]] = word[1]
        else:
            word_freq[word] = 1
    
    if not word_freq:
        return None
    
    wordcloud = WordCloud(
        width=400, height=300,
        background_color='white',
        max_words=50,
        colormap='viridis'
    ).generate_from_frequencies(word_freq)
    
    return wordcloud

def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">🎄 John Lewis Christmas Ad NLP Analysis</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-box">
    <strong>📊 About this Analysis:</strong> This dashboard provides comprehensive insights into audience reactions 
    to the John Lewis 2025 Christmas advertisement, analyzing sentiment, topics, and demographic patterns 
    using both traditional and transformer-based NLP techniques.
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown('<h2 class="sub-header">🎛️ Analysis Controls</h2>', 
                       unsafe_allow_html=True)
    
    # Data loading option
    data_option = st.sidebar.radio(
        "Data Source:",
        ["Load Cached Results", "Run New Analysis", "Upload CSV Data"]
    )
    
    if data_option == "Run New Analysis":
        if st.sidebar.button("🚀 Run Full Pipeline"):
            with st.spinner("Running complete analysis pipeline..."):
                pipeline = AnalysisPipeline()
                results = pipeline.run_full_pipeline()
                st.success("Analysis completed!")
                st.session_state.results = results
    elif data_option == "Upload CSV Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload comment data (CSV)", 
            type=['csv'],
            help="Upload a CSV file with 'text' column containing comments"
        )
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                if 'text' not in df.columns:
                    st.error("CSV must contain a 'text' column")
                else:
                    st.session_state.uploaded_data = df
                    st.success(f"Loaded {len(df)} comments")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Load results
    results = None
    if data_option == "Load Cached Results":
        results = load_pipeline_results()
    elif 'results' in st.session_state:
        results = st.session_state.results
    
    if results is None:
        st.warning("No analysis results available. Please load cached results or run new analysis.")
        return
    
    # Main navigation
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "💭 Topic Analysis", "😊 Sentiment Analysis", 
        "👥 Demographic Insights", "📥 Export & Details"
    ])
    
    with tab1:
        st.markdown('<h2 class="sub-header">📊 Analysis Overview</h2>', 
                   unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_comments = len(results.get('preprocessing', pd.DataFrame()))
            st.metric("Total Comments", f"{total_comments:,}")
        
        with col2:
            if 'traditional_topics' in results:
                traditional_count = len(results['traditional_topics']['topic_summary'])
                st.metric("Traditional Topics", traditional_count)
            else:
                st.metric("Traditional Topics", "N/A")
        
        with col3:
            if 'bertopic' in results:
                bertopic_count = len(results['bertopic']['topic_summary'])
                st.metric("BERTopic Topics", bertopic_count)
            else:
                st.metric("BERTopic Topics", "N/A")
        
        with col4:
            if 'sentiment_analysis' in results:
                sentiment_data = results['sentiment_analysis']['combined_results']
                positive_pct = (sentiment_data['consensus_sentiment'] == 'positive').mean() * 100
                st.metric("Positive Sentiment", f"{positive_pct:.1f}%")
            else:
                st.metric("Positive Sentiment", "N/A")
        
        # Sentiment gauge
        if 'sentiment_analysis' in results:
            sentiment_data = results['sentiment_analysis']['combined_results']
            avg_sentiment = sentiment_data['textblob_polarity'].mean()
            
            fig_sentiment = create_sentiment_gauge(avg_sentiment)
            st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Data quality metrics
        st.subheader("📋 Data Quality")
        if 'preprocessing' in results:
            processed_data = results['preprocessing']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                avg_length = processed_data['text_length'].mean()
                st.metric("Avg Comment Length", f"{avg_length:.1f} chars")
            
            with col2:
                avg_words = processed_data['word_count'].mean()
                st.metric("Avg Words per Comment", f"{avg_words:.1f}")
            
            with col3:
                emoji_pct = (processed_data['has_emoji']).mean() * 100
                st.metric("Comments with Emojis", f"{emoji_pct:.1f}%")
    
    with tab2:
        st.markdown('<h2 class="sub-header">💭 Topic Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Topic comparison
        st.subheader("🔍 Topic Modeling Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Traditional LDA Topics")
            if 'traditional_topics' in results:
                topic_summary = results['traditional_topics']['topic_summary']
                for _, row in topic_summary.iterrows():
                    st.markdown(f"**{row['Topic']}**: {row['Top_5_Words']}")
            else:
                st.info("Traditional topic modeling results not available")
        
        with col2:
            st.markdown("#### BERTopic Results")
            if 'bertopic' in results:
                bertopic_summary = results['bertopic']['topic_summary']
                for _, row in bertopic_summary.iterrows():
                    st.markdown(f"**{row['Topic_Name']}**: {row['Top_5_Words']}")
            else:
                st.info("BERTopic results not available")
        
        # Topic word clouds
        st.subheader("☁️ Topic Word Clouds")
        
        if 'traditional_topics' in results:
            traditional_modeler = results['traditional_topics']['modeler']
            
            # Create word clouds for top topics
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()
            
            topics = traditional_modeler.sklearn_topics
            for i, (topic_name, words) in enumerate(topics.items()):
                if i >= len(axes):
                    break
                
                wordcloud = create_topic_wordcloud(words, topic_name)
                if wordcloud:
                    axes[i].imshow(wordcloud, interpolation='bilinear')
                    axes[i].set_title(topic_name, fontsize=12, fontweight='bold')
                    axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(len(topics), len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Topic distribution
        if 'bertopic' in results:
            st.subheader("📊 Topic Distribution")
            
            bertopic_results = results['bertopic']['results']
            topic_counts = pd.Series(bertopic_results['topics']).value_counts().sort_index()
            
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                labels={'x': 'Topic ID', 'y': 'Number of Comments'},
                title='Distribution of Topics Across Comments'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">😊 Sentiment Analysis</h2>', 
                   unsafe_allow_html=True)
        
        if 'sentiment_analysis' in results:
            sentiment_data = results['sentiment_analysis']['combined_results']
            
            # Sentiment distribution
            st.subheader("📈 Sentiment Distribution")
            
            col1, col2 = st.columns(2)
            
            with col1:
                sentiment_counts = sentiment_data['consensus_sentiment'].value_counts()
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=sentiment_counts.index,
                    values=sentiment_counts.values,
                    hole=0.3,
                    marker_colors=['lightgreen', 'lightgray', 'lightcoral']
                )])
                fig_pie.update_layout(title="Sentiment Breakdown")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                # Polarity distribution
                fig_hist = px.histogram(
                    sentiment_data,
                    x='textblob_polarity',
                    nbins=30,
                    title="Sentiment Polarity Distribution",
                    labels={'textblob_polarity': 'Polarity Score'}
                )
                fig_hist.add_vline(x=0, line_dash="dash", line_color="red")
                st.plotly_chart(fig_hist, use_container_width=True)
            
            # Sentiment vs Subjectivity
            st.subheader("🔗 Sentiment vs Subjectivity")
            
            fig_scatter = px.scatter(
                sentiment_data,
                x='textblob_polarity',
                y='textblob_subjectivity',
                color='consensus_sentiment',
                title="Sentiment Polarity vs Subjectivity",
                labels={
                    'textblob_polarity': 'Polarity',
                    'textblob_subjectivity': 'Subjectivity'
                }
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Sample comments by sentiment
            st.subheader("💬 Sample Comments by Sentiment")
            
            selected_sentiment = st.selectbox(
                "Select sentiment to view sample comments:",
                sentiment_data['consensus_sentiment'].unique()
            )
            
            sample_comments = sentiment_data[
                sentiment_data['consensus_sentiment'] == selected_sentiment
            ].nlargest(5, 'sentiment_intensity')
            
            for _, row in sample_comments.iterrows():
                st.markdown(f"**Intensity:** {row['sentiment_intensity']:.2f}")
                st.markdown(f"*{row['text']}*")
                st.markdown("---")
        else:
            st.info("Sentiment analysis results not available")
    
    with tab4:
        st.markdown('<h2 class="sub-header">👥 Demographic Insights</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        <div class="insight-box">
        <strong>⚠️ Important Note:</strong> Due to YouTube API limitations, we cannot access actual demographic data. 
        The following analysis uses linguistic patterns as proxies for potential demographic differences.
        </div>
        """, unsafe_allow_html=True)
        
        if 'demographic_analysis' in results:
            demographic_data = results['demographic_analysis']['demographic_data']
            
            # Sentiment by linguistic age group
            st.subheader("🎂 Sentiment by Linguistic Age Group")
            
            age_sentiment = demographic_data.groupby(['age_group_proxy', 'consensus_sentiment']).size().reset_index(name='count')
            
            fig_age = px.bar(
                age_sentiment,
                x='age_group_proxy',
                y='count',
                color='consensus_sentiment',
                title="Sentiment Distribution by Linguistic Style (Age Proxy)",
                labels={'age_group_proxy': 'Linguistic Style', 'count': 'Number of Comments'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
            
            # Expressiveness analysis
            st.subheader("💭 Expressiveness vs Sentiment")
            
            fig_expr = px.box(
                demographic_data,
                x='expressiveness',
                y='textblob_polarity',
                color='consensus_sentiment',
                title="Sentiment by Expressiveness (Emoji Usage)",
                labels={'expressiveness': 'Emoji Usage Level', 'textblob_polarity': 'Sentiment Polarity'}
            )
            st.plotly_chart(fig_expr, use_container_width=True)
            
            # Formality vs sentiment
            st.subheader("📝 Formality vs Sentiment")
            
            fig_formality = px.scatter(
                demographic_data.sample(min(500, len(demographic_data))),  # Sample for performance
                x='formality_score',
                y='textblob_polarity',
                color='consensus_sentiment',
                title="Formality Score vs Sentiment Polarity",
                labels={
                    'formality_score': 'Formality Score',
                    'textblob_polarity': 'Sentiment Polarity'
                }
            )
            st.plotly_chart(fig_formality, use_container_width=True)
        else:
            st.info("Demographic analysis results not available")
    
    with tab5:
        st.markdown('<h2 class="sub-header">📥 Export & Detailed Results</h2>', 
                   unsafe_allow_html=True)
        
        st.subheader("📊 Download Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'preprocessing' in results:
                processed_data = results['preprocessing']
                csv = processed_data.to_csv(index=False)
                st.download_button(
                    label="📄 Download Processed Data",
                    data=csv,
                    file_name="processed_comments.csv",
                    mime="text/csv"
                )
        
        with col2:
            if 'sentiment_analysis' in results:
                sentiment_data = results['sentiment_analysis']['combined_results']
                csv = sentiment_data.to_csv(index=False)
                st.download_button(
                    label="💭 Download Sentiment Results",
                    data=csv,
                    file_name="sentiment_analysis.csv",
                    mime="text/csv"
                )
        
        with col3:
            if 'traditional_topics' in results:
                topic_summary = results['traditional_topics']['topic_summary']
                csv = topic_summary.to_csv(index=False)
                st.download_button(
                    label="🏷️ Download Topic Results",
                    data=csv,
                    file_name="topic_analysis.csv",
                    mime="text/csv"
                )
        
        # Detailed statistics
        st.subheader("📈 Detailed Statistics")
        
        if 'sentiment_analysis' in results:
            sentiment_summary = results['sentiment_analysis']['sentiment_summary']
            st.dataframe(sentiment_summary, use_container_width=True)
        
        # Pipeline status
        st.subheader("🔄 Pipeline Status")
        
        if hasattr(load_pipeline_results, '__globals__') and 'pipeline' in load_pipeline_results.__globals__:
            pipeline = load_pipeline_results.__globals__['pipeline']
            if hasattr(pipeline, 'get_results_summary'):
                summary = pipeline.get_results_summary()
                
                st.json(summary)
        
        # Methodology information
        st.subheader("🔬 Methodology")
        
        with st.expander("Analysis Methods"):
            st.markdown("""
            **Traditional Topic Modeling:**
            - TF-IDF vectorization with n-grams (1,2)
            - Latent Dirichlet Allocation (LDA)
            - Coherence evaluation using c_v measure
            
            **Transformer-based Topic Modeling:**
            - Sentence-BERT embeddings (all-MiniLM-L6-v2)
            - UMAP dimensionality reduction
            - HDBSCAN clustering
            - c-TF-IDF topic representation
            
            **Sentiment Analysis:**
            - TextBlob for baseline sentiment and subjectivity
            - RoBERTa transformer model for enhanced classification
            - Consensus approach combining both methods
            
            **Demographic Proxies:**
            - Formality score based on word length and slang usage
            - Emoji usage as expressiveness indicator
            - Linguistic patterns as age group proxies
            """)

if __name__ == "__main__":
    main()
