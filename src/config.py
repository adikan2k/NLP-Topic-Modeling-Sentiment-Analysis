"""
Configuration file for John Lewis Christmas Ad NLP Analysis Project.
Contains all model parameters, file paths, and settings.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Create directories if they don't exist
for dir_path in [DATA_DIR, OUTPUT_DIR, NOTEBOOKS_DIR]:
    dir_path.mkdir(exist_ok=True)

# YouTube scraping configuration
YOUTUBE_CONFIG = {
    "max_comments_per_video": 1000,
    "rate_limit_delay": 0.1,
    "output_filename": "youtube_comments.csv"
}

# Sample YouTube video URLs for John Lewis Christmas ads
JOHN_LEWIS_VIDEOS = [
    # Add actual John Lewis 2025 Christmas ad URLs here
    "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID_1",
    "https://www.youtube.com/watch?v=EXAMPLE_VIDEO_ID_2",
]

# Text preprocessing configuration
PREPROCESSING_CONFIG = {
    "min_text_length": 10,
    "max_text_length": 1000,
    "remove_urls": True,
    "remove_mentions": True,
    "remove_hashtags": True,
    "remove_html_tags": True,
    "normalize_case": True,
    "remove_punctuation": True,
    "remove_numbers": True,
    "lemmatize": True,
    "remove_stopwords": True
}

# Traditional topic modeling configuration
TRADITIONAL_TOPIC_CONFIG = {
    "n_topics": 5,
    "max_features": 1000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.8,
    "random_state": 42,
    "max_iter": 10
}

# BERTopic configuration
BERTOPIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "n_topics": 3,
    "min_topic_size": 2,
    "umap_n_neighbors": 3,
    "umap_n_components": 2,
    "umap_min_dist": 0.0,
    "umap_metric": "cosine",
    "hdbscan_min_cluster_size": 2,
    "hdbscan_metric": "euclidean",
    "hdbscan_cluster_selection_method": "eom",
    "vectorizer_ngram_range": (1, 2),
    "vectorizer_min_df": 1,
    "vectorizer_max_features": 100
}

# Sentiment analysis configuration
SENTIMENT_CONFIG = {
    "transformer_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "emotion_model": "j-hartmann/emotion-english-distilroberta-base",
    "use_transformer": True,
    "batch_size": 32,
    "polarity_thresholds": {
        "positive": 0.1,
        "negative": -0.1
    }
}

# Visualization configuration
VISUALIZATION_CONFIG = {
    "figure_size": (12, 8),
    "dpi": 300,
    "color_palette": "viridis",
    "save_plots": True,
    "plot_format": "png",
    "interactive_format": "html"
}

# Streamlit dashboard configuration
DASHBOARD_CONFIG = {
    "title": "John Lewis Christmas Ad - NLP Analysis Dashboard",
    "layout": "wide",
    "page_icon": "🎄",
    "initial_sidebar_state": "expanded",
    "max_upload_size": 200,
    "cache_ttl": 3600
}

# Export configuration
EXPORT_CONFIG = {
    "output_format": "csv",
    "include_metadata": True,
    "encoding": "utf-8",
    "date_format": "%Y-%m-%d %H:%M:%S"
}

# Analysis pipeline configuration
PIPELINE_CONFIG = {
    "run_scraping": True,
    "run_preprocessing": True,
    "run_traditional_topics": True,
    "run_bertopic": True,
    "run_sentiment_analysis": True,
    "run_emotion_analysis": True,
    "run_demographic_analysis": True,
    "generate_visualizations": True,
    "export_results": True
}

# Model evaluation configuration
EVALUATION_CONFIG = {
    "coherence_measures": ["c_v", "u_mass", "c_uci", "c_npmi"],
    "max_topics_to_test": 10,
    "min_topics_to_test": 2,
    "cross_validation_folds": 5
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": OUTPUT_DIR / "analysis.log"
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "use_gpu": True,
    "batch_processing": True,
    "parallel_workers": -1,  # Use all available cores
    "memory_limit_gb": 8
}

# Data validation configuration
VALIDATION_CONFIG = {
    "required_columns": ["text", "author", "timestamp"],
    "text_min_length": 3,
    "text_max_length": 5000,
    "remove_duplicates": True,
    "remove_empty_comments": True
}

# API keys (add your own if needed)
API_KEYS = {
    "youtube_api_key": os.getenv("YOUTUBE_API_KEY", ""),
    "twitter_api_key": os.getenv("TWITTER_API_KEY", ""),
    "reddit_api_key": os.getenv("REDDIT_API_KEY", "")
}

# Study-specific configuration for John Lewis ad analysis
STUDY_CONFIG = {
    "research_questions": [
        "What are the main topics discussed in comments about the John Lewis Christmas ad?",
        "How does sentiment vary across different demographic groups (using linguistic proxies)?",
        "What emotional responses does the ad elicit?",
        "How do traditional vs transformer topic modeling approaches compare?",
        "What patterns emerge in perceptions of masculinity and gender roles?"
    ],
    "hypotheses": [
        "Comments will show polarized sentiment regarding the ad's emotional tone",
        "Traditional topic modeling will identify broader themes, while BERTopic will find more nuanced topics",
        "Linguistic proxies will reveal different sentiment patterns across age groups"
    ],
    "keywords_of_interest": [
        "masculinity", "feminine", "father", "emotional", "christmas", "depressing",
        "festive", "identity", "crisis", "advertisement", "storytelling"
    ]
}

# Print configuration summary
def print_config_summary():
    """Print a summary of the current configuration."""
    print("=" * 60)
    print("JOHN LEWIS CHRISTMAS AD NLP ANALYSIS - CONFIGURATION")
    print("=" * 60)
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Max Comments per Video: {YOUTUBE_CONFIG['max_comments_per_video']}")
    print(f"Number of Topics: {TRADITIONAL_TOPIC_CONFIG['n_topics']}")
    print(f"Transformer Model: {SENTIMENT_CONFIG['transformer_model']}")
    print(f"BERTopic Model: {BERTOPIC_CONFIG['embedding_model']}")
    print("=" * 60)

if __name__ == "__main__":
    print_config_summary()
