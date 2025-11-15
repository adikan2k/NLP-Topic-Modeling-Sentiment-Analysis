# NLP Analysis: John Lewis Christmas Ad Sentiment and Topic Modeling

A project for Natural Language Processing course analyzing social media reactions to the 2025 John Lewis Christmas advertisement using traditional and transformer-based NLP techniques.

## Abstract

This project investigates audience sentiment and topic patterns in YouTube comments about the John Lewis 2025 Christmas advertisement. The ad's portrayal of masculinity and gender roles generated significant online discussion, making it an interesting case study for sentiment analysis and topic modeling. Due to platform API limitations, demographic inference was attempted using linguistic proxies rather than direct user data.

## Research Questions

1. What topics emerge in online discussions about the John Lewis Christmas ad?
2. How does sentiment distribution vary across different comment patterns?
3. Can linguistic features serve as reliable proxies for demographic analysis?
4. How do traditional TF-IDF/LDA methods compare with transformer-based BERTopic?
5. What emotional responses does the advertisement elicit in viewers?

## Background

The 2025 John Lewis Christmas advertisement sparked debate on social media platforms regarding its representation of masculinity and festive themes. This project applies NLP techniques learned in class to analyze real-world social media data, following methodologies discussed in lectures on sentiment analysis (Lecture 5) and topic modeling (Lecture 8).

## Methodology

### Data Collection
- YouTube comments scraped from official John Lewis videos using `youtube-comment-downloader`
- Rate limiting implemented to respect platform guidelines
- Comments filtered for relevance and minimum length

### Text Processing
- Social media-specific preprocessing (emoji handling, URL removal)
- Tokenization and lemmatization using NLTK
- Stopword removal with custom social media terms

### Topic Modeling Approaches
1. **Traditional**: TF-IDF vectorization + Latent Dirichlet Allocation (Gensim)
2. **Modern**: BERTopic with Sentence-BERT embeddings and HDBSCAN clustering

### Sentiment Analysis
- TextBlob for baseline sentiment scoring
- RoBERTa transformer model (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- Combined consensus approach for improved accuracy

### Demographic Proxy Analysis
*Note: Initial attempts to infer age groups from linguistic patterns revealed methodological limitations. The approach was revised to focus on linguistic style analysis rather than demographic claims.*

## Challenges and Limitations

### Technical Challenges
- YouTube API rate limiting required careful request management
- BERTopic parameter tuning for small dataset optimization
- Memory management with transformer models on local hardware

### Methodological Issues
- Demographic inference from text proved unreliable (see discussion in findings)
- Reddit API access limited due to authentication requirements
- LinkedIn data collection restricted by platform policies

### Ethical Considerations
- User privacy maintained through data anonymization
- Comments analyzed in aggregate without individual attribution
- Limitations of demographic proxy analysis clearly documented

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Run complete analysis pipeline
python -m src.pipeline

# Launch interactive dashboard
streamlit run src/dashboard/app.py
```

## Findings Summary

- **Dataset**: 876 YouTube comments from 2 official John Lewis videos
- **Sentiment**: 64% positive, 23% neutral, 14% negative
- **Topics**: 5 main themes identified through LDA, 2 clusters via BERTopic
- **Key Insight**: Linguistic style analysis shows correlation with sentiment but not reliable for demographic inference

## References

- Bird, S., Klein, E., & Loper, E. (2009). *Natural Language Processing with Python*. O'Reilly.
- Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." *EMNLP 2019*.
- Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a class-based TF-IDF procedure." *arXiv preprint*.
- Course materials: NLP Lectures

## Future Work

- Incorporate Reddit data with proper API authentication
- Explore multimodal analysis (video content + text)
- Compare with previous John Lewis ad campaigns
- Implement more sophisticated demographic inference methods

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager
- 8GB+ RAM recommended for transformer models
- GPU optional but recommended for faster processing

### Installation Steps

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd NLP_Topic-modelling-and-sentiment-analysis
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Mac/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data** (first time only):
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

## 🎮 Usage

### Quick Start - Run Complete Analysis

```python
from src.pipeline import AnalysisPipeline

# Initialize and run full pipeline
pipeline = AnalysisPipeline()
results = pipeline.run_full_pipeline()

# Get summary
summary = pipeline.get_results_summary()
print(summary)
```

### Step-by-Step Analysis

```python
# 1. Scrape YouTube comments
from src.scrapers.youtube_scraper import YouTubeCommentScraper

scraper = YouTubeCommentScraper()
comments_df = scraper.scrape_video_comments("YOUTUBE_VIDEO_URL")

# 2. Preprocess text
from src.analysis.preprocessing import TextPreprocessor

preprocessor = TextPreprocessor()
processed_df = preprocessor.preprocess_dataframe(comments_df)

# 3. Traditional topic modeling
from src.analysis.traditional_topic_modeling import TraditionalTopicModeler

modeler = TraditionalTopicModeler(n_topics=5)
sklearn_results = modeler.fit_sklearn_lda(processed_df['processed_text'])
gensim_results = modeler.fit_gensim_lda(processed_df['processed_text'])

# 4. BERTopic analysis
from src.analysis.bertopic_modeling import BERTopicModeler

bertopic_modeler = BERTopicModeler(n_topics=5)
bertopic_results = bertopic_modeler.fit_transform(processed_df['processed_text'].tolist())

# 5. Sentiment analysis
from src.analysis.sentiment_analysis import SentimentAnalyzer

analyzer = SentimentAnalyzer()
textblob_results = analyzer.analyze_textblob_sentiment(processed_df['sentiment_text'].tolist())
transformer_results = analyzer.analyze_transformer_sentiment(processed_df['sentiment_text'].tolist())
combined_results = analyzer.combine_sentiment_results(textblob_results, transformer_results)
```

### Launch Interactive Dashboard

```bash
# Navigate to project directory
cd NLP_Topic-modelling-and-sentiment-analysis

# Run Streamlit dashboard
streamlit run src/dashboard/app.py
```

The dashboard provides:
- 📊 Overview metrics and data quality indicators
- 💭 Topic comparison between traditional and transformer methods
- 😊 Sentiment distribution and analysis
- 👥 Demographic insights using linguistic proxies
- 📥 Export functionality for results

### Export Results

```python
from src.utils.export_utils import DataExporter

exporter = DataExporter()
exported_files = exporter.export_all_results(results)

# Export sample data for sharing
sample_file = exporter.create_sample_export(results, sample_size=100)
```

## ⚙️ Configuration

Customize analysis parameters in `src/config.py`:

```python
# YouTube scraping
YOUTUBE_CONFIG = {
    "max_comments_per_video": 1000,
    "rate_limit_delay": 0.1
}

# Topic modeling
TRADITIONAL_TOPIC_CONFIG = {
    "n_topics": 5,
    "max_features": 1000
}

BERTOPIC_CONFIG = {
    "embedding_model": "all-MiniLM-L6-v2",
    "n_topics": 5,
    "min_topic_size": 10
}

# Sentiment analysis
SENTIMENT_CONFIG = {
    "transformer_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "use_transformer": True
}
```

## 📊 Key Features

### 🔍 Data Collection
- YouTube comment scraping with metadata (likes, replies, timestamps)
- Rate limiting and error handling
- Support for multiple video URLs

### 🧠 Topic Modeling
- **Traditional**: TF-IDF vectorization + LDA (scikit-learn & Gensim)
- **Modern**: BERTopic with Sentence-BERT embeddings, UMAP, HDBSCAN
- Coherence evaluation and optimal topic number detection
- Interactive visualizations with pyLDAvis and word clouds

### 😊 Sentiment Analysis
- **Baseline**: TextBlob for polarity and subjectivity
- **Advanced**: RoBERTa transformer for nuanced classification
- **Emotion Detection**: Specialized model for emotional responses
- **Consensus Approach**: Combines multiple methods for robust results

### 👥 Demographic Insights
- **Linguistic Proxies**: Formality scoring, emoji usage analysis
- **Age Group Estimation**: Word choice and slang patterns
- **Expressiveness Metrics**: Emoji and punctuation usage
- **Limitations**: Explicitly documented proxy-based approach

### 📈 Visualization & Dashboard
- Interactive Streamlit dashboard with multiple tabs
- Real-time sentiment gauges and topic distributions
- Demographic comparison charts
- Export functionality for all visualizations

## 🔧 Advanced Usage

### Custom Analysis Pipeline

```python
# Run specific steps only
pipeline = AnalysisPipeline()

# Run only preprocessing and sentiment analysis
pipeline.run_step("scraping")
pipeline.run_step("preprocessing")
pipeline.run_step("sentiment_analysis")

# Resume from specific step if pipeline failed
results = pipeline.run_full_pipeline(resume_from="bertopic")
```

### Custom Topic Modeling Parameters

```python
# Find optimal number of topics
modeler = TraditionalTopicModeler()
coherence_scores = modeler.find_optimal_topics(texts, max_topics=10)

# Use custom embedding model for BERTopic
bertopic_modeler = BERTopicModeler(
    embedding_model="all-mpnet-base-v2",
    n_topics=8,
    min_topic_size=15
)
```

### Batch Processing Large Datasets

```python
# Process data in batches for memory efficiency
batch_size = 1000
all_results = []

for i in range(0, len(texts), batch_size):
    batch_texts = texts[i:i+batch_size]
    batch_results = analyzer.analyze_sentiment(batch_texts)
    all_results.append(batch_results)
```

## 📋 Dependencies

### Core Libraries
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Traditional ML algorithms
- `nltk>=3.8.0` - Natural language processing
- `gensim>=4.3.0` - Topic modeling utilities

### Modern NLP
- `transformers>=4.30.0` - Hugging Face transformers
- `torch>=2.0.0` - Deep learning framework
- `sentence-transformers>=2.2.0` - Sentence embeddings
- `bertopic>=0.15.0` - Transformer topic modeling

### Visualization & Dashboard
- `streamlit>=1.24.0` - Interactive dashboard
- `plotly>=5.14.0` - Interactive charts
- `matplotlib>=3.7.0` - Static visualizations
- `seaborn>=0.12.0` - Statistical plots
- `wordcloud>=1.9.0` - Word cloud generation
- `pyldavis>=3.4.0` - Interactive LDA visualization

### Data Collection & Processing
- `youtube-comment-downloader>=0.1.8` - YouTube scraping
- `beautifulsoup4>=4.12.0` - HTML parsing
- `requests>=2.31.0` - HTTP requests
- `textblob>=0.17.1` - Sentiment analysis

### Dimensionality Reduction & Clustering
- `umap-learn>=0.5.3` - UMAP for BERTopic
- `hdbscan>=0.8.29` - Clustering for BERTopic

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Memory Issues**:
   ```python
   # Disable GPU usage in config
   PERFORMANCE_CONFIG = {
       "use_gpu": False,
       "memory_limit_gb": 4
   }
   ```

2. **YouTube API Rate Limits**:
   ```python
   # Increase delay between requests
   YOUTUBE_CONFIG = {
       "rate_limit_delay": 0.5  # 500ms delay
   }
   ```

3. **Large Dataset Memory Issues**:
   ```python
   # Process in smaller batches
   pipeline = AnalysisPipeline()
   pipeline.run_step("scraping")
   # Process first 1000 comments only
   texts = results['preprocessing']['processed_text'].head(1000)
   ```

4. **Transformer Model Download Issues**:
   ```bash
   # Set Hugging Face cache directory
   export HF_HOME=/path/to/cache/directory
   ```

### Performance Optimization

- Use GPU for transformer models: `CUDA_VISIBLE_DEVICES=0 streamlit run src/dashboard/app.py`
- Reduce batch size for memory-constrained environments
- Disable transformer sentiment analysis for faster processing
- Use caching for repeated analyses

## 📚 Academic Context

This project addresses several NLP concepts from the course:

### Traditional Methods
- **TF-IDF Vectorization**: Term frequency-inverse document frequency for feature extraction
- **Latent Dirichlet Allocation**: Probabilistic topic modeling (Lecture 8)
- **Gensim Integration**: Efficient topic modeling with corpora and dictionaries (Folder 17)

### Modern Approaches
- **Transformer Models**: Contextual embeddings with BERT/RoBERTa
- **Sentence-BERT**: Semantic similarity for topic clustering
- **BERTopic**: Modern topic modeling pipeline combining multiple techniques

### Text Analysis
- **TextBlob**: Simple sentiment analysis and text processing (Folder 17)
- **Emotion Detection**: Fine-grained emotional classification
- **Linguistic Feature Analysis**: Demographic proxy development

## 📊 Expected Results

Based on preliminary analysis of similar content:

### Topic Themes
- Emotional storytelling and character development
- Christmas spirit and festive expectations
- Masculinity and gender role portrayals
- Advertising effectiveness and brand perception
- Family dynamics and relationships

### Sentiment Patterns
- Polarized responses to emotional tone
- Generational differences in ad perception
- Gender-based variations in character interpretation
- Cultural differences in Christmas advertising expectations

### Demographic Insights
- Linguistic formality correlates with sentiment intensity
- Emoji usage indicates emotional engagement
- Age-related language patterns affect topic preferences

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit pull request

## 📄 License

This project is for educational purposes as part of the NLP course assignment. Please refer to the LICENSE file for usage terms.

## 📞 Support

For questions or issues:
- Check the troubleshooting section above
- Review the configuration options in `src/config.py`
- Consult the lecture notes and code examples provided in the course materials

## 🔮 Future Enhancements

- **Multi-platform Data Collection**: Extend to Twitter, Reddit, Facebook comments
- **Real-time Analysis**: Live sentiment tracking during ad campaigns
- **Cross-cultural Comparison**: Analyze responses across different countries
- **Longitudinal Analysis**: Track sentiment changes over time
- **Advanced Demographics**: Integrate with demographic datasets for validation
- **Business Impact Metrics**: Correlate sentiment with sales/brand metrics

---

**Assignment**: Mini NLP Assignment - John Lewis Christmas Ad Analysis  
**Technologies**: Python, scikit-learn, Gensim, Transformers, Streamlit, BERTopic, TextBlob
