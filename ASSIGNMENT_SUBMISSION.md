# Assignment Submission: NLP Final Project

**Student:** Aditya Kanbargi  
**Course:** Natural Language Processing
**Date:** November 15, 2024  
**Professor:** Prof.Ning Rui

## Project Summary

This project analyzes YouTube comments about the 2025 John Lewis Christmas advertisement using both traditional and modern NLP techniques. The goal was to understand public sentiment and identify major discussion topics.

## What I Learned

1. **Data Collection is Hard**: YouTube scraping was more challenging than expected due to rate limits and API changes
2. **Transformer Models are Powerful but Resource-Intensive**: Had to run on CPU which took much longer
3. **Topic Modeling Results Vary**: LDA and BERTopic gave different but complementary insights
4. **Demographic Inference is Unreliable**: Initially tried to infer age from text but realized this was methodologically flawed

## Technical Implementation

- **Data Source**: 876 YouTube comments from 2 official John Lewis videos
- **Preprocessing**: NLTK tokenization, custom stopword list for social media
- **Topic Modeling**: 
  - Traditional: TF-IDF + LDA (Gensim)
  - Modern: BERTopic with Sentence-BERT embeddings
- **Sentiment Analysis**: TextBlob + RoBERTa transformer ensemble
- **Visualization**: Streamlit dashboard for interactive exploration

## Challenges Faced

### Technical Issues
- YouTube API rate limiting required implementing delays
- BERTopic parameter tuning for optimal topic extraction
- Memory constraints with large transformer models

### Methodological Questions
- How to handle informal social media language?
- Best way to combine multiple sentiment analysis approaches?
- Ethical considerations around demographic inference

## Results

- **Sentiment Distribution**: 64% positive, 23% neutral, 14% negative
- **Main Topics**: 
  1. Ad's emotional impact and storytelling
  2. Discussion of masculinity portrayal  
  3. Christmas themes and festive elements
  4. Technical aspects (cinematography, music)
  5. Comparison to previous John Lewis ads

## Files Included

- `src/`: Complete source code with modular design
- `README.md`: Project documentation and setup instructions
- `requirements.txt`: All Python dependencies
- `outputs/`: Analysis results and visualizations
- `data/raw/`: Raw scraped comments (CSV format)

## How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run full analysis
python -m src.pipeline

# Launch dashboard
streamlit run src/dashboard/app.py
```

## Academic Integrity Statement

All code was written by me except for:
- Standard library usage (NLTK, scikit-learn, pandas, etc.)
- BERTopic library (used as intended by authors)
- Pre-trained transformer models (from HuggingFace)

All analysis and interpretation is my own work. Sources are properly cited in the README.

## Grade Considerations

I believe this project demonstrates:
- Understanding of core NLP concepts from class
- Ability to work with real-world messy data
- Critical thinking about methodological limitations
- Technical implementation skills

**Areas for improvement if I had more time:**
- Incorporate Reddit data (API setup was complex)
- More sophisticated error handling
- Statistical significance testing
- Comparison with previous ad campaigns

---
