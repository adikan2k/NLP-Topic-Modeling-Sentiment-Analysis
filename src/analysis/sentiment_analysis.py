import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    """
    Comprehensive sentiment analysis using TextBlob and transformer models.
    Combines traditional and modern approaches for robust sentiment classification.
    """
    
    def __init__(self, 
                 transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                 use_transformer: bool = True):
        """
        Initialize sentiment analyzer.
        
        Args:
            transformer_model: Hugging Face model for sentiment analysis
            use_transformer: Whether to use transformer models (can be disabled for faster processing)
        """
        self.transformer_model = transformer_model
        self.use_transformer = use_transformer
        
        # Initialize TextBlob (always available)
        self.textblob_enabled = True
        
        # Initialize transformer pipeline if requested
        self.transformer_pipeline = None
        if self.use_transformer:
            try:
                self.transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=transformer_model,
                    tokenizer=transformer_model,
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=True
                )
                print(f"Transformer model loaded: {transformer_model}")
            except Exception as e:
                print(f"Warning: Could not load transformer model: {e}")
                self.use_transformer = False
    
    def analyze_textblob_sentiment(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using TextBlob.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with TextBlob sentiment results
        """
        print("Analyzing sentiment with TextBlob...")
        
        results = []
        for text in texts:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Categorize sentiment
                if polarity > 0.1:
                    sentiment_label = 'positive'
                elif polarity < -0.1:
                    sentiment_label = 'negative'
                else:
                    sentiment_label = 'neutral'
                
                results.append({
                    'text': text,
                    'textblob_polarity': polarity,
                    'textblob_subjectivity': subjectivity,
                    'textblob_sentiment': sentiment_label
                })
            except Exception as e:
                print(f"Error analyzing text with TextBlob: {e}")
                results.append({
                    'text': text,
                    'textblob_polarity': 0.0,
                    'textblob_subjectivity': 0.0,
                    'textblob_sentiment': 'neutral'
                })
        
        return pd.DataFrame(results)
    
    def analyze_transformer_sentiment(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze sentiment using transformer models.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with transformer sentiment results
        """
        if not self.use_transformer or self.transformer_pipeline is None:
            print("Transformer model not available. Skipping transformer analysis.")
            return pd.DataFrame()
        
        print("Analyzing sentiment with transformer model...")
        
        results = []
        batch_size = 32  # Process in batches for efficiency
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_results = self.transformer_pipeline(batch_texts)
                
                for j, text in enumerate(batch_texts):
                    scores = batch_results[j]
                    
                    # Find the sentiment with highest score
                    best_sentiment = max(scores, key=lambda x: x['score'])
                    
                    # Extract individual scores
                    score_dict = {item['label'].lower(): item['score'] for item in scores}
                    
                    results.append({
                        'text': text,
                        'transformer_sentiment': best_sentiment['label'].lower(),
                        'transformer_confidence': best_sentiment['score'],
                        'transformer_positive': score_dict.get('positive', 0.0),
                        'transformer_negative': score_dict.get('negative', 0.0),
                        'transformer_neutral': score_dict.get('neutral', 0.0)
                    })
                    
            except Exception as e:
                print(f"Error analyzing batch with transformer: {e}")
                # Add neutral predictions for failed batch
                for text in batch_texts:
                    results.append({
                        'text': text,
                        'transformer_sentiment': 'neutral',
                        'transformer_confidence': 0.0,
                        'transformer_positive': 0.0,
                        'transformer_negative': 0.0,
                        'transformer_neutral': 1.0
                    })
        
        return pd.DataFrame(results)
    
    def analyze_emotions(self, texts: List[str]) -> pd.DataFrame:
        """
        Analyze emotions using a specialized emotion detection model.
        
        Args:
            texts: List of text strings
            
        Returns:
            DataFrame with emotion analysis results
        """
        print("Analyzing emotions with transformer model...")
        
        try:
            # Use emotion detection model
            emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1,
                return_all_scores=True
            )
            
            results = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                try:
                    batch_results = emotion_pipeline(batch_texts)
                    
                    for j, text in enumerate(batch_texts):
                        scores = batch_results[j]
                        score_dict = {item['label'].lower(): item['score'] for item in scores}
                        
                        # Find dominant emotion
                        best_emotion = max(scores, key=lambda x: x['score'])
                        
                        results.append({
                            'text': text,
                            'dominant_emotion': best_emotion['label'].lower(),
                            'emotion_confidence': best_emotion['score'],
                            'anger': score_dict.get('anger', 0.0),
                            'joy': score_dict.get('joy', 0.0),
                            'sadness': score_dict.get('sadness', 0.0),
                            'fear': score_dict.get('fear', 0.0),
                            'surprise': score_dict.get('surprise', 0.0),
                            'disgust': score_dict.get('disgust', 0.0),
                            'neutral': score_dict.get('neutral', 0.0)
                        })
                        
                except Exception as e:
                    print(f"Error analyzing emotions in batch: {e}")
                    for text in batch_texts:
                        results.append({
                            'text': text,
                            'dominant_emotion': 'neutral',
                            'emotion_confidence': 0.0,
                            'anger': 0.0, 'joy': 0.0, 'sadness': 0.0,
                            'fear': 0.0, 'surprise': 0.0, 'disgust': 0.0, 'neutral': 1.0
                        })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Could not load emotion model: {e}")
            return pd.DataFrame()
    
    def combine_sentiment_results(self, textblob_results: pd.DataFrame, 
                                 transformer_results: pd.DataFrame) -> pd.DataFrame:
        """
        Combine results from TextBlob and transformer models.
        
        Args:
            textblob_results: Results from TextBlob analysis
            transformer_results: Results from transformer analysis
            
        Returns:
            Combined DataFrame with consensus sentiment
        """
        if transformer_results.empty:
            return textblob_results
        
        # Merge results
        combined = pd.merge(textblob_results, transformer_results, on='text', how='left')
        
        # Create consensus sentiment
        def get_consensus_sentiment(row):
            textblob_sent = row['textblob_sentiment']
            transformer_sent = row['transformer_sentiment']
            
            # If both agree, use that sentiment
            if textblob_sent == transformer_sent:
                return textblob_sent
            
            # If transformer confidence is high, use transformer
            if row['transformer_confidence'] > 0.8:
                return transformer_sent
            
            # Otherwise, use TextBlob (more conservative)
            return textblob_sent
        
        combined['consensus_sentiment'] = combined.apply(get_consensus_sentiment, axis=1)
        
        # Add sentiment intensity score
        combined['sentiment_intensity'] = combined.apply(lambda row: (
            abs(row['textblob_polarity']) + row['transformer_confidence']
        ) / 2, axis=1)
        
        return combined
    
    def analyze_sentiment_by_demographics(self, df: pd.DataFrame, 
                                         text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment patterns by linguistic demographics (age/gender proxies).
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
            
        Returns:
            DataFrame with demographic sentiment analysis
        """
        print("Analyzing sentiment patterns by linguistic demographics...")
        
        # Create linguistic feature-based demographic proxies
        df_copy = df.copy()
        
        # Text formality (proxy for age)
        def calculate_formality(text):
            words = text.split()
            if not words:
                return 0
            
            # Count formal indicators
            formal_indicators = len([word for word in words if len(word) > 6])
            slang_indicators = len([word for word in words 
                                   if word in ['lol', 'omg', 'wow', 'yeah', 'awesome', 'cool']])
            
            return (formal_indicators - slang_indicators) / len(words)
        
        df_copy['formality_score'] = df_copy[text_column].apply(calculate_formality)
        
        # Emoji usage (proxy for expressiveness/age)
        def count_emojis(text):
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags
                "]+",
                flags=re.UNICODE
            )
            return len(emoji_pattern.findall(text))
        
        df_copy['emoji_count'] = df_copy[text_column].apply(count_emojis)
        
        # Categorize by formality (age proxy)
        df_copy['age_group_proxy'] = pd.cut(
            df_copy['formality_score'],
            bins=[-np.inf, -0.1, 0.1, np.inf],
            labels=['younger_style', 'neutral_style', 'older_style']
        )
        
        # Categorize by emoji usage (expressiveness proxy)
        df_copy['expressiveness'] = pd.cut(
            df_copy['emoji_count'],
            bins=[-1, 0, 2, np.inf],
            labels=['low', 'medium', 'high']
        )
        
        return df_copy
    
    def create_sentiment_visualizations(self, df: pd.DataFrame, save_path: str = None):
        """
        Create comprehensive sentiment visualizations.
        
        Args:
            df: DataFrame with sentiment results
            save_path: Optional path to save plots
        """
        # Sentiment distribution
        if 'consensus_sentiment' in df.columns:
            sentiment_counts = df['consensus_sentiment'].value_counts()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=sentiment_counts.index,
                    y=sentiment_counts.values,
                    text=sentiment_counts.values,
                    textposition='auto',
                    marker_color=['green', 'gray', 'red']
                )
            ])
            
            fig.update_layout(
                title='Overall Sentiment Distribution',
                xaxis_title='Sentiment',
                yaxis_title='Count'
            )
            
            if save_path:
                fig.write_html(f"{save_path}/sentiment_distribution.html")
            
            fig.show()
        
        # Sentiment polarity distribution
        if 'textblob_polarity' in df.columns:
            fig = px.histogram(
                df, 
                x='textblob_polarity',
                nbins=50,
                title='Sentiment Polarity Distribution',
                labels={'textblob_polarity': 'Polarity Score'}
            )
            
            fig.add_vline(x=0, line_dash="dash", line_color="red")
            fig.update_layout(showlegend=False)
            
            if save_path:
                fig.write_html(f"{save_path}/polarity_distribution.html")
            
            fig.show()
        
        # Demographic sentiment analysis
        if 'age_group_proxy' in df.columns and 'consensus_sentiment' in df.columns:
            demographic_sentiment = df.groupby(['age_group_proxy', 'consensus_sentiment']).size().reset_index(name='count')
            
            fig = px.bar(
                demographic_sentiment,
                x='age_group_proxy',
                y='count',
                color='consensus_sentiment',
                title='Sentiment by Linguistic Age Group',
                labels={'age_group_proxy': 'Age Group (Proxy)', 'count': 'Number of Comments'}
            )
            
            if save_path:
                fig.write_html(f"{save_path}/demographic_sentiment.html")
            
            fig.show()
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive sentiment summary statistics.
        
        Args:
            df: DataFrame with sentiment results
            
        Returns:
            Summary DataFrame
        """
        summary_data = []
        
        # Overall sentiment distribution
        if 'consensus_sentiment' in df.columns:
            sentiment_dist = df['consensus_sentiment'].value_counts(normalize=True)
            summary_data.append({
                'Metric': 'Overall Sentiment Distribution',
                'Positive': f"{sentiment_dist.get('positive', 0):.1%}",
                'Neutral': f"{sentiment_dist.get('neutral', 0):.1%}",
                'Negative': f"{sentiment_dist.get('negative', 0):.1%}"
            })
        
        # Average polarity and subjectivity
        if 'textblob_polarity' in df.columns:
            summary_data.append({
                'Metric': 'Average Polarity',
                'Positive': f"{df['textblob_polarity'].mean():.3f}",
                'Neutral': 'N/A',
                'Negative': 'N/A'
            })
            
            summary_data.append({
                'Metric': 'Average Subjectivity',
                'Positive': f"{df['textblob_subjectivity'].mean():.3f}",
                'Neutral': 'N/A',
                'Negative': 'N/A'
            })
        
        # Transformer confidence
        if 'transformer_confidence' in df.columns:
            summary_data.append({
                'Metric': 'Average Transformer Confidence',
                'Positive': f"{df['transformer_confidence'].mean():.3f}",
                'Neutral': 'N/A',
                'Negative': 'N/A'
            })
        
        return pd.DataFrame(summary_data)
    
    def analyze_sentiment_over_time(self, df: pd.DataFrame, 
                                   timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Analyze sentiment trends over time.
        
        Args:
            df: DataFrame with sentiment and timestamp data
            timestamp_column: Name of timestamp column
            
        Returns:
            DataFrame with temporal sentiment analysis
        """
        if timestamp_column not in df.columns or 'consensus_sentiment' not in df.columns:
            print("Required columns not found for temporal analysis")
            return pd.DataFrame()
        
        # Convert timestamp to datetime if needed
        df_copy = df.copy()
        df_copy[timestamp_column] = pd.to_datetime(df_copy[timestamp_column])
        
        # Group by date
        df_copy['date'] = df_copy[timestamp_column].dt.date
        
        temporal_sentiment = df_copy.groupby(['date', 'consensus_sentiment']).size().reset_index(name='count')
        
        # Calculate daily sentiment proportions
        daily_totals = temporal_sentiment.groupby('date')['count'].transform('sum')
        temporal_sentiment['proportion'] = temporal_sentiment['count'] / daily_totals
        
        return temporal_sentiment

def main():
    """
    Example usage of sentiment analysis.
    """
    # Sample data (simulating John Lewis ad comments)
    sample_texts = [
        "This John Lewis ad is absolutely amazing! 😍 So emotional and touching!",
        "I don't know... this feels a bit depressing for Christmas. Not festive at all.",
        "Love the storytelling and the emotional journey of the father character.",
        "The masculinity crisis portrayed here really speaks to modern struggles.",
        "Great Christmas ad! It brings tears to my eyes every time I watch it.",
        "The father seems so lost and confused throughout the entire ad.",
        "Excellent cinematography and music choice. Very professional.",
        "This ad perfectly captures what middle-aged men are feeling today.",
        "Where's the Christmas spirit? This is too dark for a holiday ad.",
        "Powerful narrative about family dynamics and emotional vulnerability."
    ]
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(use_transformer=True)
    
    # Analyze sentiment
    textblob_results = analyzer.analyze_textblob_sentiment(sample_texts)
    transformer_results = analyzer.analyze_transformer_sentiment(sample_texts)
    emotion_results = analyzer.analyze_emotions(sample_texts)
    
    # Combine results
    combined_results = analyzer.combine_sentiment_results(textblob_results, transformer_results)
    
    # Add demographic analysis
    demographic_results = analyzer.analyze_sentiment_by_demographics(combined_results)
    
    # Display results
    print("Sentiment Analysis Summary:")
    print(analyzer.get_sentiment_summary(combined_results))
    
    print("\nCombined Results:")
    print(combined_results[['text', 'consensus_sentiment', 'sentiment_intensity']].head())
    
    if not emotion_results.empty:
        print("\nEmotion Analysis:")
        print(emotion_results[['text', 'dominant_emotion', 'emotion_confidence']].head())
    
    # Create visualizations
    analyzer.create_sentiment_visualizations(demographic_results)

if __name__ == "__main__":
    main()
