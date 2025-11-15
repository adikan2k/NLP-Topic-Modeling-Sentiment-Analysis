import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
from hdbscan import HDBSCAN
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class BERTopicModeler:
    """
    Transformer-based topic modeling using BERTopic.
    Modern approach that combines sentence transformers, UMAP, HDBSCAN, and c-TF-IDF.
    """
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 n_topics: int = 5,
                 min_topic_size: int = 10):
        """
        Initialize BERTopic modeler.
        
        Args:
            embedding_model: Sentence transformer model name
            n_topics: Number of topics to extract
            min_topic_size: Minimum size of topics to consider
        """
        self.embedding_model = embedding_model
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        
        # Initialize components
        self.topic_model = None
        self.embeddings = None
        self.topics = None
        self.probs = None
        
        # Custom vectorizer for better topic representation
        self.vectorizer_model = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=1000
        )
        
        # UMAP for dimensionality reduction
        self.umap_model = UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )
        
        # HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
    
    def fit_transform(self, texts: List[str]) -> Dict:
        """
        Fit BERTopic model and transform texts to topics.
        
        Args:
            texts: List of preprocessed text documents
            
        Returns:
            Dictionary with model results
        """
        print("Training BERTopic model...")
        print(f"Processing {len(texts)} documents...")
        
        # Initialize BERTopic with custom components
        self.topic_model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            vectorizer_model=self.vectorizer_model,
            nr_topics=self.n_topics,
            verbose=True,
            calculate_probabilities=True
        )
        
        # Fit the model and transform documents
        self.topics, self.probs = self.topic_model.fit_transform(texts)
        
        # Get embeddings for visualization
        self.embeddings = self.topic_model._extract_embeddings(texts)
        
        print(f"Found {len(set(self.topics)) - 1} topics (excluding outlier topic)")
        
        return {
            'topics': self.topics,
            'probabilities': self.probs,
            'topic_model': self.topic_model,
            'embeddings': self.embeddings
        }
    
    def get_topic_info(self) -> pd.DataFrame:
        """
        Get detailed information about topics.
        
        Returns:
            DataFrame with topic information
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        topic_info = self.topic_model.get_topic_info()
        
        # Add custom metrics and handle column name variations
        topic_info['topic_name'] = topic_info['Name'].apply(self._generate_topic_name)
        
        # Handle different BERTopic versions - check available columns
        available_columns = topic_info.columns.tolist()
        
        # Add frequency/count column if it doesn't exist
        if 'Frequency' not in available_columns and 'Count' in available_columns:
            topic_info['Frequency'] = topic_info['Count']
        elif 'Frequency' not in available_columns and 'Count' not in available_columns:
            # Create frequency count from topic assignments
            if hasattr(self, 'topics') and self.topics is not None:
                topic_counts = pd.Series(self.topics).value_counts().reset_index()
                topic_counts.columns = ['Topic', 'Frequency']
                topic_info = topic_info.merge(topic_counts, on='Topic', how='left')
                topic_info['Frequency'] = topic_info['Frequency'].fillna(0)
        
        return topic_info
    
    def _generate_topic_name(self, topic_name: str) -> str:
        """Generate human-readable topic names."""
        if topic_name == "-1":
            return "Outlier/Noise"
        
        # Extract top words and create a meaningful name
        words = topic_name.split("_")[1:]
        if len(words) >= 2:
            return f"{'_'.join(words[:2])}"
        elif len(words) == 1:
            return words[0]
        else:
            return "Uncategorized"
    
    def get_topic_words(self, topic_id: int, top_n: int = 10) -> List[str]:
        """
        Get top words for a specific topic.
        
        Args:
            topic_id: Topic ID
            top_n: Number of top words to return
            
        Returns:
            List of top words
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        return [word for word, _ in self.topic_model.get_topic(topic_id)[:top_n]]
    
    def find_similar_topics(self, topic_id: int, top_n: int = 5) -> pd.DataFrame:
        """
        Find topics similar to a given topic.
        
        Args:
            topic_id: Topic ID to find similar topics for
            top_n: Number of similar topics to return
            
        Returns:
            DataFrame with similar topics
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        similar_topics, similarity = self.topic_model.find_topics(
            self.get_topic_words(topic_id, top_n=5), 
            top_n=top_n + 1
        )
        
        # Remove the topic itself from results
        results = []
        for i, (topic, score) in enumerate(zip(similar_topics, similarity)):
            if topic != topic_id:
                results.append({
                    'Topic_ID': topic,
                    'Similarity_Score': score,
                    'Top_Words': ', '.join(self.get_topic_words(topic, top_n=5))
                })
        
        return pd.DataFrame(results)
    
    def visualize_topics_2d(self, save_path: str = None):
        """
        Create 2D visualization of topics using UMAP projections.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        # Reduce embeddings to 2D for visualization
        umap_2d = UMAP(n_components=2, random_state=42)
        embeddings_2d = umap_2d.fit_transform(self.embeddings)
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'topic': self.topics,
            'topic_name': [self._generate_topic_name(f"Topic_{t}") for t in self.topics]
        })
        
        # Create interactive plot
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color='topic_name',
            hover_data=['topic'],
            title='BERTopic 2D Visualization',
            labels={
                'x': 'UMAP Dimension 1',
                'y': 'UMAP Dimension 2',
                'topic_name': 'Topic'
            }
        )
        
        fig.update_layout(
            width=800,
            height=600,
            showlegend=True
        )
        
        if save_path:
            fig.write_html(f"{save_path}/bertopic_2d.html")
        
        fig.show()
    
    def visualize_topic_hierarchy(self, save_path: str = None):
        """
        Visualize topic hierarchy using dendrogram.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        # Create hierarchical clustering visualization
        fig = self.topic_model.visualize_hierarchy()
        
        if save_path:
            fig.write_html(f"{save_path}/topic_hierarchy.html")
        
        fig.show()
    
    def visualize_topic_distribution(self, texts: List[str], save_path: str = None):
        """
        Visualize topic distribution across documents.
        
        Args:
            texts: Original texts
            save_path: Optional path to save the plot
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        # Create topic distribution bar chart
        topic_counts = pd.Series(self.topics).value_counts().sort_index()
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_counts.index,
                y=topic_counts.values,
                text=topic_counts.values,
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Topic Distribution Across Documents',
            xaxis_title='Topic ID',
            yaxis_title='Number of Documents',
            showlegend=False
        )
        
        if save_path:
            fig.write_html(f"{save_path}/topic_distribution.html")
        
        fig.show()
    
    def analyze_topic_sentiment(self, texts: List[str], sentiment_scores: List[float]) -> pd.DataFrame:
        """
        Analyze sentiment distribution across topics.
        
        Args:
            texts: Original texts
            sentiment_scores: Sentiment scores for each text
            
        Returns:
            DataFrame with sentiment analysis by topic
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        # Create DataFrame with topics and sentiments
        df = pd.DataFrame({
            'text': texts,
            'topic': self.topics,
            'sentiment_score': sentiment_scores
        })
        
        # Calculate sentiment statistics by topic
        sentiment_analysis = df.groupby('topic')['sentiment_score'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        sentiment_analysis.columns = [
            'Topic_ID', 'Document_Count', 'Mean_Sentiment', 
            'Std_Sentiment', 'Min_Sentiment', 'Max_Sentiment'
        ]
        
        # Add topic names
        sentiment_analysis['Topic_Name'] = sentiment_analysis['Topic_ID'].apply(
            lambda x: self._generate_topic_name(f"Topic_{x}")
        )
        
        return sentiment_analysis
    
    def predict_topics(self, new_texts: List[str]) -> Dict:
        """
        Predict topics for new documents.
        
        Args:
            new_texts: List of new text documents
            
        Returns:
            Dictionary with predictions
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        topics, probs = self.topic_model.transform(new_texts)
        
        return {
            'predicted_topics': topics,
            'probabilities': probs
        }
    
    def get_topic_summary(self) -> pd.DataFrame:
        """
        Get a comprehensive summary of all topics.
        
        Returns:
            DataFrame with topic summary
        """
        if self.topic_model is None:
            raise ValueError("Model not trained yet")
        
        topic_info = self.get_topic_info()
        
        # Add additional insights
        summary_data = []
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:  # Skip outliers
                topic_words = self.get_topic_words(row['Topic'], top_n=10)
                
                # Handle frequency/count column variations
                frequency = row.get('Frequency', row.get('Count', 0))
                
                summary_data.append({
                    'Topic_ID': row['Topic'],
                    'Topic_Name': row['topic_name'],
                    'Document_Count': frequency,
                    'Frequency': frequency,  # Keep both for compatibility
                    'Top_5_Words': ', '.join(topic_words[:5]),
                    'Top_10_Words': ', '.join(topic_words)
                })
        
        return pd.DataFrame(summary_data)
    
    def compare_with_traditional(self, traditional_topics: Dict) -> pd.DataFrame:
        """
        Compare BERTopic results with traditional topic modeling.
        
        Args:
            traditional_topics: Dictionary of traditional topics
            
        Returns:
            DataFrame with comparison
        """
        bertopic_summary = self.get_topic_summary()
        
        comparison_data = []
        for i, (_, bertopic_row) in enumerate(bertopic_summary.iterrows()):
            traditional_key = list(traditional_topics.keys())[i] if i < len(traditional_topics) else None
            
            if traditional_key:
                traditional_words = ', '.join([word for word, _ in traditional_topics[traditional_key][:5]])
            else:
                traditional_words = "N/A"
            
            comparison_data.append({
                'Method': 'BERTopic',
                'Topic_ID': bertopic_row['Topic_ID'],
                'Topic_Name': bertopic_row['Topic_Name'],
                'Top_Words': bertopic_row['Top_5_Words'],
                'Document_Count': bertopic_row['Document_Count']
            })
            
            if traditional_key:
                comparison_data.append({
                    'Method': 'Traditional LDA',
                    'Topic_ID': i,
                    'Topic_Name': traditional_key,
                    'Top_Words': traditional_words,
                    'Document_Count': 'N/A'
                })
        
        return pd.DataFrame(comparison_data)

def main():
    """
    Example usage of BERTopic modeling.
    """
    # Sample data (simulating John Lewis ad comments)
    sample_texts = [
        "the ad is emotional and touching father character",
        "christmas advertisement feels depressing not festive",
        "love the storytelling and emotional journey",
        "masculinity crisis portrayed in this advertisement",
        "great christmas ad brings tears to my eyes",
        "the father seems lost and confused throughout",
        "excellent cinematography and music choice",
        "this ad speaks to middle aged men struggles",
        "christmas spirit missing from this campaign",
        "powerful narrative about family dynamics",
        "the pink circles and blue squares symbolism",
        "feminine authority dominates the household",
        "emotional fragility of the father character",
        "nostalgic longing and awkward dance moves",
        "decline of traditional male roles in advertising"
    ]
    
    # Initialize and train BERTopic
    modeler = BERTopicModeler(n_topics=5, min_topic_size=2)
    results = modeler.fit_transform(sample_texts)
    
    # Display topic summary
    print("BERTopic Summary:")
    print(modeler.get_topic_summary())
    
    # Visualize topics
    modeler.visualize_topics_2d()
    modeler.visualize_topic_distribution(sample_texts)
    
    # Get topic information
    print("\nTopic Information:")
    print(modeler.get_topic_info())

if __name__ == "__main__":
    main()
