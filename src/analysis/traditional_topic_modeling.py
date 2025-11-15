import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim import corpora, models
from gensim.models import LdaModel, CoherenceModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

class TraditionalTopicModeler:
    """
    Traditional topic modeling using TF-IDF + LDA (sklearn and Gensim implementations).
    Based on Lecture 8 code with enhancements for social media analysis.
    """
    
    def __init__(self, n_topics: int = 5, max_features: int = 1000):
        self.n_topics = n_topics
        self.max_features = max_features
        
        # sklearn components
        self.tfidf_vectorizer = None
        self.sklearn_lda = None
        self.sklearn_feature_names = None
        
        # Gensim components
        self.gensim_dictionary = None
        self.gensim_corpus = None
        self.gensim_lda = None
        
        # Results storage
        self.sklearn_topics = {}
        self.gensim_topics = {}
        self.coherence_scores = {}
    
    def fit_sklearn_lda(self, texts: pd.Series) -> dict:
        """
        Fit sklearn LDA model using TF-IDF features.
        
        Args:
            texts: Preprocessed text series
            
        Returns:
            Dictionary with model results
        """
        print("Training sklearn LDA model...")
        
        # Create TF-IDF features
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        self.sklearn_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        
        # Fit LDA
        self.sklearn_lda = LatentDirichletAllocation(
            n_components=self.n_topics,
            random_state=42,
            max_iter=10,
            learning_method='online'
        )
        
        self.sklearn_lda.fit(tfidf_matrix)
        
        # Extract topics
        self.sklearn_topics = self._extract_sklearn_topics()
        
        # Calculate perplexity
        perplexity = self.sklearn_lda.perplexity(tfidf_matrix)
        
        results = {
            'model': self.sklearn_lda,
            'vectorizer': self.tfidf_vectorizer,
            'topics': self.sklearn_topics,
            'perplexity': perplexity,
            'feature_matrix': tfidf_matrix
        }
        
        print(f"Sklearn LDA trained with perplexity: {perplexity:.2f}")
        return results
    
    def fit_gensim_lda(self, texts: pd.Series) -> dict:
        """
        Fit Gensim LDA model using bag-of-words features.
        
        Args:
            texts: Preprocessed text series
            
        Returns:
            Dictionary with model results
        """
        print("Training Gensim LDA model...")
        
        # Tokenize texts
        tokenized_texts = [text.split() for text in texts]
        
        # Create dictionary and corpus
        self.gensim_dictionary = corpora.Dictionary(tokenized_texts)
        
        # Filter extremes
        self.gensim_dictionary.filter_extremes(
            no_below=2, 
            no_above=0.8, 
            keep_n=self.max_features
        )
        
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in tokenized_texts]
        
        # Train LDA model
        self.gensim_lda = LdaModel(
            corpus=self.gensim_corpus,
            id2word=self.gensim_dictionary,
            num_topics=self.n_topics,
            random_state=42,
            passes=10,
            alpha='auto',
            per_word_topics=True
        )
        
        # Extract topics
        self.gensim_topics = self._extract_gensim_topics()
        
        # Calculate coherence score
        coherence_model = CoherenceModel(
            model=self.gensim_lda,
            texts=tokenized_texts,
            dictionary=self.gensim_dictionary,
            coherence='c_v'
        )
        coherence_score = coherence_model.get_coherence()
        
        results = {
            'model': self.gensim_lda,
            'dictionary': self.gensim_dictionary,
            'corpus': self.gensim_corpus,
            'topics': self.gensim_topics,
            'coherence_score': coherence_score
        }
        
        print(f"Gensim LDA trained with coherence score: {coherence_score:.3f}")
        return results
    
    def _extract_sklearn_topics(self) -> dict:
        """Extract topics from sklearn LDA model."""
        topics = {}
        
        for topic_idx, topic in enumerate(self.sklearn_lda.components_):
            # Get top 10 words for this topic
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [(self.sklearn_feature_names[i], topic[i]) 
                        for i in top_words_idx]
            topics[f"Topic_{topic_idx}"] = top_words
        
        return topics
    
    def _extract_gensim_topics(self) -> dict:
        """Extract topics from Gensim LDA model."""
        topics = {}
        
        for topic_idx in range(self.n_topics):
            topic_words = self.gensim_lda.show_topic(topic_idx, topn=10)
            topics[f"Topic_{topic_idx}"] = topic_words
        
        return topics
    
    def predict_topics(self, texts: pd.Series, method: str = 'sklearn') -> pd.DataFrame:
        """
        Predict topic distribution for new texts.
        
        Args:
            texts: Text series to predict
            method: 'sklearn' or 'gensim'
            
        Returns:
            DataFrame with topic probabilities
        """
        if method == 'sklearn':
            if self.sklearn_lda is None:
                raise ValueError("Sklearn LDA model not trained yet")
            
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            topic_distributions = self.sklearn_lda.transform(tfidf_matrix)
            
            columns = [f"Topic_{i}" for i in range(self.n_topics)]
            return pd.DataFrame(topic_distributions, columns=columns)
        
        elif method == 'gensim':
            if self.gensim_lda is None:
                raise ValueError("Gensim LDA model not trained yet")
            
            tokenized_texts = [text.split() for text in texts]
            bow_corpus = [self.gensim_dictionary.doc2bow(text) for text in tokenized_texts]
            
            topic_distributions = []
            for bow in bow_corpus:
                topics = self.gensim_lda.get_document_topics(bow, minimum_probability=0)
                topic_dist = [0.0] * self.n_topics
                for topic_id, prob in topics:
                    topic_dist[topic_id] = prob
                topic_distributions.append(topic_dist)
            
            columns = [f"Topic_{i}" for i in range(self.n_topics)]
            return pd.DataFrame(topic_distributions, columns=columns)
    
    def evaluate_topics(self, texts: pd.Series, method: str = 'gensim') -> dict:
        """
        Evaluate topic model quality.
        
        Args:
            texts: Original texts
            method: 'sklearn' or 'gensim'
            
        Returns:
            Dictionary with evaluation metrics
        """
        if method == 'gensim':
            tokenized_texts = [text.split() for text in texts]
            
            # Coherence scores
            coherence_model = CoherenceModel(
                model=self.gensim_lda,
                texts=tokenized_texts,
                dictionary=self.gensim_dictionary,
                coherence='c_v'
            )
            coherence_score = coherence_model.get_coherence()
            
            # Perplexity (for Gensim)
            perplexity = self.gensim_lda.log_perplexity(self.gensim_corpus)
            
            return {
                'coherence_score': coherence_score,
                'perplexity': perplexity
            }
        
        elif method == 'sklearn':
            # For sklearn, we can only calculate perplexity
            tfidf_matrix = self.tfidf_vectorizer.transform(texts)
            perplexity = self.sklearn_lda.perplexity(tfidf_matrix)
            
            return {'perplexity': perplexity}
    
    def visualize_topics(self, method: str = 'sklearn', save_path: str = None):
        """
        Create visualizations for topics.
        
        Args:
            method: 'sklearn' or 'gensim'
            save_path: Optional path to save visualizations
        """
        topics = self.sklearn_topics if method == 'sklearn' else self.gensim_topics
        
        # Create word clouds for each topic
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (topic_name, words) in enumerate(topics.items()):
            if i >= len(axes):
                break
                
            # Create word frequency dictionary
            word_freq = {word: weight for word, weight in words}
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=400, height=300,
                background_color='white',
                max_words=50
            ).generate_from_frequencies(word_freq)
            
            axes[i].imshow(wordcloud, interpolation='bilinear')
            axes[i].set_title(f'{topic_name}', fontsize=14, fontweight='bold')
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(topics), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/wordclouds_{method}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_pyldavis_visualization(self, method: str = 'gensim', save_path: str = None):
        """
        Create interactive pyLDAvis visualization.
        
        Args:
            method: 'sklearn' or 'gensim' (only gensim supported for pyLDAvis)
            save_path: Optional path to save HTML
        """
        if method != 'gensim':
            print("pyLDAvis visualization only supported for Gensim model")
            return
        
        vis = gensimvis.prepare(
            self.gensim_lda, 
            self.gensim_corpus, 
            self.gensim_dictionary,
            sort_topics=False
        )
        
        if save_path:
            pyLDAvis.save_html(vis, f"{save_path}/ldavis_{method}.html")
        
        return vis
    
    def get_topic_summary(self, method: str = 'sklearn') -> pd.DataFrame:
        """
        Get a summary table of topics with their top words.
        
        Args:
            method: 'sklearn' or 'gensim'
            
        Returns:
            DataFrame with topic summary
        """
        topics = self.sklearn_topics if method == 'sklearn' else self.gensim_topics
        
        summary_data = []
        for topic_name, words in topics.items():
            top_words = ', '.join([word for word, _ in words[:5]])
            summary_data.append({
                'Topic': topic_name,
                'Top_5_Words': top_words,
                'Top_10_Words': ', '.join([word for word, _ in words])
            })
        
        return pd.DataFrame(summary_data)
    
    def find_optimal_topics(self, texts: pd.Series, max_topics: int = 10) -> dict:
        """
        Find optimal number of topics using coherence score.
        
        Args:
            texts: Preprocessed text series
            max_topics: Maximum number of topics to test
            
        Returns:
            Dictionary with coherence scores for each topic number
        """
        print("Finding optimal number of topics...")
        
        tokenized_texts = [text.split() for text in texts]
        dictionary = corpora.Dictionary(tokenized_texts)
        dictionary.filter_extremes(no_below=2, no_above=0.8)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
        
        coherence_scores = {}
        for n_topics in range(2, max_topics + 1):
            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=n_topics,
                random_state=42,
                passes=5
            )
            
            coherence_model = CoherenceModel(
                model=lda_model,
                texts=tokenized_texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            
            coherence_scores[n_topics] = coherence_model.get_coherence()
            print(f"Topics: {n_topics}, Coherence: {coherence_scores[n_topics]:.3f}")
        
        # Plot coherence scores
        plt.figure(figsize=(10, 6))
        plt.plot(list(coherence_scores.keys()), list(coherence_scores.values()), 'bo-')
        plt.xlabel('Number of Topics')
        plt.ylabel('Coherence Score')
        plt.title('Optimal Number of Topics')
        plt.grid(True)
        plt.show()
        
        return coherence_scores

def main():
    """
    Example usage of traditional topic modeling.
    """
    # Sample data (simulating John Lewis ad comments)
    sample_texts = pd.Series([
        "the ad is emotional and touching father character",
        "christmas advertisement feels depressing not festive",
        "love the storytelling and emotional journey",
        "masculinity crisis portrayed in this advertisement",
        "great christmas ad brings tears to my eyes",
        "the father seems lost and confused throughout",
        "excellent cinematography and music choice",
        "this ad speaks to middle aged men struggles",
        "christmas spirit missing from this campaign",
        "powerful narrative about family dynamics"
    ])
    
    # Initialize and train models
    modeler = TraditionalTopicModeler(n_topics=3, max_features=100)
    
    # Train sklearn LDA
    sklearn_results = modeler.fit_sklearn_lda(sample_texts)
    
    # Train Gensim LDA
    gensim_results = modeler.fit_gensim_lda(sample_texts)
    
    # Display topics
    print("\nSklearn Topics:")
    print(modeler.get_topic_summary('sklearn'))
    
    print("\nGensim Topics:")
    print(modeler.get_topic_summary('gensim'))
    
    # Visualize topics
    modeler.visualize_topics('sklearn')
    modeler.visualize_topics('gensim')

if __name__ == "__main__":
    main()
