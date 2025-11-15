import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """
    Comprehensive text preprocessing for NLP analysis.
    Based on lecture code with enhancements for social media text.
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Social media specific stopwords
        self.social_media_stops = {
            'like', 'lol', 'omg', 'wow', 'yeah', 'yes', 'no', 'ok', 'okay',
            'good', 'great', 'bad', 'nice', 'really', 'very', 'just', 'would',
            'could', 'should', 'might', 'must', 'will', 'going', 'get', 'got'
        }
        self.stop_words.update(self.social_media_stops)
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning - remove URLs, mentions, special characters.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions and hashtags (keep the text)
        text = re.sub(r'@\w+|#', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text - lowercase, remove punctuation, numbers.
        
        Args:
            text: Text string
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text: str) -> List[str]:
        """
        Tokenize text and remove stopwords.
        
        Args:
            text: Text string
            
        Returns:
            List of tokens without stopwords
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) > 2
                 and token.isalpha()]
        
        return tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens to their base form.
        
        Args:
            tokens: List of tokens
            
        Returns:
            List of lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_for_topic_modeling(self, text: str) -> str:
        """
        Complete preprocessing pipeline for topic modeling.
        Less aggressive for small datasets.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Clean text
        text = self.clean_text(text)
        
        # Normalize (but keep some punctuation for small datasets)
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation but keep spaces
        text = re.sub(r'\d+', '', text)  # Remove numbers
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords (but be less aggressive)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) > 1  # Allow 2-letter words for small datasets
                 and token.isalpha()]
        
        # Lemmatize
        tokens = self.lemmatize_tokens(tokens)
        
        # Join back to string
        result = ' '.join(tokens)
        
        # Ensure we have some content
        if len(result.split()) < 2:
            # Fallback: just return cleaned lowercase text
            return self.clean_text(text).lower()
        
        return result
    
    def preprocess_for_sentiment_analysis(self, text: str) -> str:
        """
        Preprocessing for sentiment analysis (less aggressive).
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text string
        """
        # Basic cleaning only for sentiment analysis
        text = self.clean_text(text)
        
        # Remove extra whitespace but keep punctuation and capitalization
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def add_text_features(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Add additional text features for analysis.
        
        Args:
            df: DataFrame with text data
            text_column: Name of text column
            
        Returns:
            DataFrame with additional features
        """
        df = df.copy()
        
        # Basic text statistics
        df['text_length'] = df[text_column].astype(str).apply(len)
        df['word_count'] = df[text_column].astype(str).apply(lambda x: len(x.split()))
        df['avg_word_length'] = df[text_column].astype(str).apply(
            lambda x: sum(len(word) for word in x.split()) / len(x.split()) if x.split() else 0
        )
        
        # Sentiment polarity using TextBlob
        df['sentiment_polarity'] = df[text_column].astype(str).apply(
            lambda x: TextBlob(x).sentiment.polarity if x else 0
        )
        df['sentiment_subjectivity'] = df[text_column].astype(str).apply(
            lambda x: TextBlob(x).sentiment.subjectivity if x else 0
        )
        
        # Presence of emotional indicators
        df['has_exclamation'] = df[text_column].astype(str).apply(lambda x: '!' in x)
        df['has_question'] = df[text_column].astype(str).apply(lambda x: '?' in x)
        df['has_emoji'] = df[text_column].astype(str).apply(self._has_emoji)
        
        return df
    
    def _has_emoji(self, text: str) -> bool:
        """Check if text contains emojis."""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return bool(emoji_pattern.search(text))
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Apply preprocessing to entire DataFrame.
        
        Args:
            df: Input DataFrame
            text_column: Name of text column to preprocess
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Add original text length before preprocessing
        df['original_length'] = df[text_column].astype(str).apply(len)
        
        # Apply preprocessing
        df['clean_text'] = df[text_column].apply(self.clean_text)
        df['normalized_text'] = df['clean_text'].apply(self.normalize_text)
        df['processed_text'] = df['normalized_text'].apply(self.preprocess_for_topic_modeling)
        df['sentiment_text'] = df['clean_text'].apply(self.preprocess_for_sentiment_analysis)
        
        # Add text features
        df = self.add_text_features(df, text_column)
        
        # Filter out very short or empty texts
        df = df[df['processed_text'].str.len() > 10]
        
        print(f"Preprocessing complete. {len(df)} comments remaining after filtering.")
        
        return df

def main():
    """
    Example usage of the preprocessing pipeline.
    """
    # Sample data
    sample_data = {
        'text': [
            "Wow! This John Lewis ad is amazing 😍 #ChristmasAd",
            "I don't know... this feels a bit depressing for Christmas 🤔",
            "Can't believe they spent so much on this ad... https://youtube.com/watch?v=abc",
            "@user123 exactly what I was thinking! The father character seems lost",
            "Great ad! Love the emotional storytelling ❤️"
        ]
    }
    
    df = pd.DataFrame(sample_data)
    preprocessor = TextPreprocessor()
    
    # Preprocess the data
    processed_df = preprocessor.preprocess_dataframe(df)
    
    print("Original DataFrame:")
    print(df[['text']])
    
    print("\nProcessed DataFrame:")
    print(processed_df[['text', 'processed_text', 'sentiment_polarity', 'word_count']])

if __name__ == "__main__":
    main()
