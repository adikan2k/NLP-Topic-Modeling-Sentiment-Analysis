"""
Unified Analysis Pipeline for John Lewis Christmas Ad NLP Project

This pipeline orchestrates the entire NLP analysis workflow from data collection
through visualization. It was developed as part of a final project for NLP course.

Author: [Your Name]
Course: Natural Language Processing - Fall 2024
Date: November 2024

Note: This implementation went through several iterations during development.
Some design decisions were made due to time constraints and API limitations.
"""

import pandas as pd
import numpy as np
import pickle
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
from src.scrapers.youtube_scraper import YouTubeCommentScraper
from src.scrapers.reddit_scraper import RedditScraper
from src.analysis.preprocessing import TextPreprocessor
from src.analysis.traditional_topic_modeling import TraditionalTopicModeler
from src.analysis.bertopic_modeling import BERTopicModeler
from src.analysis.sentiment_analysis import SentimentAnalyzer
from src.utils.export_utils import DataExporter

# Import configuration
from src.config import (
    DATA_DIR, OUTPUT_DIR, PIPELINE_CONFIG, YOUTUBE_CONFIG,
    TRADITIONAL_TOPIC_CONFIG, BERTOPIC_CONFIG, SENTIMENT_CONFIG,
    LOGGING_CONFIG
)

class AnalysisPipeline:
    """
    Unified pipeline for comprehensive NLP analysis of John Lewis Christmas ad comments.
    Features modular execution, error recovery, and intermediate result caching.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        self.config = config or {}
        self.results = {}
        self.step_status = {}
        self.errors = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.logger.info("Initializing pipeline components...")
        
        self.scraper = YouTubeCommentScraper()
        self.preprocessor = TextPreprocessor()
        self.traditional_modeler = TraditionalTopicModeler(
            n_topics=TRADITIONAL_TOPIC_CONFIG['n_topics'],
            max_features=TRADITIONAL_TOPIC_CONFIG['max_features']
        )
        self.bertopic_modeler = BERTopicModeler(
            embedding_model=BERTOPIC_CONFIG['embedding_model'],
            n_topics=BERTOPIC_CONFIG['n_topics'],
            min_topic_size=BERTOPIC_CONFIG['min_topic_size']
        )
        self.sentiment_analyzer = SentimentAnalyzer(
            transformer_model=SENTIMENT_CONFIG['transformer_model'],
            use_transformer=SENTIMENT_CONFIG['use_transformer']
        )
        
        self.logger.info("Analysis pipeline initialized")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, LOGGING_CONFIG['level']),
            format=LOGGING_CONFIG['format'],
            handlers=[
                logging.FileHandler(LOGGING_CONFIG['file']),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _get_cache_path(self, step_name: str) -> Path:
        """Get cache file path for a given step."""
        cache_dir = OUTPUT_DIR / "cache"
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{step_name}.pkl"
    
    def _save_cache(self, step_name: str, data: Any):
        """Save intermediate results to cache."""
        cache_path = self._get_cache_path(step_name)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Cached results for {step_name}")
        except Exception as e:
            self.logger.error(f"Failed to cache {step_name}: {e}")
    
    def _load_cache(self, step_name: str) -> Optional[Any]:
        """Load cached results if available."""
        cache_path = self._get_cache_path(step_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                self.logger.info(f"Loaded cached results for {step_name}")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load cache for {step_name}: {e}")
        return None
    
    def _get_data_hash(self, data: pd.DataFrame) -> str:
        """Generate hash for data to detect changes."""
        data_str = data.to_string()
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _update_step_status(self, step_name: str, status: str, error: Optional[str] = None):
        """Update step status and save progress."""
        self.step_status[step_name] = {
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'error': error
        }
        
        if error:
            self.errors[step_name] = error
        
        # Save progress
        progress_path = OUTPUT_DIR / "pipeline_progress.json"
        try:
            with open(progress_path, 'w') as f:
                json.dump({
                    'step_status': self.step_status,
                    'errors': self.errors,
                    'last_updated': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")
    
    def run_step(self, step_name: str, force_rerun: bool = False) -> bool:
        """
        Run a specific step in the pipeline.
        
        Args:
            step_name: Name of the step to run
            force_rerun: Whether to force rerun even if cached results exist
            
        Returns:
            True if step completed successfully, False otherwise
        """
        self.logger.info(f"Running step: {step_name}")
        
        try:
            # Check if we have cached results and don't need to rerun
            if not force_rerun:
                cached_data = self._load_cache(step_name)
                if cached_data is not None:
                    self.results[step_name] = cached_data
                    self._update_step_status(step_name, "completed")
                    self.logger.info(f"Step {step_name} loaded from cache")
                    return True
            
            # Run the specific step
            if step_name == "scraping":
                success = self._run_scraping()
            elif step_name == "preprocessing":
                success = self._run_preprocessing()
            elif step_name == "traditional_topics":
                success = self._run_traditional_topics()
            elif step_name == "bertopic":
                success = self._run_bertopic()
            elif step_name == "sentiment_analysis":
                success = self._run_sentiment_analysis()
            elif step_name == "emotion_analysis":
                success = self._run_emotion_analysis()
            elif step_name == "demographic_analysis":
                success = self._run_demographic_analysis()
            elif step_name == "visualization":
                success = self._run_visualization()
            elif step_name == "export":
                success = self._run_export()
            else:
                raise ValueError(f"Unknown step: {step_name}")
            
            if success:
                self._save_cache(step_name, self.results[step_name])
                self._update_step_status(step_name, "completed")
                self.logger.info(f"Step {step_name} completed successfully")
            else:
                self._update_step_status(step_name, "failed", "Step returned False")
                self.logger.error(f"Step {step_name} failed")
            
            return success
            
        except Exception as e:
            error_msg = f"Step {step_name} failed with error: {str(e)}"
            self.logger.error(error_msg)
            self._update_step_status(step_name, "failed", error_msg)
            return False
    
    def _run_scraping(self) -> bool:
        """Run real data scraping from YouTube and Reddit."""
        if not PIPELINE_CONFIG.get("run_scraping", True):
            self.logger.info("Scraping disabled in config")
            return True
        
        try:
            # Import scrapers
            from src.scrapers.youtube_scraper import YouTubeCommentScraper
            from src.scrapers.reddit_scraper import RedditScraper
            
            all_data = []
            
            # 1. Scrape YouTube comments
            self.logger.info("Starting YouTube comment scraping...")
            youtube_scraper = YouTubeCommentScraper()
            
            # John Lewis Christmas ad URLs
            youtube_urls = [
                "https://www.youtube.com/watch?v=dc5S4IV_NeA",  # John Lewis Christmas Ad 2025
                "https://www.youtube.com/watch?v=z1bRlnyQeDk",  # Another John Lewis Christmas video
            ]
            
            youtube_data = youtube_scraper.scrape_multiple_videos(
                youtube_urls, 
                max_comments_per_video=500
            )
            
            if not youtube_data.empty:
                # Standardize column names
                youtube_data = youtube_data.rename(columns={
                    'text': 'text',
                    'author': 'author', 
                    'likes': 'likes',
                    'replies': 'replies',
                    'timestamp': 'timestamp'
                })
                youtube_data['source'] = 'youtube'
                all_data.append(youtube_data)
                self.logger.info(f"Scraped {len(youtube_data)} YouTube comments")
            else:
                self.logger.warning("No YouTube comments were scraped")
            
            # 2. Scrape Reddit discussions
            self.logger.info("Starting Reddit scraping...")
            reddit_scraper = RedditScraper()
            
            reddit_data = reddit_scraper.search_john_lewis_posts(limit=50)
            
            if not reddit_data.empty:
                # Standardize column names to match YouTube format
                reddit_data = reddit_data.rename(columns={
                    'text': 'text',
                    'author': 'author',
                    'score': 'likes',  # Reddit score acts like likes
                    'num_comments': 'replies',  # Post comment count
                    'created_utc': 'timestamp'
                })
                reddit_data['source'] = 'reddit'
                all_data.append(reddit_data)
                self.logger.info(f"Scraped {len(reddit_data)} Reddit posts")
            else:
                self.logger.warning("No Reddit posts were scraped")
            
            # 3. LinkedIn scraping (manual note - LinkedIn API is restricted)
            self.logger.info("LinkedIn scraping skipped - requires manual export due to API restrictions")
            self.logger.info("To include LinkedIn data, manually export comments from: https://www.linkedin.com/posts/johnlewisandpartners_wherelovelives-jlchristmas-activity-7391396118141743105-wYM2")
            
            # Combine all data
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Ensure required columns exist
                required_columns = ['text', 'author', 'likes', 'replies', 'timestamp', 'source']
                for col in required_columns:
                    if col not in combined_data.columns:
                        if col == 'likes':
                            combined_data[col] = 0
                        elif col == 'replies':
                            combined_data[col] = 0
                        elif col == 'timestamp':
                            combined_data[col] = pd.NaT
                        else:
                            combined_data[col] = ''
                
                # Clean and filter data
                combined_data = combined_data.dropna(subset=['text'])
                combined_data = combined_data[combined_data['text'].str.len() > 10]
                
                self.results['scraping'] = combined_data
                self.logger.info(f"Total scraped data: {len(combined_data)} comments/posts from {len(all_data)} sources")
                
                # Save raw data
                self._save_raw_data(combined_data)
                
                return True
            else:
                self.logger.error("No data was scraped from any source")
                return False
                
        except Exception as e:
            self.logger.error(f"Scraping failed: {e}")
            # Fallback to sample data if real scraping fails
            self.logger.info("Falling back to sample data for demonstration...")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> bool:
        """Create sample data as fallback."""
        sample_texts = [
            "This John Lewis ad is absolutely amazing! So emotional and touching!",
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
        
        sample_data = pd.DataFrame({
            'text': sample_texts,
            'author': [f'user_{i}' for i in range(len(sample_texts))],
            'timestamp': pd.date_range('2024-11-01', periods=len(sample_texts), freq='H'),
            'likes': np.random.randint(0, 100, len(sample_texts)),
            'replies': np.random.randint(0, 20, len(sample_texts)),
            'source': 'sample'
        })
        
        self.results['scraping'] = sample_data
        self.logger.info(f"Created sample data with {len(sample_data)} comments")
        return True
    
    def _save_raw_data(self, data: pd.DataFrame):
        """Save raw scraped data."""
        try:
            os.makedirs('data/raw', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f'data/raw/scraped_data_{timestamp}.csv'
            data.to_csv(filepath, index=False, encoding='utf-8')
            self.logger.info(f"Raw data saved to: {filepath}")
        except Exception as e:
            self.logger.warning(f"Could not save raw data: {e}")
    
    def _run_preprocessing(self) -> bool:
        """Run text preprocessing."""
        if not PIPELINE_CONFIG.get("run_preprocessing", True):
            return True
        
        if 'scraping' not in self.results:
            raise ValueError("Scraping results not available")
        
        raw_data = self.results['scraping']
        processed_data = self.preprocessor.preprocess_dataframe(raw_data)
        
        self.results['preprocessing'] = processed_data
        self.logger.info(f"Preprocessed {len(processed_data)} comments")
        return True
    
    def _run_traditional_topics(self) -> bool:
        """Run traditional topic modeling."""
        if not PIPELINE_CONFIG.get("run_traditional_topics", True):
            return True
        
        if 'preprocessing' not in self.results:
            raise ValueError("Preprocessing results not available")
        
        processed_data = self.results['preprocessing']
        texts = processed_data['processed_text']
        
        # Train sklearn LDA
        sklearn_results = self.traditional_modeler.fit_sklearn_lda(texts)
        
        # Train Gensim LDA
        gensim_results = self.traditional_modeler.fit_gensim_lda(texts)
        
        # Get topic predictions
        topic_predictions = self.traditional_modeler.predict_topics(texts, method='sklearn')
        
        self.results['traditional_topics'] = {
            'sklearn_results': sklearn_results,
            'gensim_results': gensim_results,
            'topic_predictions': topic_predictions,
            'topic_summary': self.traditional_modeler.get_topic_summary('sklearn'),
            'modeler': self.traditional_modeler
        }
        
        self.logger.info("Traditional topic modeling completed")
        return True
    
    def _run_bertopic(self) -> bool:
        """Run BERTopic analysis."""
        if not PIPELINE_CONFIG.get("run_bertopic", True):
            return True
        
        if 'preprocessing' not in self.results:
            raise ValueError("Preprocessing results not available")
        
        processed_data = self.results['preprocessing']
        texts = processed_data['processed_text'].tolist()
        
        # Fit BERTopic model
        bertopic_results = self.bertopic_modeler.fit_transform(texts)
        
        # Get topic predictions and summary
        topic_summary = self.bertopic_modeler.get_topic_summary()
        topic_info = self.bertopic_modeler.get_topic_info()
        
        self.results['bertopic'] = {
            'results': bertopic_results,
            'topic_summary': topic_summary,
            'topic_info': topic_info,
            'modeler': self.bertopic_modeler
        }
        
        self.logger.info("BERTopic analysis completed")
        return True
    
    def _run_sentiment_analysis(self) -> bool:
        """Run sentiment analysis."""
        if not PIPELINE_CONFIG.get("run_sentiment_analysis", True):
            return True
        
        if 'preprocessing' not in self.results:
            raise ValueError("Preprocessing results not available")
        
        processed_data = self.results['preprocessing']
        texts = processed_data['sentiment_text'].tolist()
        
        # Analyze sentiment with TextBlob
        textblob_results = self.sentiment_analyzer.analyze_textblob_sentiment(texts)
        
        # Analyze sentiment with transformer
        transformer_results = self.sentiment_analyzer.analyze_transformer_sentiment(texts)
        
        # Combine results
        combined_results = self.sentiment_analyzer.combine_sentiment_results(
            textblob_results, transformer_results
        )
        
        # Get summary statistics
        sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(combined_results)
        
        self.results['sentiment_analysis'] = {
            'textblob_results': textblob_results,
            'transformer_results': transformer_results,
            'combined_results': combined_results,
            'sentiment_summary': sentiment_summary,
            'analyzer': self.sentiment_analyzer
        }
        
        self.logger.info("Sentiment analysis completed")
        return True
    
    def _run_emotion_analysis(self) -> bool:
        """Run emotion analysis."""
        if not PIPELINE_CONFIG.get("run_emotion_analysis", True):
            return True
        
        if 'preprocessing' not in self.results:
            raise ValueError("Preprocessing results not available")
        
        processed_data = self.results['preprocessing']
        texts = processed_data['sentiment_text'].tolist()
        
        # Analyze emotions
        emotion_results = self.sentiment_analyzer.analyze_emotions(texts)
        
        self.results['emotion_analysis'] = {
            'emotion_results': emotion_results
        }
        
        self.logger.info("Emotion analysis completed")
        return True
    
    def _run_demographic_analysis(self) -> bool:
        """Run demographic analysis using linguistic proxies."""
        if not PIPELINE_CONFIG.get("run_demographic_analysis", True):
            return True
        
        if 'sentiment_analysis' not in self.results:
            raise ValueError("Sentiment analysis results not available")
        
        sentiment_data = self.results['sentiment_analysis']['combined_results']
        demographic_data = self.sentiment_analyzer.analyze_sentiment_by_demographics(
            sentiment_data, text_column='text'
        )
        
        self.results['demographic_analysis'] = {
            'demographic_data': demographic_data
        }
        
        self.logger.info("Demographic analysis completed")
        return True
    
    def _run_visualization(self) -> bool:
        """Run visualization generation."""
        if not PIPELINE_CONFIG.get("generate_visualizations", True):
            return True
        
        viz_dir = OUTPUT_DIR / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Generate topic model visualizations
        if 'traditional_topics' in self.results:
            traditional_modeler = self.results['traditional_topics']['modeler']
            traditional_modeler.visualize_topics('sklearn', str(viz_dir))
            traditional_modeler.visualize_topics('gensim', str(viz_dir))
        
        # Generate BERTopic visualizations
        if 'bertopic' in self.results:
            bertopic_modeler = self.results['bertopic']['modeler']
            bertopic_modeler.visualize_topics_2d(str(viz_dir))
            bertopic_modeler.visualize_topic_distribution(
                self.results['preprocessing']['processed_text'].tolist(),
                str(viz_dir)
            )
        
        # Generate sentiment visualizations
        if 'demographic_analysis' in self.results:
            demographic_data = self.results['demographic_analysis']['demographic_data']
            sentiment_analyzer = self.results['sentiment_analysis']['analyzer']
            sentiment_analyzer.create_sentiment_visualizations(demographic_data, str(viz_dir))
        
        self.results['visualization'] = {'output_dir': str(viz_dir)}
        self.logger.info("Visualizations generated")
        return True
    
    def _run_export(self) -> bool:
        """Export results to CSV files."""
        if not PIPELINE_CONFIG.get("export_results", True):
            return True
        
        export_dir = OUTPUT_DIR / "exports"
        export_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export main results
        if 'preprocessing' in self.results:
            self.results['preprocessing'].to_csv(
                export_dir / f"processed_data_{timestamp}.csv", index=False
            )
        
        if 'sentiment_analysis' in self.results:
            self.results['sentiment_analysis']['combined_results'].to_csv(
                export_dir / f"sentiment_results_{timestamp}.csv", index=False
            )
        
        if 'traditional_topics' in self.results:
            self.results['traditional_topics']['topic_summary'].to_csv(
                export_dir / f"traditional_topics_{timestamp}.csv", index=False
            )
        
        if 'bertopic' in self.results:
            self.results['bertopic']['topic_summary'].to_csv(
                export_dir / f"bertopic_topics_{timestamp}.csv", index=False
            )
        
        # Export comprehensive summary
        self._create_comprehensive_report(export_dir, timestamp)
        
        self.results['export'] = {'output_dir': str(export_dir)}
        self.logger.info("Results exported")
        return True
    
    def _create_comprehensive_report(self, export_dir: Path, timestamp: str):
        """Create a comprehensive analysis report."""
        report_data = {
            'analysis_timestamp': timestamp,
            'total_comments_analyzed': len(self.results.get('preprocessing', pd.DataFrame())),
            'pipeline_steps_completed': list(self.results.keys()),
            'step_status': self.step_status
        }
        
        # Add topic modeling summary
        if 'traditional_topics' in self.results:
            report_data['traditional_topics_found'] = len(self.results['traditional_topics']['topic_summary'])
        
        if 'bertopic' in self.results:
            report_data['bertopic_topics_found'] = len(self.results['bertopic']['topic_summary'])
        
        # Add sentiment summary
        if 'sentiment_analysis' in self.results:
            sentiment_summary = self.results['sentiment_analysis']['sentiment_summary']
            report_data['sentiment_distribution'] = sentiment_summary.to_dict('records')
        
        # Save report
        with open(export_dir / f"analysis_report_{timestamp}.json", 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def run_full_pipeline(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            resume_from: Optional step name to resume from
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting full analysis pipeline")
        
        # Define step order
        steps = [
            "scraping",
            "preprocessing", 
            "traditional_topics",
            "bertopic",
            "sentiment_analysis",
            "emotion_analysis",
            "demographic_analysis",
            "visualization",
            "export"
        ]
        
        # If resuming, find the step to start from
        start_index = 0
        if resume_from:
            if resume_from in steps:
                start_index = steps.index(resume_from)
                self.logger.info(f"Resuming pipeline from step: {resume_from}")
            else:
                self.logger.warning(f"Unknown resume step: {resume_from}, starting from beginning")
        
        # Run steps
        for step in steps[start_index:]:
            success = self.run_step(step)
            if not success:
                self.logger.error(f"Pipeline failed at step: {step}")
                break
        
        self.logger.info("Pipeline execution completed")
        return self.results
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get a summary of all results."""
        summary = {
            'completed_steps': list(self.results.keys()),
            'step_status': self.step_status,
            'errors': self.errors
        }
        
        # Add key metrics
        if 'preprocessing' in self.results:
            summary['total_comments'] = len(self.results['preprocessing'])
        
        if 'sentiment_analysis' in self.results:
            sentiment_data = self.results['sentiment_analysis']['combined_results']
            sentiment_dist = sentiment_data['consensus_sentiment'].value_counts().to_dict()
            summary['sentiment_distribution'] = sentiment_dist
        
        if 'traditional_topics' in self.results:
            summary['traditional_topics_count'] = len(self.results['traditional_topics']['topic_summary'])
        
        if 'bertopic' in self.results:
            summary['bertopic_topics_count'] = len(self.results['bertopic']['topic_summary'])
        
        return summary

def main():
    """
    Example usage of the analysis pipeline.
    """
    # Initialize pipeline
    pipeline = AnalysisPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline()
    
    # Print summary
    summary = pipeline.get_results_summary()
    print("\nPipeline Results Summary:")
    print(json.dumps(summary, indent=2, default=str))

if __name__ == "__main__":
    main()
