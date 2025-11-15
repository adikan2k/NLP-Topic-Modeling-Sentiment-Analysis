import pandas as pd
import json
import time
from datetime import datetime
from youtube_comment_downloader import *
import re
from typing import List, Dict, Optional

class YouTubeCommentScraper:
    """
    A comprehensive YouTube comment scraper for the John Lewis Christmas ad analysis.
    Collects comments along with metadata for demographic and temporal analysis.
    """
    
    def __init__(self):
        self.downloader = YoutubeCommentDownloader()
        
    def scrape_video_comments(self, video_url: str, max_comments: int = 1000) -> pd.DataFrame:
        """
        Scrape comments from a YouTube video.
        
        Args:
            video_url: YouTube video URL
            max_comments: Maximum number of comments to collect
            
        Returns:
            DataFrame with comments and metadata
        """
        comments_data = []
        
        try:
            # Extract video ID from URL
            video_id = self._extract_video_id(video_url)
            if not video_id:
                raise ValueError("Invalid YouTube URL")
            
            print(f"Scraping comments from video: {video_id}")
            
            # Download comments (without max_comments parameter)
            comments = self.downloader.get_comments_from_url(video_url)
            
            # Manually limit the number of comments
            for i, comment in enumerate(comments):
                if i >= max_comments:
                    break
                    
                # Process and clean comment data
                processed_comment = self._process_comment(comment, video_id)
                if processed_comment:
                    comments_data.append(processed_comment)
                    
                # Rate limiting
                time.sleep(0.1)
            
            df = pd.DataFrame(comments_data)
            print(f"Successfully scraped {len(df)} comments")
            return df
            
        except Exception as e:
            print(f"Error scraping comments: {e}")
            return pd.DataFrame()
    
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
            r'youtube\.com/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None
    
    def _process_comment(self, comment: Dict, video_id: str) -> Optional[Dict]:
        """Process and clean individual comment data."""
        try:
            # Handle likes field that might be empty string
            likes_str = comment.get('likes', '0')
            try:
                likes = int(likes_str) if likes_str and likes_str.strip() else 0
            except (ValueError, TypeError):
                likes = 0
            
            # Handle replies field that might be empty string
            replies_str = comment.get('replies', '0')
            try:
                replies = int(replies_str) if replies_str and replies_str.strip() else 0
            except (ValueError, TypeError):
                replies = 0
            
            processed = {
                'video_id': video_id,
                'comment_id': comment.get('cid', ''),
                'text': comment.get('text', '').strip(),
                'author': comment.get('author', ''),
                'channel_id': comment.get('channel_id', ''),
                'likes': likes,
                'replies': replies,
                'time': comment.get('time', ''),
                'timestamp': comment.get('time_parsed', None),
                'scraped_at': datetime.now().isoformat()
            }
            
            # Skip empty or very short comments
            if len(processed['text']) < 3:
                return None
                
            return processed
            
        except Exception as e:
            print(f"Error processing comment: {e}")
            return None
    
    def scrape_multiple_videos(self, video_urls: List[str], max_comments_per_video: int = 500) -> pd.DataFrame:
        """
        Scrape comments from multiple videos.
        
        Args:
            video_urls: List of YouTube video URLs
            max_comments_per_video: Max comments per video
            
        Returns:
            Combined DataFrame with all comments
        """
        all_comments = []
        
        for i, url in enumerate(video_urls):
            print(f"Processing video {i+1}/{len(video_urls)}")
            df = self.scrape_video_comments(url, max_comments_per_video)
            if not df.empty:
                all_comments.append(df)
            
            # Delay between videos
            time.sleep(1)
        
        if all_comments:
            return pd.concat(all_comments, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_comments(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save comments to CSV file.
        
        Args:
            df: DataFrame with comments
            filename: Optional custom filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"youtube_comments_{timestamp}.csv"
        
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Comments saved to: {filepath}")
        return filepath

def main():
    """
    Example usage for scraping John Lewis Christmas ad comments.
    """
    scraper = YouTubeCommentScraper()
    
    # John Lewis 2025 Christmas ad URLs
    video_urls = [
        "https://www.youtube.com/watch?v=dc5S4IV_NeA",  # John Lewis Christmas Ad 2025
        "https://www.youtube.com/watch?v=z1bRlnyQeDk",  # Another John Lewis Christmas video
    ]
    
    # Scrape comments
    comments_df = scraper.scrape_multiple_videos(video_urls, max_comments_per_video=500)
    
    if not comments_df.empty:
        # Save comments
        saved_path = scraper.save_comments(comments_df)
        
        # Display summary
        print(f"\nScraping Summary:")
        print(f"Total comments: {len(comments_df)}")
        print(f"Unique authors: {comments_df['author'].nunique()}")
        print(f"Average likes per comment: {comments_df['likes'].mean():.2f}")
        print(f"Date range: {comments_df['timestamp'].min()} to {comments_df['timestamp'].max()}")
        
        # Display sample comments
        print(f"\nSample comments:")
        print(comments_df[['text', 'author', 'likes']].head())
    else:
        print("No comments were scraped.")

if __name__ == "__main__":
    main()
