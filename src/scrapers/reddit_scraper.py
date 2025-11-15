import pandas as pd
import praw
from datetime import datetime
import time
from typing import List, Dict, Optional
import os

class RedditScraper:
    """
    Reddit scraper for collecting comments about John Lewis Christmas ads.
    Uses PRAW (Python Reddit API Wrapper) for authentic API access.
    """
    
    def __init__(self, client_id: str = None, client_secret: str = None, user_agent: str = None):
        """
        Initialize Reddit scraper with API credentials.
        
        Args:
            client_id: Reddit API client ID
            client_secret: Reddit API client secret
            user_agent: Reddit API user agent
        """
        # Try to get credentials from environment variables or use defaults for testing
        self.client_id = client_id or os.getenv('REDDIT_CLIENT_ID', 'placeholder_client_id')
        self.client_secret = client_secret or os.getenv('REDDIT_CLIENT_SECRET', 'placeholder_client_secret')
        self.user_agent = user_agent or os.getenv('REDDIT_USER_AGENT', 'NLP Analysis Bot 1.0')
        
        self.reddit = None
        self._initialize_reddit()
    
    def _initialize_reddit(self):
        """Initialize Reddit API connection."""
        try:
            self.reddit = praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent
            )
            
            # Test the connection
            print(f"Reddit API connection successful. Read-only: {self.reddit.read_only}")
            
        except Exception as e:
            print(f"Reddit API connection failed: {e}")
            print("To use Reddit scraping, please:")
            print("1. Create a Reddit app at https://www.reddit.com/prefs/apps")
            print("2. Set environment variables: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
            print("3. Or pass credentials directly to RedditScraper()")
    
    def search_posts(self, query: str, subreddit: str = None, limit: int = 100, time_filter: str = 'year') -> pd.DataFrame:
        """
        Search Reddit posts about John Lewis Christmas ads.
        
        Args:
            query: Search query (e.g., "John Lewis Christmas ad")
            subreddit: Specific subreddit to search (None for all)
            limit: Maximum number of posts to retrieve
            time_filter: Time period (hour, day, week, month, year, all)
            
        Returns:
            DataFrame with posts and their comments
        """
        if not self.reddit:
            print("Reddit API not initialized. Returning empty DataFrame.")
            return pd.DataFrame()
        
        posts_data = []
        
        try:
            print(f"Searching Reddit for: '{query}' in {subreddit or 'all subreddits'}")
            
            # Search for posts
            if subreddit:
                subreddit_obj = self.reddit.subreddit(subreddit)
                search_results = subreddit_obj.search(query, limit=limit, time_filter=time_filter)
            else:
                search_results = self.reddit.subreddit('all').search(query, limit=limit, time_filter=time_filter)
            
            for submission in search_results:
                # Process post
                post_data = self._process_post(submission)
                if post_data:
                    posts_data.append(post_data)
                
                # Rate limiting
                time.sleep(0.5)
            
            df = pd.DataFrame(posts_data)
            print(f"Found {len(df)} posts from Reddit")
            return df
            
        except Exception as e:
            print(f"Error searching Reddit posts: {e}")
            return pd.DataFrame()
    
    def get_post_comments(self, submission_url: str, max_comments: int = 100) -> pd.DataFrame:
        """
        Get comments from a specific Reddit post.
        
        Args:
            submission_url: URL of the Reddit post
            max_comments: Maximum number of comments to retrieve
            
        Returns:
            DataFrame with comments
        """
        if not self.reddit:
            print("Reddit API not initialized. Returning empty DataFrame.")
            return pd.DataFrame()
        
        comments_data = []
        
        try:
            submission = self.reddit.submission(url=submission_url)
            print(f"Getting comments from post: {submission.title}")
            
            # Replace more (load more comments) to get all comments
            submission.comments.replace_more(limit=0)
            
            for comment in submission.comments.list()[:max_comments]:
                comment_data = self._process_comment(comment, submission.id)
                if comment_data:
                    comments_data.append(comment_data)
                
                # Rate limiting
                time.sleep(0.1)
            
            df = pd.DataFrame(comments_data)
            print(f"Retrieved {len(df)} comments from Reddit post")
            return df
            
        except Exception as e:
            print(f"Error getting Reddit comments: {e}")
            return pd.DataFrame()
    
    def _process_post(self, submission) -> Optional[Dict]:
        """Process Reddit post data."""
        try:
            processed = {
                'post_id': submission.id,
                'text': submission.title + "\n" + (submission.selftext or ''),
                'author': str(submission.author) if submission.author else '[deleted]',
                'subreddit': str(submission.subreddit),
                'score': submission.score,
                'upvote_ratio': submission.upvote_ratio,
                'num_comments': submission.num_comments,
                'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                'url': submission.url,
                'scraped_at': datetime.now().isoformat(),
                'source': 'reddit_post'
            }
            
            # Skip empty or very short posts
            if len(processed['text'].strip()) < 10:
                return None
                
            return processed
            
        except Exception as e:
            print(f"Error processing Reddit post: {e}")
            return None
    
    def _process_comment(self, comment, post_id: str) -> Optional[Dict]:
        """Process Reddit comment data."""
        try:
            processed = {
                'post_id': post_id,
                'comment_id': comment.id,
                'text': comment.body,
                'author': str(comment.author) if comment.author else '[deleted]',
                'score': comment.score,
                'depth': comment.depth,
                'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat(),
                'scraped_at': datetime.now().isoformat(),
                'source': 'reddit_comment'
            }
            
            # Skip empty or very short comments
            if len(processed['text'].strip()) < 3:
                return None
                
            return processed
            
        except Exception as e:
            print(f"Error processing Reddit comment: {e}")
            return None
    
    def search_john_lewis_posts(self, limit: int = 50) -> pd.DataFrame:
        """
        Search specifically for John Lewis Christmas ad discussions.
        
        Args:
            limit: Maximum number of posts to retrieve
            
        Returns:
            DataFrame with relevant posts
        """
        search_queries = [
            "John Lewis Christmas ad",
            "John Lewis Christmas advert",
            "John Lewis ad 2025",
            "where love lives John Lewis"
        ]
        
        all_posts = []
        
        for query in search_queries:
            print(f"Searching for: '{query}'")
            posts_df = self.search_posts(query, limit=limit//len(search_queries))
            if not posts_df.empty:
                all_posts.append(posts_df)
            
            time.sleep(1)  # Rate limiting between searches
        
        if all_posts:
            return pd.concat(all_posts, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """Save scraped data to CSV file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"reddit_data_{timestamp}.csv"
        
        filepath = f"data/{filename}"
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Reddit data saved to: {filepath}")
        return filepath

def main():
    """
    Example usage for scraping Reddit discussions about John Lewis Christmas ads.
    """
    scraper = RedditScraper()
    
    # Search for John Lewis Christmas ad discussions
    posts_df = scraper.search_john_lewis_posts(limit=30)
    
    if not posts_df.empty:
        # Save posts
        saved_path = scraper.save_data(posts_df)
        
        # Display summary
        print(f"\nReddit Scraping Summary:")
        print(f"Total posts: {len(posts_df)}")
        print(f"Unique subreddits: {posts_df['subreddit'].nunique()}")
        print(f"Average score: {posts_df['score'].mean():.2f}")
        print(f"Date range: {posts_df['created_utc'].min()} to {posts_df['created_utc'].max()}")
        
        # Display sample posts
        print(f"\nSample posts:")
        print(posts_df[['text', 'author', 'subreddit', 'score']].head())
    else:
        print("No Reddit posts were found.")

if __name__ == "__main__":
    main()
