
import re
from datetime import datetime, timedelta
import json

class Utils:
    @staticmethod
    def format_number(num):
        """Format large numbers for display (e.g., 1500000 -> 1.5M)"""
        if num >= 1000000000:
            return f"{num/1000000000:.1f}B"
        elif num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}K"
        else:
            return str(num)
    
    @staticmethod
    def parse_duration_to_seconds(duration):
        """Parse YouTube ISO 8601 duration to seconds"""
        if not duration:
            return 0
            
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        
        if not match:
            return 0
        
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        
        return hours * 3600 + minutes * 60 + seconds
    
    @staticmethod
    def seconds_to_duration_string(seconds):
        """Convert seconds to human readable duration"""
        if seconds < 60:
            return f"{seconds}s"
        elif seconds < 3600:
            minutes = seconds // 60
            secs = seconds % 60
            return f"{minutes}m {secs}s" if secs > 0 else f"{minutes}m"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            return f"{hours}h {minutes}m" if minutes > 0 else f"{hours}h"
    
    @staticmethod
    def clean_text(text):
        """Clean text for safe display"""
        if not text:
            return ""
        
        # Remove or replace problematic characters
        text = re.sub(r'[^\w\s\-\.,\!\?\(\)\[\]\'\"]+', '', text)
        text = text.strip()
        
        # Truncate if too long
        if len(text) > 100:
            text = text[:97] + "..."
        
        return text
    
    @staticmethod
    def validate_multiplier_range(min_val, max_val):
        """Validate multiplier range inputs"""
        try:
            min_mult = float(min_val) if min_val else 1.0
            max_mult = float(max_val) if max_val else 500.0
            
            if min_mult < 0:
                min_mult = 1.0
            if max_mult < min_mult:
                max_mult = min_mult + 1
            if max_mult > 1000:
                max_mult = 1000.0
                
            return min_mult, max_mult
        except (ValueError, TypeError):
            return 1.0, 500.0
    
    @staticmethod
    def validate_views_range(min_val, max_val):
        """Validate views range inputs"""
        try:
            min_views = int(min_val) if min_val else 0
            max_views = int(max_val) if max_val else float('inf')
            
            if min_views < 0:
                min_views = 0
            if max_views < min_views:
                max_views = min_views + 1
                
            return min_views, max_views
        except (ValueError, TypeError):
            return 0, float('inf')
    
    @staticmethod
    def is_valid_youtube_video_id(video_id):
        """Check if a string is a valid YouTube video ID"""
        if not video_id or len(video_id) != 11:
            return False
        
        # YouTube video IDs are 11 characters long and contain only alphanumeric characters, hyphens, and underscores
        pattern = r'^[a-zA-Z0-9_-]{11}$'
        return bool(re.match(pattern, video_id))
    
    @staticmethod
    def extract_video_id_from_url(url):
        """Extract video ID from YouTube URL"""
        if not url:
            return None
            
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        return None
    
    @staticmethod
    def calculate_engagement_rate(likes, comments, views):
        """Calculate engagement rate as percentage"""
        if views == 0:
            return 0
        
        total_engagement = likes + comments
        return (total_engagement / views) * 100
    
    @staticmethod
    def get_video_age_days(published_at):
        """Calculate video age in days"""
        try:
            # Parse YouTube's datetime format
            pub_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
            now = datetime.utcnow()
            age = now - pub_date
            return age.days
        except (ValueError, TypeError):
            return 0
    
    @staticmethod
    def is_recent_video(published_at, days=30):
        """Check if video was published within specified days"""
        age_days = Utils.get_video_age_days(published_at)
        return age_days <= days
    
    @staticmethod
    def export_to_json(data, filename=None):
        """Export data to JSON format"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'youtube_outliers_{timestamp}.json'
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return filename
        except Exception as e:
            print(f"Error exporting to JSON: {e}")
            return None
    
    @staticmethod
    def sanitize_filename(filename):
        """Sanitize filename for safe file operations"""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = filename.strip()
        
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        
        return filename or 'unnamed_file'
    
    @staticmethod
    def batch_process(items, batch_size=50):
        """Split items into batches for processing"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]
    
    @staticmethod
    def safe_divide(numerator, denominator, default=0):
        """Safely divide two numbers, return default if denominator is 0"""
        try:
            if denominator == 0:
                return default
            return numerator / denominator
        except (TypeError, ValueError):
            return default
