from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from googleapiclient.discovery import build
import logging
import json
import os
from googleapiclient.errors import HttpError
import re
from youtube_thumbnail_similarity import find_similar_videos_enhanced
import socket
from datetime import datetime, timedelta
import requests
from PIL import Image
import numpy as np
from io import BytesIO
import time
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import random

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for front-end requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# Load API keys from environment variables
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'your_youtube_api_key')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key')
ACCESS_TOKEN = os.getenv('TRENDING_ACCESS_TOKEN', 'your_secure_token_123')

# Configure Gemini client
genai.configure(api_key=GEMINI_API_KEY)

# In-memory storage for search history (no caching for videos or channels)
search_history = []

# --- Helper Functions ---

def check_internet(host="youtube.googleapis.com", port=443, timeout=5):
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except OSError as e:
        app.logger.error(f"Internet connectivity check failed: {str(e)}")
        return False

def parse_duration(duration):
    """Parse ISO 8601 duration to seconds."""
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    return 0

def search_videos(query, max_results=50, region_code='US', published_after=None):
    """Search YouTube videos using the API."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            part='id,snippet',
            q=query,
            type='video',
            maxResults=min(max_results, 50),
            order='viewCount',
            regionCode=region_code,
            publishedAfter=published_after
        )
        response = request.execute()
        video_ids = [item['id']['videoId'] for item in response['items']]
        # Fetch additional video details including contentDetails
        videos = []
        if video_ids:
            video_details = get_video_details_batch(video_ids)
            for item in response['items']:
                video_id = item['id']['videoId']
                details = video_details.get(video_id, {})
                videos.append({
                    'video_id': video_id,
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'language': item['snippet'].get('defaultLanguage', 'en'),
                    'views': details.get('views', 0),
                    'likes': details.get('likes', 0),
                    'comments': details.get('comments', 0),
                    'duration': details.get('duration', 'PT0S')
                })
        app.logger.debug(f"Fetched {len(videos)} videos for query: {query}, region: {region_code}")
        return videos
    except Exception as e:
        app.logger.error(f"YouTube search error: {str(e)}")
        return []

def get_video_details_batch(video_ids):
    """Get details for a batch of video IDs including statistics and contentDetails."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    details = {}
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        try:
            request = youtube.videos().list(
                part='statistics,contentDetails',
                id=','.join(batch_ids)
            )
            response = request.execute()
            for item in response['items']:
                video_id = item['id']
                details[video_id] = {
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'comments': int(item['statistics'].get('commentCount', 0)),
                    'duration': item['contentDetails']['duration']
                }
        except Exception as e:
            app.logger.error(f"YouTube video details error: {str(e)}")
    return details

def get_video_stats(video_ids):
    """Get statistics for a list of video IDs."""
    return get_video_details_batch(video_ids)

def calculate_channel_average_views(channel_id):
    """Calculate average views and subscriber count for a channel."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        channels_request = youtube.channels().list(
            part='contentDetails,statistics',
            id=channel_id
        )
        channels_response = channels_request.execute()
        if not channels_response['items']:
            return 0, 0
        
        uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        subscriber_count = int(channels_response['items'][0]['statistics'].get('subscriberCount', 0))
        
        playlist_request = youtube.playlistItems().list(
            part='contentDetails',
            playlistId=uploads_playlist_id,
            maxResults=50
        )
        playlist_response = playlist_request.execute()
        video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
        
        video_stats = get_video_stats(video_ids)
        total_views = sum(stats['views'] for stats in video_stats.values())
        avg_views = total_views / len(video_stats) if video_stats else 0
        
        app.logger.debug(f"Calculated avg views {avg_views} and subscribers {subscriber_count} for channel {channel_id}")
        return avg_views, subscriber_count
    except Exception as e:
        app.logger.error(f"Channel avg views error: {str(e)}")
        return 0, 0

def get_random_trending_videos(max_results=200):
    """Fetch random trending videos using varied queries, regions, and dates."""
    if not check_internet():
        app.logger.error("No internet connection to youtube.googleapis.com")
        return []
    
    # Define a list of broad, generic queries to simulate trending topics
    trending_queries = [
        "trending", "popular", "viral", "new video", "latest", "hot", "buzz", "top videos",
        "music", "gaming", "vlog", "tutorial", "review", "reaction", "challenge"
    ]
    
    # Randomly select a query
    query = random.choice(trending_queries)
    
    # Randomly select a region
    regions = ['US', 'GB', 'IN', 'CA', 'AU', 'DE', 'FR']
    region_code = random.choice(regions)
    
    # Randomize publishedAfter to get recent videos from different time windows
    days_back = random.randint(1, 30)  # Videos from the last 1-30 days
    published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat("T") + "Z"
    
    try:
        videos = search_videos(
            query=query,
            max_results=max_results,
            region_code=region_code,
            published_after=published_after
        )
        # Shuffle videos to ensure random order
        random.shuffle(videos)
        app.logger.debug(f"Fetched {len(videos)} random trending videos for query: {query}, region: {region_code}, published after: {published_after}")
        return videos
    except Exception as e:
        app.logger.error(f"Error fetching random trending videos: {str(e)}")
        return []

def get_top_videos(channel_id, n=5):
    """Get top videos for a channel."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            part='id,snippet',
            channelId=channel_id,
            order='viewCount',
            type='video',
            maxResults=n
        )
        response = request.execute()
        videos = [
            {
                'video_id': item['id']['videoId'],
                'title': item['snippet']['title']
            } for item in response['items']
        ]
        app.logger.debug(f"Fetched {len(videos)} top videos for channel {channel_id}")
        return videos
    except Exception as e:
        app.logger.error(f"Error fetching top videos for channel {channel_id}: {str(e)}")
        return []

def get_related_videos(video_id, m=15):
    """Get related videos."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
    try:
        request = youtube.search().list(
            part='snippet',
            relatedToVideoId=video_id,
            type='video',
            maxResults=m
        )
        response = request.execute()
        related_videos = []
        for item in response.get('items', []):
            if 'id' in item and 'videoId' in item['id'] and 'snippet' in item:
                channel_id = item['snippet'].get('channelId')
                if not channel_id:
                    app.logger.debug(f"Skipping related video {item['id']['videoId']}: missing channelId")
                    continue
                related_videos.append({
                    'video_id': item['id']['videoId'],
                    'channel_id': channel_id,
                    'channel_title': item['snippet'].get('channelTitle', 'Unknown Channel'),
                    'title': item['snippet'].get('title', 'Unknown Title'),
                    'published_at': item['snippet'].get('publishedAt', ''),
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url']
                })
        app.logger.debug(f"Fetched {len(related_videos)} related videos for video {video_id}")
        return related_videos
    except HttpError as e:
        app.logger.error(f"YouTube API error in get_related_videos for video {video_id}: {str(e)}")
        return []
    except Exception as e:
        app.logger.error(f"Unexpected error in get_related_videos for video {video_id}: {str(e)}")
        return []
def search_videos_by_query(query, max_results=15):
    """Search videos by query."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=max_results,
            order='viewCount'
        )
        response = request.execute()
        videos = []
        for item in response['items']:
            videos.append({
                'video_id': item['id']['videoId'],
                'channel_id': item['snippet']['channelId'],
                'channel_title': item['snippet']['channelTitle'],
                'title': item['snippet']['title'],
                'published_at': item['snippet']['publishedAt'],
                'thumbnail_url': item['snippet']['thumbnails']['high']['url']
            })
        app.logger.debug(f"Fetched {len(videos)} videos for query '{query}'")
        return videos
    except Exception as e:
        app.logger.error(f"Error searching videos for query '{query}': {str(e)}")
        return []

def analyze_videos_for_outliers(videos):
    """Analyze videos for outliers."""
    video_ids = [v['video_id'] for v in videos if 'video_id' in v]
    video_stats = get_video_stats(video_ids)
    channel_ids = list(set(v['channel_id'] for v in videos if 'channel_id' in v))
    processed_channels = {}
    for channel_id in channel_ids:
        avg_views, subscriber_count = calculate_channel_average_views(channel_id)
        processed_channels[channel_id] = {'average_views': avg_views, 'subscriber_count': subscriber_count}
    outlier_videos = []
    for video in videos:
        video_id = video.get('video_id')
        channel_id = video.get('channel_id')
        if not video_id or not channel_id:
            app.logger.debug(f"Skipping video: missing video_id or channel_id - {video}")
            continue
        if video_id not in video_stats or channel_id not in processed_channels:
            app.logger.debug(f"Skipping video {video_id}: missing stats or channel data")
            continue
        stats = video_stats[video_id]
        channel_avg = processed_channels[channel_id]['average_views']
        if channel_avg == 0:
            app.logger.debug(f"Skipping video {video_id}: channel average views is 0")
            continue
        multiplier = stats['views'] / channel_avg
        if multiplier > 5:
            video_result = {
                'video_id': video_id,
                'title': video.get('title', 'Unknown Title'),
                'channel_id': channel_id,
                'channel_title': video.get('channel_title', 'Unknown Channel'),
                'views': stats['views'],
                'channel_avg_views': channel_avg,
                'multiplier': round(multiplier, 2),
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'thumbnail_url': video.get('thumbnail_url', '')
            }
            outlier_videos.append(video_result)
    app.logger.debug(f"Found {len(outlier_videos)} outliers in analysis")
    return outlier_videos

# --- Outlier Detection Logic ---

class OutlierDetector:
    def __init__(self):
        self.api_calls_count = 0
        self.max_api_calls = 9000
        self.global_search_params = {
            'regions': ['US', 'IN', 'GB', 'CA'],
            'languages': ['en', 'hi', 'es'],
            'content_types': ['video']
        }

    def detect_outliers(self, query, filters=None):
        """Detect outlier videos with global analysis."""
        if filters is None:
            filters = {}
        
        app.logger.debug(f"Starting search for: {query}")
        
        related_queries = self._generate_related_queries(query)
        all_videos = []
        
        videos = self._get_videos_for_query(query, enhanced=True)
        all_videos.extend(videos)
        
        for related_query in related_queries[:3]:
            if self._check_api_quota():
                related_videos = self._get_videos_for_query(related_query, enhanced=True)
                all_videos.extend(related_videos)
            else:
                app.logger.warning("API quota limit approaching")
                break
        
        unique_videos = self._remove_duplicate_videos(all_videos)
        if not unique_videos:
            return []
        
        outlier_videos = self._analyze_videos_advanced(unique_videos, query, filters)
        scored_videos = self._calculate_advanced_scores(outlier_videos)
        final_results = self._rank_videos_multi_criteria(scored_videos)
        
        search_history.append({
            'query': query,
            'results_count': len(final_results),
            'filters': filters,
            'related_queries': related_queries,
            'total_videos_analyzed': len(unique_videos),
            'api_calls_used': self.api_calls_count
        })
        
        app.logger.debug(f"Detected {len(final_results)} outliers for query: {query}")
        return final_results

    def _generate_related_queries(self, original_query):
        """Generate related search queries."""
        key_terms = re.findall(r'\b\w+', original_query.lower())
        variations = [
            f"{original_query} explained",
            f"{original_query} review",
            f"{original_query} facts"
        ]
        for term in key_terms[:2]:
            if len(term) > 3:
                variations.append(term)
        return [v for v in variations if v.lower() != original_query.lower()][:3]

    def _get_videos_for_query(self, query, enhanced=False):
        """Retrieve videos for a query without caching."""
        if not self._check_api_quota():
            return []
        
        videos = []
        if enhanced:
            for region in self.global_search_params['regions'][:2]:
                if not self._check_api_quota():
                    break
                region_videos = search_videos(query, max_results=25, region_code=region)
                videos.extend(region_videos)
                self.api_calls_count += 1
                time.sleep(0.1)
        else:
            videos = search_videos(query, max_results=50)
            self.api_calls_count += 1
        
        return videos

    def _remove_duplicate_videos(self, videos):
        """Remove duplicate videos by video_id."""
        seen_ids = set()
        unique_videos = []
        for video in videos:
            if video['video_id'] not in seen_ids:
                seen_ids.add(video['video_id'])
                unique_videos.append(video)
        return unique_videos

    def _analyze_videos_advanced(self, videos, original_query, filters):
        """Analyze videos with advanced metrics."""
        if not self._check_api_quota():
            return []
        
        video_ids = [v['video_id'] for v in videos]
        video_stats = get_video_stats(video_ids)
        self.api_calls_count += len(video_ids) // 50 + 1
        
        channel_ids = list(set(v['channel_id'] for v in videos))
        processed_channels = {}
        for channel_id in channel_ids:
            if self._check_api_quota():
                avg_views, subscriber_count = calculate_channel_average_views(channel_id)
                processed_channels[channel_id] = {'average_views': avg_views, 'subscriber_count': subscriber_count}
        
        outlier_videos = []
        for video in videos:
            video_id = video['video_id']
            channel_id = video['channel_id']
            if video_id not in video_stats or channel_id not in processed_channels:
                continue
            
            stats = video_stats[video_id]
            channel_data = processed_channels[channel_id]
            channel_avg = channel_data['average_views']
            if channel_avg == 0:
                continue
            
            multiplier = stats['views'] / channel_avg
            engagement_rate = self._calculate_engagement_rate(stats)
            viral_score = self._calculate_viral_score(stats, video)
            relevance_score = self._calculate_relevance_score(video, original_query)
            
            video_result = {
                'video_id': video_id,
                'title': video['title'],
                'channel_title': video['channel_title'],
                'channel_id': channel_id,
                'views': stats['views'],
                'channel_avg_views': channel_avg,
                'multiplier': round(multiplier, 2),
                'likes': stats['likes'],
                'comments': stats['comments'],
                'duration': stats['duration'],
                'duration_seconds': parse_duration(stats['duration']),
                'published_at': video['published_at'],
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'thumbnail_url': video['thumbnail_url'],
                'engagement_rate': round(engagement_rate, 4),
                'viral_score': round(viral_score, 2),
                'relevance_score': round(relevance_score, 2),
                'video_age_days': self._get_video_age_days(video['published_at']),
                'subscriber_count': channel_data.get('subscriber_count', 0)
            }
            
            if self._passes_enhanced_filters(video_result, filters):
                outlier_videos.append(video_result)
        
        return outlier_videos

    def _calculate_engagement_rate(self, stats):
        """Calculate engagement rate."""
        return (stats['likes'] + stats['comments']) / stats['views'] if stats['views'] > 0 else 0

    def _calculate_viral_score(self, stats, video):
        """Calculate viral potential score."""
        base_score = 0
        if stats['views'] > 0:
            base_score += min(10, (stats['views'] / 1000000) * 2)
        if stats['views'] > 0:
            engagement = (stats['likes'] + stats['comments']) / stats['views']
            base_score += min(5, engagement * 1000)
        age_days = self._get_video_age_days(video['published_at'])
        if age_days <= 7:
            base_score += 3
        elif age_days <= 30:
            base_score += 1
        return base_score

    def _calculate_relevance_score(self, video, original_query):
        """Calculate relevance score."""
        title = video['title'].lower()
        query_terms = original_query.lower().split()
        matches = sum(1 for term in query_terms if term in title)
        return (matches / len(query_terms)) * 10 if query_terms else 0

    def _get_video_age_days(self, published_at):
        """Calculate video age in days."""
        try:
            pub_date = datetime.strptime(published_at, '%Y-%m-%dT%H:%M:%SZ')
            return (datetime.utcnow() - pub_date).days
        except:
            return 0

    def _calculate_advanced_scores(self, videos):
        """Calculate composite scores."""
        for video in videos:
            composite_score = (
                video['multiplier'] * 0.4 +
                video['viral_score'] * 0.25 +
                video['engagement_rate'] * 1000 * 0.2 +
                video['relevance_score'] * 0.15
            )
            video['composite_score'] = round(composite_score, 2)
        return videos

    def _rank_videos_multi_criteria(self, videos):
        """Rank videos using multiple criteria."""
        videos.sort(key=lambda x: (x['composite_score'], x['multiplier']), reverse=True)
        for i, video in enumerate(videos):
            video['rank'] = i + 1
            video['performance_tier'] = self._get_performance_tier(video['multiplier'])
        return videos

    def _get_performance_tier(self, multiplier):
        """Categorize video performance."""
        if multiplier >= 50:
            return "Mega Viral"
        elif multiplier >= 20:
            return "Super Viral"
        elif multiplier >= 10:
            return "Highly Viral"
        elif multiplier >= 5:
            return "Viral"
        elif multiplier >= 2:
            return "Above Average"
        else:
            return "Average"

    def _passes_enhanced_filters(self, video, filters):
        """Check if video passes enhanced filters."""
        min_multiplier = filters.get('multiplier', 1.0)
        if video['multiplier'] < min_multiplier:
            return False
        min_views = filters.get('views', 0)
        if video['views'] < min_views:
            return False
        min_duration = filters.get('duration', 0)
        if video['duration_seconds'] < min_duration:
            return False
        min_subscribers = filters.get('subscribers')
        if min_subscribers and video['subscriber_count'] < min_subscribers:
            return False
        return True

    def _check_api_quota(self):
        """Check API quota."""
        return self.api_calls_count < self.max_api_calls

    def get_search_statistics(self):
        """Get search statistics."""
        return {
            'popular_searches': [h['query'] for h in search_history[-10:]],
            'total_searches': len(search_history)
        }

    def format_number(self, num):
        """Format large numbers for display."""
        if num >= 1000000:
            return f"{num/1000000:.1f}M"
        elif num >= 1000:
            return f"{num/1000:.1f}K"
        else:
            return str(num)

# --- Thumbnail Similarity Logic ---

class UltraFastYouTubeSimilarity:
    def __init__(self):
        self._session = requests.Session()
        self._batch_size = 64
        self._max_workers = 16

    def _get_video_durations_batch(self, video_ids):
        """Get durations for multiple videos."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        duration_map = {}
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            try:
                request = youtube.videos().list(
                    part='contentDetails',
                    id=','.join(batch_ids)
                )
                response = request.execute()
                for item in response['items']:
                    video_id = item['id']
                    duration = item['contentDetails']['duration']
                    duration_map[video_id] = parse_duration(duration)
            except Exception as e:
                app.logger.error(f"Duration batch error: {str(e)}")
                for video_id in batch_ids:
                    duration_map[video_id] = 300
        return duration_map

    def _filter_shorts_strict(self, videos, max_duration=60):
        """Filter out YouTube Shorts."""
        video_ids = [video[0] for video in videos]
        duration_map = self._get_video_durations_batch(video_ids)
        filtered_videos = [v for v in videos if duration_map.get(v[0], 0) > max_duration]
        return filtered_videos

    def _get_channel_videos_no_shorts(self, channel_id, max_results=50):
        """Get channel videos, excluding Shorts."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        videos = []
        try:
            channels_request = youtube.channels().list(
                part='contentDetails',
                id=channel_id
            )
            channels_response = channels_request.execute()
            if not channels_response['items']:
                return videos
            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            playlist_request = youtube.playlistItems().list(
                part='snippet',
                playlistId=uploads_playlist_id,
                maxResults=50
            )
            playlist_response = playlist_request.execute()
            raw_videos = [
                [item['snippet']['resourceId']['videoId'], item['snippet']['title'], item['snippet']['thumbnails']['high']['url']]
                for item in playlist_response['items']
            ]
            videos = self._filter_shorts_strict(raw_videos)
            app.logger.debug(f"Fetched {len(videos)} channel videos for {channel_id}")
            return videos[:max_results]
        except Exception as e:
            app.logger.error(f"Channel videos error: {str(e)}")
            return []

    def _search_videos_no_shorts(self, query, max_results):
        """Search videos, excluding Shorts."""
        videos = search_videos(query, max_results * 2)
        raw_videos = [[v['video_id'], v['title'], v['thumbnail_url']] for v in videos]
        return self._filter_shorts_strict(raw_videos)[:max_results]

    def _get_smart_related_videos_no_shorts(self, video_title, input_video_id, channel_id=None, max_results=200):
        """Fetch related videos, excluding Shorts."""
        all_videos = []
        video_ids_seen = {input_video_id}
        if channel_id:
            channel_videos = self._get_channel_videos_no_shorts(channel_id, max_results=50)
            for video in channel_videos:
                if video[0] not in video_ids_seen:
                    video_ids_seen.add(video[0])
                    all_videos.append(video)
        
        remaining_results = max_results - len(all_videos)
        if remaining_results > 0:
            search_strategies = [
                ' '.join([w for w in video_title.split() if len(w) > 3 and w.lower() not in ['the', 'and', 'video']])[:30],
                next((w for w in video_title.split() if len(w) > 4), video_title.split()[0] if video_title.split() else '')
            ]
            search_strategies = [s.strip() for s in search_strategies if s.strip()]
            for query in search_strategies[:2]:
                if len(all_videos) >= max_results:
                    break
                videos = self._search_videos_no_shorts(query, remaining_results // 2)
                for video in videos:
                    if video[0] not in video_ids_seen and len(all_videos) < max_results:
                        video_ids_seen.add(video[0])
                        all_videos.append(video)
        app.logger.debug(f"Fetched {len(all_videos)} related videos")
        return all_videos

    def _preprocess_image_ultra_fast(self, image_url):
        """Preprocess image for similarity."""
        try:
            if image_url.startswith('http'):
                response = self._session.get(image_url, timeout=5)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
            else:
                img = Image.open(image_url)
            img = img.convert('RGB').resize((224, 224), Image.Resampling.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - 0.5) * 2.0
            img_array = img_array.flatten()
            return img_array
        except Exception as e:
            app.logger.error(f"Image processing error: {str(e)}")
            return None

    def _extract_features_batch(self, image_urls):
        """Extract features for a batch of images."""
        features = []
        valid_urls = []
        for url in image_urls:
            img_array = self._preprocess_image_ultra_fast(url)
            if img_array is not None:
                features.append(img_array)
                valid_urls.append(url)
        return np.array(features), valid_urls

    def _compute_pure_visual_similarity(self, query_features, database_features):
        """Compute visual similarity."""
        if len(query_features) == 0 or len(database_features) == 0:
            return []
        query_features = np.array(query_features).reshape(1, -1)
        database_features = np.array(database_features)
        return cosine_similarity(query_features, database_features)[0]

def extract_video_id(url):
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'youtu\.be\/([0-9A-Za-z_-]{11})',
        r'embed\/([0-9A-Za-z_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    raise ValueError("Invalid YouTube URL format")

def get_video_details(video_id):
    """Get video details."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        if response['items']:
            snippet = response['items'][0]['snippet']
            return snippet['title'], snippet['thumbnails']['high']['url'], snippet['channelId']
        raise ValueError(f"Video {video_id} not found")
    except Exception as e:
        raise ValueError(f"Failed to get video details: {str(e)}")

def find_similar_videos_enhanced(input_url, max_results=150, top_similar=50, similarity_threshold=0.70):
    """Find similar videos based on thumbnails."""
    similarity_finder = UltraFastYouTubeSimilarity()
    try:
        input_video_id = extract_video_id(input_url)
        input_title, input_thumbnail_url, channel_id = get_video_details(input_video_id)
        
        related_videos = similarity_finder._get_smart_related_videos_no_shorts(
            input_title, input_video_id, channel_id, max_results
        )
        if not related_videos:
            return None, None, "No related videos found."
        
        input_features, _ = similarity_finder._extract_features_batch([input_thumbnail_url])
        if len(input_features) == 0:
            return None, None, "Failed to process input video thumbnail."
        
        related_thumbnail_urls = [video[2] for video in related_videos]
        related_features, valid_thumbnail_urls = similarity_finder._extract_features_batch(related_thumbnail_urls)
        if len(related_features) == 0:
            return None, None, "Failed to process related video thumbnails."
        
        similarities = similarity_finder._compute_pure_visual_similarity(input_features[0], related_features)
        url_to_video = {video[2]: video for video in related_videos}
        
        similar_videos_data = []
        for thumbnail_url, similarity_score in zip(valid_thumbnail_urls, similarities):
            if similarity_score >= similarity_threshold and thumbnail_url in url_to_video:
                video_data = url_to_video[thumbnail_url]
                similar_videos_data.append({
                    "video_id": video_data[0],
                    "title": video_data[1],
                    "thumbnail_url": video_data[2],
                    "video_url": f"https://www.youtube.com/watch?v={video_data[0]}",
                    "visual_similarity": float(similarity_score)
                })
        
        similar_videos_data.sort(key=lambda x: x['visual_similarity'], reverse=True)
        similar_videos_data = similar_videos_data[:top_similar]
        
        result_data = {
            "input_video": {
                "title": input_title,
                "video_url": input_url,
                "thumbnail_url": input_thumbnail_url,
                "channel_id": channel_id
            },
            "similar_videos": similar_videos_data,
            "processing_stats": {
                "total_videos_analyzed": len(related_videos),
                "thumbnails_processed": len(related_features),
                "processing_time": round(time.time() - time.time(), 2),
                "average_visual_similarity": round(np.mean([v['visual_similarity'] for v in similar_videos_data]), 3) if similar_videos_data else 0
            }
        }
        app.logger.debug(f"Found {len(similar_videos_data)} similar videos")
        return result_data, input_title, None
    except Exception as e:
        app.logger.error(f"Similarity analysis error: {str(e)}")
        return None, None, f"Analysis failed: {str(e)}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Render the main page with the filter interface."""
    return render_template('index.html')

@app.route('/api/search', methods=['POST'])
def search_outliers():
    """Search for outlier videos."""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'Query parameter is required'}), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        filters = data.get('filters', {})
        filters.setdefault('multiplier', 150.0)
        filters.setdefault('views', 150.0)
        filters.setdefault('subscribers', 150.0)
        filters.setdefault('duration', 150.0)
        filters.setdefault('publication_date', 'All Time')
        filters.setdefault('language', 'English')
        
        outlier_detector = OutlierDetector()
        outliers = outlier_detector.detect_outliers(query, filters)
        
        formatted_outliers = []
        for video in outliers:
            formatted_video = {
                'video_id': video['video_id'],
                'title': video['title'],
                'channel_title': video['channel_title'],
                'views': video['views'],
                'views_formatted': outlier_detector.format_number(video['views']),
                'channel_avg_views': video['channel_avg_views'],
                'channel_avg_views_formatted': outlier_detector.format_number(video['channel_avg_views']),
                'multiplier': video['multiplier'],
                'likes': video['likes'],
                'likes_formatted': outlier_detector.format_number(video['likes']),
                'comments': video['comments'],
                'comments_formatted': outlier_detector.format_number(video['comments']),
                'duration': video['duration'],
                'duration_seconds': video['duration_seconds'],
                'published_at': video['published_at'],
                'rank': video.get('rank'),
                'performance_tier': video.get('performance_tier'),
                'composite_score': video.get('composite_score'),
                'viral_score': video.get('viral_score'),
                'engagement_rate': video['engagement_rate'],
                'relevance_score': video['relevance_score'],
                'video_age_days': video.get('video_age_days'),
                'thumbnail_url': video['thumbnail_url'],
                'subscriber_count': video['subscriber_count']
            }
            formatted_outliers.append(formatted_video)
        
        return jsonify({
            'success': True,
            'query': query,
            'total_results': len(formatted_outliers),
            'outliers': formatted_outliers,
            'filters_applied': filters
        })
    except Exception as e:
        app.logger.error(f"Search outliers error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/trending_outliers', methods=['POST'])
def trending_outliers():
    """Fetch random trending outlier videos without caching, excluding Shorts."""
    try:
        request_id = str(time.time())
        app.logger.debug(f"Request ID: {request_id}")
        
        data = request.get_json()
        if not data or 'access_token' not in data:
            return jsonify({'error': 'Access token is required'}), 400
        
        access_token = data['access_token'].strip()
        if access_token != ACCESS_TOKEN:
            return jsonify({'error': 'Invalid access token'}), 401
        
        if not YOUTUBE_API_KEY:
            app.logger.error("YOUTUBE_API_KEY is not set")
            return jsonify({'error': 'YouTube API key is not configured'}), 500
        
        # Fetch trending videos
        trending_videos = get_random_trending_videos(max_results=200)
        if not trending_videos:
            app.logger.warning("No trending videos retrieved, check internet or API key")
            return jsonify({'error': 'No trending videos found, check internet connection or API key'}), 503
        
        # Filter out Shorts
        similarity_finder = UltraFastYouTubeSimilarity()
        raw_videos = [[v['video_id'], v['title'], v['thumbnail_url']] for v in trending_videos]
        filtered_videos = similarity_finder._filter_shorts_strict(raw_videos, max_duration=60)
        video_id_to_data = {v['video_id']: v for v in trending_videos}
        trending_videos = [video_id_to_data[video[0]] for video in filtered_videos if video[0] in video_id_to_data]
        
        app.logger.debug(f"Fetched {len(trending_videos)} trending videos after filtering out Shorts")
        
        # Process videos for outliers
        outlier_detector = OutlierDetector()
        video_ids = [v['video_id'] for v in trending_videos]
        video_stats = get_video_stats(video_ids)
        channel_ids = list(set(v['channel_id'] for v in trending_videos))
        processed_channels = {}
        for channel_id in channel_ids:
            avg_views, subscriber_count = calculate_channel_average_views(channel_id)
            processed_channels[channel_id] = {'average_views': avg_views, 'subscriber_count': subscriber_count}
        
        outliers = []
        for video in trending_videos:
            video_id = video['video_id']
            channel_id = video['channel_id']
            if video_id not in video_stats or channel_id not in processed_channels:
                app.logger.debug(f"Skipping video {video_id}: missing stats or channel data")
                continue
            stats = video_stats[video_id]
            channel_data = processed_channels[channel_id]
            channel_avg = channel_data['average_views']
            if channel_avg == 0:
                app.logger.debug(f"Skipping video {video_id}: channel average views is 0")
                continue
            multiplier = stats['views'] / channel_avg
            
            if multiplier <= 0.5:  # Low threshold to maximize results
                app.logger.debug(f"Skipping video {video_id}: multiplier {multiplier} <= 0.5")
                continue
                
            duration_seconds = parse_duration(stats.get('duration', 'PT0S'))
            video_result = {
                'video_id': video_id,
                'title': video['title'],
                'channel_id': channel_id,  # Added
                'channel_title': video['channel_title'],
                'views': stats['views'],
                'channel_avg_views': channel_avg,
                'multiplier': round(multiplier, 2),
                'likes': stats['likes'],
                'comments': stats['comments'],
                'duration': stats['duration'],
                'duration_seconds': duration_seconds,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'published_at': video['published_at'],
                'thumbnail_url': video['thumbnail_url'],
                'viral_score': multiplier / 10,
                'engagement_rate': (stats['likes'] + stats['comments']) / stats['views'] if stats['views'] > 0 else 0,
                'subscriber_count': channel_data.get('subscriber_count', 0),
                'language': video.get('language', 'en')
            }
            outliers.append(video_result)
        
        app.logger.debug(f"Outliers before limiting: {len(outliers)}")
        random.shuffle(outliers)  # Shuffle outliers for randomness
        outliers = outliers[:200]  # Return up to 200 results
        
        formatted_outliers = []
        for video in outliers:
            formatted_video = {
                'video_id': video['video_id'],
                'title': video['title'],
                'channel_id': video['channel_id'],  # Added
                'channel_title': video['channel_title'],
                'views': video['views'],
                'views_formatted': outlier_detector.format_number(video['views']),
                'channel_avg_views': video['channel_avg_views'],
                'channel_avg_views_formatted': outlier_detector.format_number(video['channel_avg_views']),
                'multiplier': video['multiplier'],
                'likes': video['likes'],
                'likes_formatted': outlier_detector.format_number(video['likes']),
                'comments': video['comments'],
                'comments_formatted': outlier_detector.format_number(video['comments']),
                'duration': video['duration'],
                'duration_seconds': video['duration_seconds'],
                'url': video['url'],
                'published_at': video['published_at'],
                'viral_score': round(video['viral_score'], 2),
                'engagement_rate': round(video['engagement_rate'], 4),
                'thumbnail_url': video['thumbnail_url'],
                'subscriber_count': video['subscriber_count'],
                'language': video['language']
            }
            formatted_outliers.append(formatted_video)
        
        search_history.append({
            'query': 'trending_outliers',
            'timestamp': datetime.now().isoformat(),
            'results_count': len(formatted_outliers)
        })
        
        app.logger.info(f"Found {len(formatted_outliers)} random trending outliers (Shorts excluded)")
        return jsonify({
            'success': True,
            'total_results': len(formatted_outliers),
            'outliers': formatted_outliers
        })
    except Exception as e:
        app.logger.error(f"Trending outliers error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500
        

@app.route('/api/similar-thumbnails', methods=['POST'])
def get_similar_thumbnails():
    """API endpoint to get similar thumbnails for a given video ID"""
    try:
        data = request.get_json()
        video_id = data.get('video_id')
        if not video_id:
            return jsonify({'error': 'Video ID is required'}), 400

        video_url = f"https://www.youtube.com/watch?v={video_id}"
        result_data, _, error = find_similar_videos_enhanced(video_url)
        
        if error:
            return jsonify({'success': False, 'error': error}), 400

        return jsonify({'success': True, **result_data})
    
    except Exception as e:
        print(f"Error in get_similar_thumbnails: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get search statistics."""
    try:
        outlier_detector = OutlierDetector()
        stats = outlier_detector.get_search_statistics()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        app.logger.error(f"Stats error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'YouTube Title Generation and Outlier Detection Tool',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')
    })

@app.route('/api/channel_outliers_by_id', methods=['POST'])
def channel_outliers_by_id():
    """Find outlier videos from related channels."""
    try:
        data = request.get_json()
        if not data or 'channel_id' not in data:
            return jsonify({'error': 'Channel ID is required'}), 400
        channel_id = data['channel_id'].strip()
        if not channel_id:
            return jsonify({'error': 'Channel ID cannot be empty'}), 400
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        channel_request = youtube.channels().list(
            part='id,snippet',
            id=channel_id
        )
        response = channel_request.execute()
        if not response['items']:
            return jsonify({'error': 'Channel not found'}), 404
        channel_title = response['items'][0]['snippet']['title']
        
        top_videos = get_top_videos(channel_id, n=5)
        if not top_videos:
            return jsonify({'error': 'No top videos found for the channel'}), 404
        
        related_videos = []
        for video in top_videos:
            video_id = video['video_id']
            rel_vids = get_related_videos(video_id, m=15)
            for vid in rel_vids:
                if vid.get('channel_id') != channel_id:
                    related_videos.append(vid)
        
        if not related_videos:
            app.logger.info(f"No related videos found for channel {channel_id}, falling back to keyword search")
            for video in top_videos:
                query = ' '.join([word for word in video['title'].split() if len(word) > 3 and word.lower() not in ['the', 'and', 'video']])[:50]
                if query:
                    search_vids = search_videos_by_query(query, max_results=15)
                    for vid in search_vids:
                        if vid.get('channel_id') != channel_id:
                            related_videos.append(vid)
        
        if not related_videos:
            return jsonify({'error': 'No related videos found from similar channels after fallback search'}), 404
        
        outlier_videos = analyze_videos_for_outliers(related_videos)
        
        outlier_detector = OutlierDetector()
        formatted_outliers = []
        for video in outlier_videos:
            formatted_video = {
                'video_id': video['video_id'],
                'title': video['title'],
                'channel_id': video['channel_id'],
                'channel_title': video['channel_title'],
                'views': video['views'],
                'views_formatted': outlier_detector.format_number(video['views']),
                'channel_avg_views': video['channel_avg_views'],
                'channel_avg_views_formatted': outlier_detector.format_number(video['channel_avg_views']),
                'multiplier': video['multiplier'],
                'url': video['url'],
                'thumbnail_url': video['thumbnail_url']
            }
            formatted_outliers.append(formatted_video)
        
        return jsonify({
            'success': True,
            'channel_id': channel_id,
            'channel_title': channel_title,
            'total_results': len(formatted_outliers),
            'outliers': formatted_outliers
        })
    except Exception as e:
        app.logger.error(f"Channel outliers by ID error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
        
@app.route('/generate_titles', methods=['POST'])
def generate_titles():
    """Generate viral YouTube titles."""
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        idea = data.get('idea')
        script = data.get('script')
        if not idea:
            return jsonify({'error': 'Idea is required'}), 400
        
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        search_response = youtube.search().list(
            q=idea,
            part='snippet',
            type='video',
            order='viewCount',
            maxResults=10
        ).execute()
        titles = [item['snippet']['title'] for item in search_response.get('items', [])]
        if not titles:
            return jsonify({'error': 'No videos found for the given idea'}), 404
        
        prompt = (
            f"Given the idea: '{idea}' and the following popular video titles related to it: "
            f"{', '.join(titles)}, generate 5 viral video title options that are similar and likely to attract viewers. "
        )
        if script:
            prompt += (
                f"Also consider the following video script for context: '{script[:500]}' (truncated for brevity). "
                f"Use the script's themes to inform the titles, tags, and description. "
            )
        prompt += (
            f"For each title, provide a virality score out of 100. Also, provide 10 tags and one description that are common to all 5 titles. "
            f"Return the response in valid JSON format wrapped in ```json\n...\n``` with the following structure: "
            f"{{ 'titles': [{{'title': 'string', 'virality_score': int}}, ...], 'tags': ['string', ...], 'description': 'string' }}"
        )
        
        client = genai.GenerativeModel('gemini-2.0-flash')
        response = client.generate_content(prompt)
        gemini_response = response.text
        if gemini_response.startswith('```json\n') and gemini_response.endswith('\n```'):
            gemini_response = gemini_response[7:-4]
        gemini_response = json.loads(gemini_response)
        return jsonify(gemini_response)
    except Exception as e:
        app.logger.error(f"Generate titles error: {str(e)}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/similar_channels', methods=['POST'])
def similar_channels():
    """Fetch up to 15 similar channels based on a given channel ID."""
    try:
        data = request.get_json()
        if not data or 'channel_id' not in data:
            return jsonify({'error': 'Channel ID is required'}), 400
        
        channel_id = data['channel_id'].strip()
        if not channel_id:
            return jsonify({'error': 'Channel ID cannot be empty'}), 400
        
        # Verify the input channel exists
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        channel_request = youtube.channels().list(
            part='id,snippet',
            id=channel_id
        )
        response = channel_request.execute()
        if not response['items']:
            return jsonify({'error': 'Channel not found'}), 404
        input_channel_title = response['items'][0]['snippet']['title']
        
        # Get top videos from the input channel
        top_videos = get_top_videos(channel_id, n=5)
        if not top_videos:
            return jsonify({'error': 'No top videos found for the channel'}), 404
        
        # Collect related videos to find similar channels
        related_channel_ids = set()
        for video in top_videos:
            video_id = video['video_id']
            related_videos = get_related_videos(video_id, m=15)
            for vid in related_videos:
                if vid['channel_id'] != channel_id and len(related_channel_ids) < 15:
                    related_channel_ids.add(vid['channel_id'])
        
        # If insufficient channels, fall back to keyword search based on top video titles
        if len(related_channel_ids) < 15:
            app.logger.info(f"Only {len(related_channel_ids)} related channels found for channel {channel_id}, falling back to keyword search")
            for video in top_videos:
                query = ' '.join([word for word in video['title'].split() if len(word) > 3 and word.lower() not in ['the', 'and', 'video']])[:50]
                if query:
                    search_vids = search_videos_by_query(query, max_results=15)
                    for vid in search_vids:
                        if vid['channel_id'] != channel_id and len(related_channel_ids) < 15:
                            related_channel_ids.add(vid['channel_id'])
        
        if not related_channel_ids:
            return jsonify({'error': 'No similar channels found'}), 404
        
        # Fetch details for similar channels
        similar_channels = []
        channel_details_request = youtube.channels().list(
            part='snippet,statistics',
            id=','.join(list(related_channel_ids)[:15])
        )
        channel_details_response = channel_details_request.execute()
        
        for item in channel_details_response.get('items', []):
            similar_channels.append({
                'channel_id': item['id'],
                'channel_title': item['snippet']['title'],
                'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                'subscriber_count': int(item['statistics'].get('subscriberCount', 0))
            })
        
        app.logger.info(f"Found {len(similar_channels)} similar channels for channel {channel_id}")
        return jsonify({
            'success': True,
            'input_channel_id': channel_id,
            'input_channel_title': input_channel_title,
            'total_results': len(similar_channels),
            'similar_channels': similar_channels
        })
    except Exception as e:
        app.logger.error(f"Similar channels error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

def parse_duration(duration):
    """Parse ISO 8601 duration (e.g., PT1H2M3S) to seconds."""
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration)
    if match:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2) or 0)
        seconds = int(match.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds
    return 0

def format_number(num):
    """Format large numbers into a readable string (e.g., 1200 -> 1.2K, 1500000 -> 1.5M)."""
    if num >= 1000000:
        return f"{num/1000000:.1f}M"
    elif num >= 1000:
        return f"{num/1000:.1f}K"
    else:
        return str(num)

@app.route('/api/similar_videos', methods=['POST'])
def similar_videos():
    """Fetch up to 15 similar videos for a given video ID using a title-based search."""
    try:
        # Validate input
        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'error': 'Video ID is required'}), 400
        
        video_id = data['video_id'].strip()
        if not video_id:
            return jsonify({'error': 'Video ID cannot be empty'}), 400
        
        # Initialize YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Step 1: Fetch input video details
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            return jsonify({'error': 'Video not found'}), 404
        
        input_video = video_response['items'][0]
        input_title = input_video['snippet']['title']
        input_channel_id = input_video['snippet']['channelId']
        input_channel_title = input_video['snippet']['channelTitle']
        
        # Step 2: Generate search query from the title
        query = ' '.join([word for word in input_title.split() if len(word) > 3 and word.lower() not in ['the', 'and', 'video']])
        if not query:
            return jsonify({'error': 'No valid search query generated from title'}), 404
        
        # Step 3: Search for similar videos
        search_response = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=25,  # Fetch extra results to allow filtering
            order='relevance'
        ).execute()
        
        if not search_response.get('items', []):
            return jsonify({'error': 'No similar videos found'}), 404
        
        # Step 4: Filter results
        related_videos = []
        video_ids = []
        seen_video_ids = {video_id}  # Track seen IDs to avoid duplicates
        for item in search_response['items']:
            rel_video_id = item['id']['videoId']
            rel_channel_id = item['snippet']['channelId']
            # Exclude input video and same-channel videos
            if rel_video_id not in seen_video_ids and rel_channel_id != input_channel_id:
                seen_video_ids.add(rel_video_id)
                video_ids.append(rel_video_id)
                related_videos.append({
                    'video_id': rel_video_id,
                    'title': item['snippet']['title'],
                    'channel_id': rel_channel_id,  # Store channel_id
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'language': item['snippet'].get('defaultLanguage', 'en')
                })
        
        if not related_videos:
            return jsonify({'error': 'No similar videos found from different channels'}), 404
        
        # Step 5: Fetch additional details for videos
        details_response = youtube.videos().list(
            part='statistics,contentDetails',
            id=','.join(video_ids[:25])  # Limit to 25 IDs per API call
        ).execute()
        
        video_stats = {}
        for item in details_response.get('items', []):
            vid = item['id']
            video_stats[vid] = {
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'comments': int(item['statistics'].get('commentCount', 0)),
                'duration': item['contentDetails']['duration']
            }
        
        # Step 6: Fetch channel stats for average views and subscribers
        channel_ids = list(set(video['channel_id'] for video in related_videos))
        processed_channels = {}
        for channel_id in channel_ids:
            avg_views, subscriber_count = calculate_channel_average_views(channel_id)
            processed_channels[channel_id] = {
                'average_views': avg_views,
                'subscriber_count': subscriber_count
            }
        
        # Step 7: Format the response
        formatted_videos = []
        for video in related_videos[:15]:  # Limit to 15 similar videos
            vid = video['video_id']
            if vid in video_stats:
                stats = video_stats[vid]
                channel_data = processed_channels.get(video['channel_id'], {'average_views': 0, 'subscriber_count': 0})
                channel_avg = channel_data['average_views']
                multiplier = stats['views'] / channel_avg if channel_avg > 0 else 0
                duration_seconds = parse_duration(stats['duration'])
                engagement_rate = (stats['likes'] + stats['comments']) / stats['views'] if stats['views'] > 0 else 0
                formatted_video = {
                    'video_id': vid,
                    'title': video['title'],
                    'channel_id': video['channel_id'],  # Added
                    'channel_title': video['channel_title'],
                    'views': stats['views'],
                    'views_formatted': format_number(stats['views']),
                    'likes': stats['likes'],
                    'likes_formatted': format_number(stats['likes']),
                    'comments': stats['comments'],
                    'comments_formatted': format_number(stats['comments']),
                    'duration': stats['duration'],
                    'duration_seconds': duration_seconds,
                    'url': f"https://www.youtube.com/watch?v={vid}",
                    'published_at': video['published_at'],
                    'thumbnail_url': video['thumbnail_url'],
                    'engagement_rate': round(engagement_rate, 4),
                    'language': video['language'],
                    'multiplier': round(multiplier, 2),  # Added
                    'channel_avg_views': channel_avg,  # Added
                    'channel_avg_views_formatted': format_number(channel_avg),  # Added
                    'subscriber_count': channel_data['subscriber_count']  # Added
                }
                formatted_videos.append(formatted_video)
        
        if not formatted_videos:
            return jsonify({'error': 'No valid similar videos after processing'}), 404
        
        # Return successful response
        return jsonify({
            'success': True,
            'input_video_id': video_id,
            'input_video_title': input_title,
            'input_channel_title': input_channel_title,
            'total_results': len(formatted_videos),
            'similar_videos': formatted_videos
        })
    
    except HttpError as e:
        # Handle YouTube API-specific errors (e.g., quota exceeded)
        return jsonify({'success': False, 'error': f'YouTube API error: {str(e)}'}), 503
    except Exception as e:
        # Handle unexpected errors
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500
        
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
