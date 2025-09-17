from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
from googleapiclient.discovery import build
from script_generate import generate_script
import logging
import json
from youtube_api import get_viral_thumbnails

import isodate
from script_generate import generate_script_from_title
from thumbnail_review import review_thumbnail
import base64
import googleapiclient.discovery
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
from collections import defaultdict

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
def find_similar_channels(channel_id, max_channels=10):
    """Find up to 5 similar channels based on top videos' related content."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    top_videos = get_top_videos(channel_id, n=5)
    related_channel_ids = set()
    for video in top_videos:
        rel_vids = get_related_videos(video['video_id'], m=15)
        for vid in rel_vids:
            if vid.get('channel_id') and vid['channel_id'] != channel_id:
                related_channel_ids.add(vid['channel_id'])
    if len(related_channel_ids) < max_channels:
        for video in top_videos:
            query = ' '.join([word for word in video['title'].split() if len(word) > 3 and word.lower() not in ['the', 'and', 'video']])[:50]
            if query:
                search_vids = search_videos_by_query(query, max_results=15)
                for vid in search_vids:
                    if vid.get('channel_id') and vid['channel_id'] != channel_id:
                        related_channel_ids.add(vid['channel_id'])
    return list(related_channel_ids)[:max_channels]

def get_channel_shorts_details(channel_id, max_results=50):
    """Fetch up to 50 Shorts (<= 60 seconds) from a channel with detailed stats."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        # Get uploads playlist ID
        channels_request = youtube.channels().list(part='contentDetails,statistics', id=channel_id)
        channels_response = channels_request.execute()
        if not channels_response['items']:
            return []
        uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        # Get subscriber count
        subscriber_count = int(channels_response['items'][0]['statistics'].get('subscriberCount', 0))

        # Get playlist items
        playlist_request = youtube.playlistItems().list(part='snippet', playlistId=uploads_playlist_id, maxResults=max_results)
        playlist_response = playlist_request.execute()
        video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_response['items']]
        if not video_ids:
            return []

        # Get video details
        video_details_request = youtube.videos().list(part='snippet,statistics,contentDetails', id=','.join(video_ids))
        video_details_response = video_details_request.execute()
        raw_videos = [
            [item['id'], item['snippet']['title'], item['snippet']['thumbnails']['high']['url'], 
             item['contentDetails']['duration'], item['snippet']['channelId'], item['snippet']['channelTitle'], 
             item['snippet']['publishedAt'], int(item['statistics'].get('viewCount', 0)),
             int(item['statistics'].get('likeCount', 0)), int(item['statistics'].get('commentCount', 0))]
            for item in video_details_response['items']
        ]
        
        # Filter Shorts
        similarity_finder = UltraFastYouTubeSimilarity()
        filtered_videos = similarity_finder._filter_shorts_only(raw_videos, max_duration=60)
        videos = []
        for video in filtered_videos:
            duration = video[3]
            duration_seconds = parse_duration(duration)
            videos.append({
                'video_id': video[0],
                'title': video[1],
                'channel_id': video[4],
                'channel_title': video[5],
                'published_at': video[6],
                'thumbnail_url': video[2],
                'views': video[7],
                'likes': video[8],
                'comments': video[9],
                'duration': duration,
                'duration_seconds': duration_seconds,
                'subscriber_count': subscriber_count
            })
        return videos
    except Exception as e:
        app.logger.error(f"Error fetching Shorts for channel {channel_id}: {str(e)}")
        return []


def get_channel_videos_details(channel_id, max_results=50):
    """Fetch up to 50 videos from a channel, excluding Shorts, with detailed stats."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        # Get uploads playlist ID
        channels_request = youtube.channels().list(part='contentDetails', id=channel_id)
        channels_response = channels_request.execute()
        if not channels_response['items']:
            return []
        uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # Get playlist items
        playlist_request = youtube.playlistItems().list(part='snippet', playlistId=uploads_playlist_id, maxResults=max_results)
        playlist_response = playlist_request.execute()
        video_ids = [item['snippet']['resourceId']['videoId'] for item in playlist_response['items']]
        if not video_ids:
            return []

        # Get video details
        video_details_request = youtube.videos().list(part='snippet,statistics,contentDetails', id=','.join(video_ids))
        video_details_response = video_details_request.execute()
        videos = []
        for item in video_details_response['items']:
            duration = item['contentDetails']['duration']
            duration_seconds = parse_duration(duration)
            if duration_seconds > 60:  # Exclude Shorts
                videos.append({
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'comments': int(item['statistics'].get('commentCount', 0)),
                    'duration': duration,
                    'duration_seconds': duration_seconds
                })
        return videos
    except Exception as e:
        app.logger.error(f"Error fetching videos for channel {channel_id}: {str(e)}")
        return []
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

def get_random_trending_videos(max_results=200, query=None):
    """Fetch random trending videos using varied queries, regions, and dates."""
    if not check_internet():
        app.logger.error("No internet connection to youtube.googleapis.com")
        return []
    
    trending_queries = [
        "trending", "popular", "viral", "new video", "latest", "hot", "buzz", "top videos",
        "music", "gaming", "vlog", "tutorial", "review", "reaction", "challenge"
    ]
    
    # Use provided query or select a random one
    query = query or random.choice(trending_queries)
    
    regions = ['US', 'GB', 'IN', 'CA', 'AU', 'DE', 'FR']
    region_code = random.choice(regions)
    
    days_back = 7  # Focus on recent videos
    published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat("T") + "Z"
    
    try:
        videos = search_videos(
            query=query,
            max_results=max_results,
            region_code=region_code,
            published_after=published_after
        )
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
    def _filter_shorts_only(self, videos, max_duration=60):
        """Filter to include only YouTube Shorts (duration <= 60 seconds)."""
        video_ids = [video[0] for video in videos]
        duration_map = self._get_video_durations_batch(video_ids)
        filtered_videos = [v for v in videos if duration_map.get(v[0], 0) <= max_duration and duration_map.get(v[0], 0) > 0]
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
    """API endpoint to get similar thumbnails with full video details."""
    try:
        data = request.get_json()
        app.logger.debug(f"Received request data: {data}")
        video_id = data.get('video_id')
        if not video_id:
            app.logger.error("Missing video_id in request")
            return jsonify({'success': False, 'error': 'Video ID is required'}), 400
        similarity_threshold = float(data.get('similarity_threshold', 0.70))
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        result_data, _, error = find_similar_videos_enhanced(
            video_url,
            max_results=150,
            top_similar=50,
            similarity_threshold=similarity_threshold
        )
        if error:
            app.logger.error(f"Similarity analysis failed: {error}")
            return jsonify({'success': False, 'error': error}), 400
        app.logger.info(f"Returning {len(result_data['similar_videos'])} similar videos for video_id: {video_id}")
        return jsonify({
            'success': True,
            'input_video': result_data['input_video'],
            'similar_videos': result_data['similar_videos'],
            'processing_stats': result_data['processing_stats']
        })
    except Exception as e:
        app.logger.error(f"Error in get_similar_thumbnails: {str(e)}")
        return jsonify({'success': False, 'error': f'Internal server error: {str(e)}'}), 500

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
from flask import Flask, request, jsonify
import googleapiclient.discovery
import googleapiclient.errors
import google.generativeai as genai
from datetime import datetime
import os
import isodate  # pip install isodate

app = Flask(__name__)

# 🔑 Your API keys (set them in your environment)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini SDK
genai.configure(api_key=GEMINI_API_KEY)


def format_number(num: int) -> str:
    """Format large numbers into human-readable strings (e.g., 1.5M)."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def parse_duration(duration: str) -> int:
    """Convert ISO 8601 duration (e.g., PT5M30S) into seconds."""
    try:
        return int(isodate.parse_duration(duration).total_seconds())
    except Exception:
        return 0
def get_category_id(niche: str) -> str:
    """
    Map Gemini niche (text) into a YouTube videoCategoryId.
    Fallback = 24 (Entertainment).
    """
    niche = niche.lower()

    mapping = {
        "music": "10",
        "gaming": "20",
        "news": "25",
        "sports": "17",
        "travel": "19",
        "education": "27",
        "science": "28",
        "technology": "28",
        "film": "1",
        "entertainment": "24",
        "comedy": "23",
        "howto": "26",
        "lifestyle": "22",
        "people": "22",
        "autos": "2",
        "pets": "15",
        "animals": "15",
        "food": "26",
        "finance": "27",
        "fitness": "17"
    }

    for key, cid in mapping.items():
        if key in niche:
            return cid

    return "24"  # default to Entertainment
def fetch_video_details(youtube, video_ids):
    all_videos = []
    for i in range(0, len(video_ids), 50):
        batch_ids = video_ids[i:i+50]
        resp = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=",".join(batch_ids)
        ).execute()
        all_videos.extend(resp.get('items', []))
    return all_videos






@app.route("/api/channel_outliers_by_id", methods=["POST"])
def channel_outliers_by_id():
    try:
        data = request.get_json()
        channel_id = data.get("channel_id", "").strip()
        if not channel_id:
            return jsonify({"error": "Channel ID is required"}), 400

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # 1. Fetch channel info (include contentDetails for uploads playlist)
        channel_resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()

        if not channel_resp["items"]:
            return jsonify({"error": "Channel not found"}), 404

        channel_info = channel_resp["items"][0]
        channel_title = channel_info["snippet"]["title"]
        subs_count = int(channel_info["statistics"].get("subscriberCount", 1))
        uploads_playlist = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]

        # 2. Get last 10 uploads, filter medium/long → take 5
        videos_resp = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist,
            maxResults=10
        ).execute()

        candidate_videos = []
        for item in videos_resp.get("items", []):
            vid_id = item["contentDetails"]["videoId"]
            vid_details = youtube.videos().list(
                part="contentDetails,snippet",
                id=vid_id
            ).execute()

            if not vid_details["items"]:
                continue

            v = vid_details["items"][0]
            duration = parse_duration(v["contentDetails"]["duration"])
            if duration >= 240:  # medium/long only
                candidate_videos.append({
                    "id": vid_id,
                    "title": v["snippet"]["title"]
                })

            if len(candidate_videos) >= 5:
                break

        if not candidate_videos:
            return jsonify({"error": "No medium/long videos found for this channel"}), 404

        # 3. Collect competitor channels from search results
        competitor_channels = {}
        for video in candidate_videos:
            search_resp = youtube.search().list(
                part="snippet",
                q=video["title"],
                type="video",
                maxResults=20,
                relevanceLanguage="en"  # adjust as needed
            ).execute()

            for item in search_resp.get("items", []):
                ch_id = item["snippet"]["channelId"]
                ch_title = item["snippet"]["channelTitle"]

                if ch_id != channel_id and ch_id not in competitor_channels:
                    competitor_channels[ch_id] = ch_title

                if len(competitor_channels) >= 20:
                    break
            if len(competitor_channels) >= 20:
                break

        competitors = list(competitor_channels.items())[:20]

        # 4. For each competitor, get top 5 by views + latest 5 with ratio > 1
        all_videos = []
        for comp_id, comp_title in competitors:
            comp_resp = youtube.channels().list(
                part="statistics,contentDetails",
                id=comp_id
            ).execute()

            if not comp_resp["items"]:
                continue

            comp_info = comp_resp["items"][0]
            comp_subs = int(comp_info["statistics"].get("subscriberCount", 1))
            comp_uploads = comp_info["contentDetails"]["relatedPlaylists"]["uploads"]

            # Fetch up to 50 videos to sort/filter
            comp_uploads_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=comp_uploads,
                maxResults=50
            ).execute()

            comp_video_ids = [v["contentDetails"]["videoId"] for v in comp_uploads_resp.get("items", [])]
            comp_videos = fetch_video_details(youtube, comp_video_ids)

            # Sort by views → Top 5
            top_videos = sorted(
                comp_videos,
                key=lambda x: int(x["statistics"].get("viewCount", 0)),
                reverse=True
            )[:5]

            # Latest 5 (ratio > 1)
            latest_videos = []
            for v in comp_videos[:10]:  # just take first 10 uploads
                views = int(v["statistics"].get("viewCount", 0))
                if comp_subs > 0 and (views / comp_subs) > 1:
                    latest_videos.append(v)
                if len(latest_videos) >= 5:
                    break

            # Format videos
            for v in top_videos + latest_videos:
                duration = parse_duration(v["contentDetails"]["duration"])
                views = int(v["statistics"].get("viewCount", 0))
                multiplier = round(views / comp_subs, 2) if comp_subs > 0 else 0

                all_videos.append({
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel_id": comp_id,
                    "channel_title": comp_title,
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "multiplier": multiplier,
                    "thumbnail_url": v["snippet"]["thumbnails"]["high"]["url"],
                    "url": f"https://www.youtube.com/watch?v={v['id']}"
                })

            if len(all_videos) >= 200:
                break

        return jsonify({
            "success": True,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "competitors": [{"id": cid, "title": ctitle} for cid, ctitle in competitors],
            "total_videos": len(all_videos),
            "videos": all_videos[:200]
        })

    except Exception as e:
        app.logger.error(f"Channel outliers error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_titles', methods=['POST'])
def generate_titles():
    """Generate viral YouTube titles based on provided topic, prompt, or script."""
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        topic = data.get('topic')
        prompt = data.get('prompt')
        script = data.get('script')
        niche = data.get('niche', 'general')  # Optional: niche for tailoring titles
        audience = data.get('audience', 'general')  # Optional: target audience
        
        # Check if at least one input is provided
        if not any([topic, prompt, script]):
            return jsonify({'error': 'At least one of topic, prompt, or script is required'}), 400
        
        # Initialize base prompt from provided guidelines
        base_prompt = """
Generate 5 viral YouTube video title options that maximize click-through rate (CTR) and align with the video's content, adhering to the following comprehensive guidelines for creating high-performing titles. **Critical Instruction**: Strictly avoid using special symbols such as !, @, #, $, %, ^, &, *, or any other non-alphanumeric characters except spaces, commas, periods, question marks, and hyphens. Titles must be clean, professional, and use only alphanumeric characters, spaces, commas, periods, question marks, and hyphens to ensure compatibility and readability.

### Foundation of a Viral Title
- **Role and Importance**: The title acts as the headline or 'billboard' in the crowded YouTube feed, appearing alongside the thumbnail. Its primary jobs are to help YouTube recommend the content to the right audience through metadata (title, description, tags), accurately confirm what the viewer will get to encourage continued watching, and boost reach even for small channels. A single title change can revive underperforming videos. Titles directly impact CTR, impressions, and watch time. In education niches, clear titles may outperform flashy thumbnails; in entertainment, thumbnails lead but titles frame the story. Always prioritize writing for humans first, as algorithms reward what people click and watch.
- **How YouTube Processes Titles**: YouTube uses the title as part of metadata to understand the video's topic via keywords. Balance human appeal (emotional pull for clicks) with algorithmic needs (1-2 relevant keywords for accurate categorization).
- **Core Qualities for High-CTR Titles**: Keep titles short and easy to read, ideally under 70 characters (absolute max 100, with key words front-loaded since only 40-50 characters may show in Suggested Videos). Ensure accuracy and honesty to avoid misleading viewers, which harms retention and rankings—no clickbait that doesn't deliver. Drive emotions like curiosity, fear, excitement, urgency, or shock. Provide a clear benefit or hook, telling viewers what they'll gain or learn.
- **Psychological Drivers**: Leverage three click-worthy emotions: Curiosity (opens an information gap, e.g., 'How did they do that'), Desire (promises an outcome or status, e.g., 'faster, richer, calmer'), Fear (helps avoid loss or mistakes, e.g., 'don’t waste time, avoid this mistake'). Combine Curiosity + Desire for sustainable appeal or Curiosity + Fear for stronger ethical impact. Ensure the video's intro (script) confirms and exceeds title expectations for 'click confirmation.'

### Core Title Frameworks
- **Title Modes Based on Discovery Path**: Select one of three modes fitting the video's path:
  - **Searchable**: Clear keywords and outcomes for tutorials or evergreen topics (e.g., 'How to Start a YouTube Channel in 2025').
  - **Browse/Intriguing**: Sparks curiosity for recommendation feeds, paired with bold thumbnails (e.g., 'This Camera is Worse and That’s Why I Bought It').
  - **Hybrid**: Blends keywords with intrigue for both search and browse (e.g., 'How to Make 100 Dollars a Day Without a Job').
- **Power Words for Appeal**: Incorporate 1-2 words from these categories to enhance pull:
  - Authority: Pro, Expert, Insider, Proven, Blueprint.
  - Urgency: Today, Now, Before You, Last Chance.
  - Exclusivity: Secret, Hidden, Little-Known, Behind-the-Scenes.
  - Niche-Specific: For finance (millionaire, passive income), fitness (shredded, fat-burning), creators (algorithm, CTR, hook).
  - Avoid dull words that reduce appeal.
- **Proven High-Performing Formats**: Adapt one of these for each 5 titles diifferen structures to the video’s topic and niche:
  - How-To: 'How to Do X in Timeframe'.
  - List: 'Number Mistakes Every Audience Makes'.
  - Unexpected: 'Why I Did Something Unexpected'.
  - Urgency: 'Do This Before Event or Deadline'.
  - Experiment: 'I Tried Thing So You Don’t Have To'.
  - Result/Proof: 'How I Made X Dollars in Timeframe'.
  - Question: 'Is This the End of Topic'.
  - Before/After: 'From 0 to Goal in Timeframe'.
  - Challenge: 'I Tried Action for Timeframe - Here’s What Happened'.
  - Audience Callout: 'If You’re a Specific Audience, Watch This'.
  - Open a Loop: 'I Flew 7292 Miles to Eat Here'.
  - Additional List: 'Best 10 Ways to Grow on YouTube'.

### Process for Writing Viral Titles
Follow this step-by-step research and creation flow:
1. **Start with Video Content**: Review the script or edited video first. Identify the main payoff or hook moment to ensure the title matches the content exactly. Viewers should feel 'Yes, this is exactly what I clicked for.'
2. **Research Before Writing**: Based on your knowledge of YouTube trends, infer popular title structures and keywords for the given topic. Identify high-performing patterns typically used for similar topics, niches, or audiences (e.g., tutorials, vlogs, or challenges). Consider common title frameworks used by successful videos in the niche. If available, draw inspiration from past high-performing titles, adapting their structures by swapping nouns or verbs.
3. **Tweak and Adapt**: Adjust successful title formats to fit the video’s unique angle, ensuring it aligns with active searches while adding curiosity for browse recommendations.
4. **Title & Thumbnail Chemistry**: Ensure title and thumbnail work together—divide labor: one teases (e.g., shocking visual), the other clarifies the promise. Avoid repeating text unless the concept is exceptionally strong. For intriguing titles, pair with visual proof in thumbnails (e.g., surprising face with text).
5. **Final Pass for Accuracy & Strength**: Confirm 100% accuracy to the video—no misleading elements that kill retention. Adhere to length: Ideal 50-70 characters, max 100, front-load key words. Test hook strength: 'Would someone who clicked feel the video delivered exactly what was promised?' Rework if there’s any risk of misleading.

Tailor all titles to the video’s niche (e.g., education, entertainment, finance, fitness) and target audience using specific language or callouts. **Reiterated Instruction**: Ensure titles contain no special symbols (e.g., !, @, #, $, %, ^, &, *); use only alphanumeric characters, spaces, commas, periods, question marks, and hyphens.
"""
        
        # Add topic and research data if provided
        if topic:
            try:
                youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
                search_response = youtube.search().list(
                    q=topic,
                    part='snippet',
                    type='video',
                    order='viewCount',
                    maxResults=10
                ).execute()
                titles = [item['snippet']['title'] for item in search_response.get('items', [])]
                if not titles:
                    app.logger.warning(f"No videos found for topic: {topic}")
                else:
                    base_prompt += (
                        f"\nGiven the topic: '{topic}' and the following popular video titles related to it: "
                        f"{', '.join(titles)}, use these as inspiration for title structures and keywords. "
                    )
            except HttpError as e:
                app.logger.error(f"YouTube API error while searching topic '{topic}': {str(e)}")
                base_prompt += f"\nGiven the topic: '{topic}', infer relevant keywords and title structures. "
        
        # Add niche and audience if provided
        if niche != 'general' or audience != 'general':
            base_prompt += f"\nTailor the titles for the niche: '{niche}' and target audience: '{audience}'. "
        
        # Add custom prompt if provided
        if prompt:
            base_prompt += f"\nUse the following user prompt for additional context: '{prompt[:500]}' (truncated for brevity). "
        
        # Add script to prompt if provided
        if script:
            base_prompt += (
                f"\nAlso consider the following video script for context: '{script[:500]}' (truncated for brevity). "
                f"Extract the main hook or payoff from the script to ensure the titles match the video’s content. "
            )
        
        # Finalize prompt with output instructions
        base_prompt += (
            f"\nFor each title, provide a virality score out of 100 based on its potential CTR, emotional appeal, and alignment with YouTube’s algorithm. "
            f"Also, provide 10 relevant tags and one description (100-150 words) common to all 5 titles, optimized for the topic, niche, and audience. "
            f"Return the response in valid JSON format with the following structure: "
            f"{{ 'titles': [{{'title': 'string', 'virality_score': int}}, ...], 'tags': ['string', ...], 'description': 'string' }}"
        )
        
        # Generate titles using the AI model
        client = genai.GenerativeModel('gemini-2.0-flash')
        response = client.generate_content(base_prompt)
        gemini_response = response.text
        
        # Clean and parse the response
        if gemini_response.startswith('```json\n') and gemini_response.endswith('\n```'):
            gemini_response = gemini_response[7:-4]
        try:
            gemini_response = json.loads(gemini_response)
        except json.JSONDecodeError as e:
            app.logger.error(f"Invalid JSON response from AI model: {str(e)}")
            return jsonify({'error': 'Failed to parse AI response'}), 500
        
        # Validate response structure
        if not isinstance(gemini_response, dict) or 'titles' not in gemini_response or 'tags' not in gemini_response or 'description' not in gemini_response:
            app.logger.error("Invalid response structure from AI model")
            return jsonify({'error': 'Invalid response structure from AI model'}), 500
        
        app.logger.info(f"Generated {len(gemini_response['titles'])} titles for topic: {topic or 'unknown'}")
        return jsonify(gemini_response)
    
    except HttpError as e:
        app.logger.error(f"YouTube API error in generate_titles: {str(e)}")
        return jsonify({'error': f'YouTube API error: {str(e)}'}), 503
    except Exception as e:
        app.logger.error(f"Generate titles error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/generate_ideas', methods=['POST'])
def generate_ideas():
    """Generate 20 similar viral YouTube video ideas based on provided title."""
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        title = data.get('title')
        
        # Check if title is provided
        if not title:
            return jsonify({'error': 'Title is required'}), 400
        
        # Initialize base prompt for idea generation
        base_prompt = """
Generate exactly 20 similar viral YouTube video idea options based on the provided input title. These ideas should be variations, expansions, or related angles on the core theme, designed to maximize click-through rate (CTR) while aligning with potential video content. Present each idea as a concise, title-like phrase that could serve as a video concept or actual title. **Critical Instruction**: Strictly avoid using special symbols such as !, @, #, $, %, ^, &, *, or any other non-alphanumeric characters except spaces, commas, periods, question marks, and hyphens. Ideas must be clean, professional, and use only alphanumeric characters, spaces, commas, periods, question marks, and hyphens to ensure compatibility and readability.

### Foundation for Generating Similar Ideas
- **Role and Importance**: These ideas act as brainstorming tools for creators, helping to expand on a successful or core concept by exploring different angles, depths, or related subtopics. They should inspire new videos that build on the original theme, boosting channel growth through series or related content. Ideas directly impact potential CTR, impressions, and watch time by suggesting hooks that resonate with audiences. Prioritize human appeal first, as algorithms reward engaging content.
- **How to Generate Ideas**: Use the input title as the seed. Identify the core theme (e.g., 'making money on Etsy' involves e-commerce, side hustles, creative selling). Then, vary by adding twists like timelines, challenges, hacks, case studies, or audience-specific adaptations. Balance keyword relevance for search with emotional hooks for recommendations.
- **Core Qualities for Viral Ideas**: Keep ideas short and punchy, ideally under 70 characters. Ensure they are honest and deliverable to maintain viewer trust. Drive emotions like curiosity, aspiration, urgency, or surprise. Provide clear value, such as skills, secrets, or results.
- **Psychological Drivers**: Leverage curiosity (e.g., 'hidden ways'), desire (e.g., 'achieve riches'), fear (e.g., 'avoid failures'). Combine them to create compelling variations on the original idea.

### Core Frameworks for Similar Ideas
- **Idea Variation Modes**: Base ideas on the original's discovery path, but diversify:
  - Searchable Variations: Keyword-focused tweaks (e.g., add 'beginner' or 'advanced').
  - Intriguing Twists: Add curiosity elements (e.g., 'what I wish I knew').
  - Hybrid Expansions: Blend with new angles (e.g., 'in 2025' or 'without experience').
- **Power Words for Appeal**: Incorporate 1-2 words to enhance variations:
  - Authority: Ultimate, Master, Proven.
  - Urgency: Fast, Quick, Instant.
  - Exclusivity: Secret, Underrated, Insider.
  - Niche-Specific: Adapt to the theme (e.g., for e-commerce: passive, scalable, profitable).
- **Proven Formats for Idea Variations**: Use these structures, adapting to the input theme:
  - How-To Expansion: 'How to Achieve Bigger Outcome in Theme'.
  - List Variation: 'Number Hacks for Theme Success'.
  - Unexpected Angle: 'Why Theme is Changing Forever'.
  - Urgency Twist: 'Master Theme Before It's Too Late'.
  - Experiment Idea: 'I Tested Theme Method - Results'.
  - Result-Focused: 'From Zero to Milestone in Theme'.
  - Question Variation: 'Can You Succeed in Theme Without X'.
  - Before/After: 'Transform Your Theme Game in Timeframe'.
  - Challenge: 'Theme Challenge for Beginners'.
  - Audience Callout: 'Theme Tips for Specific Group'.
  - Open Loop: 'The One Thing Missing from Your Theme Strategy'.
  - Additional: 'Best Alternatives to Original Theme'.

### Process for Generating Similar Ideas
Follow this step-by-step flow:
1. **Start with Input Title**: Analyze the provided title. Extract the core theme, key elements, and unique angles.
2. **Generate Variations**: Create variations by modifying elements like scope, depth, or focus. Use popular structures and keywords inferred from the theme.
3. **Diversify and Adapt**: Ensure the 20 ideas cover a range of formats (at least 5 different structures). Make them similar yet distinct, expanding on the original without copying it directly.
4. **Idea & Content Chemistry**: Ideas should pair well with thumbnails and scripts—suggest hooks that can be visually teased.
5. **Final Pass for Relevance & Strength**: Ensure ideas are 100% aligned with the theme. Adhere to length: Ideal 50-70 characters. Test: 'Does this idea build on the original while offering fresh value?'

**Reiterated Instruction**: Ensure ideas contain no special symbols (e.g., !, @, #, $, %, ^, &, *); use only alphanumeric characters, spaces, commas, periods, question marks, and hyphens.

Given the input title: '{title}'.

Return the response in valid JSON format with the following structure:
{{ 'ideas': ['string', ...] }}
Return exactly 20 ideas, each as a string in the 'ideas' array, with no additional fields like virality scores, tags, or descriptions.
"""
        
        # Generate ideas using the AI model
        client = genai.GenerativeModel('gemini-2.0-flash')
        response = client.generate_content(base_prompt)
        gemini_response = response.text
        
        # Clean and parse the response
        if gemini_response.startswith('```json\n') and gemini_response.endswith('\n```'):
            gemini_response = gemini_response[7:-4]
        try:
            gemini_response = json.loads(gemini_response)
        except json.JSONDecodeError as e:
            app.logger.error(f"Invalid JSON response from AI model: {str(e)}")
            return jsonify({'error': 'Failed to parse AI response'}), 500
        
        # Validate response structure
        if not isinstance(gemini_response, dict) or 'ideas' not in gemini_response:
            app.logger.error("Invalid response structure from AI model")
            return jsonify({'error': 'Invalid response structure from AI model'}), 500
        
        # Ensure exactly 20 ideas
        if len(gemini_response['ideas']) != 20:
            app.logger.warning(f"Expected 20 ideas, got {len(gemini_response['ideas'])}")
            return jsonify({'error': 'Expected exactly 20 ideas'}), 500
        
        app.logger.info(f"Generated 20 ideas for title: {title}")
        return jsonify(gemini_response)
    
    except Exception as e:
        app.logger.error(f"Generate ideas error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500



@app.route('/api/generate_thumbnail', methods=['POST'])
def generate_thumbnail():
    """
    Generate a YouTube video thumbnail based on a user-provided prompt with 1280x720 resolution.
    The prompt is sent as a JSON payload to this endpoint.
    """
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        prompt = data.get('prompt')

        if not prompt:
            # Return an error if the prompt is missing from the request
            return jsonify({'error': 'Prompt is required'}), 400

        # Use the specific model for image generation
        client = genai.GenerativeModel('gemini-2.0-flash-preview-image-generation')

        # Enhanced prompt with explicit dimension requirement
        enhanced_prompt = (
            f"{prompt}. Generate the image as a YouTube thumbnail with a resolution of 1280x720 pixels, "
            f"ensuring it is vibrant, eye-catching, and optimized for clickability with bold colors and clear text."
        )

        # Use dictionary for generation settings, removing unsupported width/height
        generation_config = {
            "response_modalities": ["TEXT", "IMAGE"]
        }

        # Make the API call to generate content
        response = client.generate_content(
            enhanced_prompt,
            generation_config=generation_config
        )

        # Log the response structure for debugging
        app.logger.debug(f"Response structure: {response.__dict__}")

        # Check if the response contains content parts
        if not response.candidates or not response.candidates[0].content.parts:
            app.logger.error("No content parts returned by the generative model.")
            return jsonify({'error': 'Failed to generate thumbnail: no content'}), 500

        # Find the base64-encoded image data within the response parts
        image_part = None
        mime_type = 'image/png'  # Default assumption
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                image_part = base64.b64encode(part.inline_data.data).decode('utf-8')
                mime_type = part.inline_data.mime_type
                break
            elif hasattr(part, 'text'):
                app.logger.debug(f"Text output (if any): {part.text}")

        if not image_part:
            app.logger.error("No image data found in the response parts.")
            return jsonify({'error': 'Failed to generate thumbnail: no image data'}), 500

        # Validate base64 string
        try:
            image_data = base64.b64decode(image_part)
        except Exception as e:
            app.logger.error(f"Invalid base64 image data: {str(e)}")
            return jsonify({'error': 'Invalid thumbnail data generated'}), 500

        # Optional: Verify image dimensions
        try:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            app.logger.debug(f"Generated image dimensions: {width}x{height}")
            if width != 1280 or height != 720:
                app.logger.warning(f"Image dimensions {width}x{height} do not match requested 1280x720")
        except Exception as e:
            app.logger.error(f"Failed to verify image dimensions: {str(e)}")

        # Determine format from mime_type
        format_map = {
            'image/png': 'png',
            'image/jpeg': 'jpeg',
            'image/jpg': 'jpeg'
        }
        image_format = format_map.get(mime_type, 'png')

        # Return the successful response with the base64-encoded image data
        return jsonify({
            'success': True,
            'prompt': prompt,
            'thumbnail': {
                'format': image_format,
                'data': image_part
            }
        })

    except Exception as e:
        # Catch and handle any unexpected errors
        app.logger.error(f"Generate thumbnail error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
@app.route('/api/generate_thumbnail_from_title', methods=['POST'])
def generate_thumbnail_from_title():
    """
    Generate a YouTube video thumbnail from a title using Gemini with 1024x576 resolution.
    """
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        title = data.get('title')

        if not title:
            return jsonify({'error': 'Title is required'}), 400

        # Step 1: Generate a thumbnail prompt using Gemini
        prompt_client = genai.GenerativeModel('gemini-2.0-flash')
        prompt_request = (
            f"Based on the video title '{title}', create a detailed prompt for generating a YouTube thumbnail. "
            f"The prompt should describe a vibrant, eye-catching image optimized for clickability with bold colors and clear text. "
            f"Explicitly include that the image must have a resolution of 1024x576 pixels. "
            f"Return only the prompt text."
        )
        
        prompt_response = prompt_client.generate_content(prompt_request)
        
        # Log the prompt response
        app.logger.debug(f"Prompt response structure: {prompt_response.__dict__}")

        # Extract the generated prompt
        if not prompt_response.candidates or not prompt_response.candidates[0].content.parts:
            app.logger.error("No content parts in prompt response.")
            return jsonify({'error': 'Failed to generate thumbnail prompt: no content'}), 500

        generated_prompt = next((
            part.text for part in prompt_response.candidates[0].content.parts
            if hasattr(part, 'text')
        ), None)

        if not generated_prompt:
            app.logger.error("No text prompt generated.")
            return jsonify({'error': 'Failed to generate thumbnail prompt: no text'}), 500

        app.logger.debug(f"Generated prompt: {generated_prompt}")

        # Step 2: Generate the thumbnail
        thumbnail_client = genai.GenerativeModel('gemini-2.0-flash-preview-image-generation')
        generation_config = {
            "response_modalities": ["TEXT", "IMAGE"]
        }

        response = thumbnail_client.generate_content(
            generated_prompt,
            generation_config=generation_config
        )

        # Log the thumbnail response
        app.logger.debug(f"Thumbnail response structure: {response.__dict__}")

        # Check if the response contains content parts
        if not response.candidates or not response.candidates[0].content.parts:
            app.logger.error("No content parts returned by the generative model.")
            return jsonify({'error': 'Failed to generate thumbnail: no content'}), 500

        # Find the base64-encoded image data
        image_part = None
        mime_type = 'image/png'
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                image_part = base64.b64encode(part.inline_data.data).decode('utf-8')
                mime_type = part.inline_data.mime_type
                break
            elif hasattr(part, 'text'):
                app.logger.debug(f"Text output (if any): {part.text}")

        if not image_part:
            app.logger.error("No image data found in the response parts.")
            return jsonify({'error': 'Failed to generate thumbnail: no image data'}), 500

        # Validate base64 string
        try:
            image_data = base64.b64decode(image_part)
        except Exception as e:
            app.logger.error(f"Invalid base64 image data: {str(e)}")
            return jsonify({'error': 'Invalid thumbnail data generated'}), 500

        # Verify image dimensions
        try:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))
            width, height = image.size
            app.logger.debug(f"Generated image dimensions: {width}x{height}")
            if width != 1024 or height != 576:
                app.logger.warning(f"Image dimensions {width}x{height} do not match requested 1024x576")
        except Exception as e:
            app.logger.error(f"Failed to verify image dimensions: {str(e)}")

        # Determine format from mime_type
        format_map = {
            'image/png': 'png',
            'image/jpeg': 'jpeg',
            'image/jpg': 'jpeg'
        }
        image_format = format_map.get(mime_type, 'png')

        # Return the response
        return jsonify({
            'success': True,
            'title': title,
            'generated_prompt': generated_prompt,
            'thumbnail': {
                'format': image_format,
                'data': image_part
            }
        })

    except Exception as e:
        app.logger.error(f"Generate thumbnail error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/ideas_by_channel_id', methods=['POST'])
def ideas_by_channel_id():
    """Fetch the last 15 videos (including Shorts) from a YouTube channel by channel ID."""
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        channel_id = data.get('channel_id')

        if not channel_id:
            return jsonify({'error': 'Channel ID is required'}), 400
        channel_id = channel_id.strip()
        if not channel_id:
            return jsonify({'error': 'Channel ID cannot be empty'}), 400

        if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY':
            app.logger.error("YouTube API key not configured.")
            return jsonify({'error': 'YouTube API key not configured'}), 500

        # Build YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        # Verify channel exists and fetch statistics
        channel_request = youtube.channels().list(part='snippet,statistics', id=channel_id)
        channel_response = channel_request.execute()
        if not channel_response.get('items'):
            app.logger.error(f"Channel not found for ID: {channel_id}")
            return jsonify({'error': 'Channel not found'}), 404
        channel_title = channel_response['items'][0]['snippet']['title']
        subscriber_count = int(channel_response['items'][0]['statistics'].get('subscriberCount', 0))

        # Fetch the last 15 videos
        search_request = youtube.search().list(
            part='id',
            channelId=channel_id,
            order='date',
            maxResults=15,
            type='video'
        )
        search_response = search_request.execute()

        if not search_response.get('items'):
            app.logger.error(f"No videos found for channel ID: {channel_id}")
            return jsonify({'error': 'No videos found for the channel'}), 404

        # Get video details
        video_ids = [item['id']['videoId'] for item in search_response['items']]
        video_details_request = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(video_ids)
        )
        video_details_response = video_details_request.execute()

        videos = []
        total_views = 0
        for item in video_details_response.get('items', []):
            try:
                duration = item['contentDetails']['duration']
                duration_seconds = parse_duration(duration)
                views = int(item['statistics'].get('viewCount', 0))
                likes = int(item['statistics'].get('likeCount', 0))
                comments = int(item['statistics'].get('commentCount', 0))
                engagement_rate = (likes + comments) / views if views > 0 else 0
                try:
                    pub_date = datetime.datetime.strptime(item['snippet']['publishedAt'], '%Y-%m-%dT%H:%M:%SZ')
                    video_age_days = (datetime.datetime.utcnow() - pub_date).days
                except:
                    video_age_days = 0

                multiplier = engagement_rate * (views / 1000) if views > 0 else 0
                total_views += views
                video = {
                    'video_id': item['id'],
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'views': views,
                    'views_formatted': format_number(views),
                    'likes': likes,
                    'likes_formatted': format_number(likes),
                    'comments': comments,
                    'comments_formatted': format_number(comments),
                    'duration': duration,
                    'duration_seconds': duration_seconds,
                    'url': f"https://www.youtube.com/watch?v={item['id']}",
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'engagement_rate': round(engagement_rate, 4),
                    'video_age_days': video_age_days,
                    'language': item['snippet'].get('defaultLanguage', 'en'),
                    'multiplier': round(multiplier, 4),
                    'channel_avg_views': int(total_views / len(videos) + 1) if videos else 0,
                    'channel_avg_views_formatted': format_number(int(total_views / len(videos) + 1)) if videos else '0',
                    'subscriber_count': subscriber_count
                }
                videos.append(video)
            except KeyError as e:
                app.logger.error(f"Missing key {e} in video details for video ID: {item.get('id', 'unknown')}")
                continue

        # Sort by published date (most recent first)
        videos.sort(key=lambda x: x['published_at'], reverse=True)
        videos = videos[:15]  # Ensure max 15 videos

        app.logger.info(f"Found {len(videos)} videos for channel {channel_id}")
        return jsonify({
            'success': True,
            'channel_id': channel_id,
            'channel_title': channel_title,
            'total_results': len(videos),
            'videos': videos
        })

    except HttpError as e:
        app.logger.error(f"YouTube API error in ideas_by_channel_id for channel {channel_id}: {str(e)}")
        return jsonify({'error': f'YouTube API error: {str(e)}'}), 503
    except Exception as e:
        app.logger.error(f"Ideas by channel ID error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/get_viral_thumbnails', methods=['POST'])
def viral_thumbnails():
    """Fetch 10 viral long-video thumbnail URLs from YouTube based on a query."""
    try:
        data = request.get_json()
        app.logger.debug(f"Received data: {data}")
        query = data.get('query')

        if not query:
            return jsonify({'error': 'Query is required'}), 400

        result = get_viral_thumbnails(query)
        return jsonify(result)
    
    except Exception as e:
        app.logger.error(f"Get viral thumbnails error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/similar_channels', methods=['POST'])
def similar_channels():
    """Fetch up to 15 similar channels based on a given channel ID, including descriptions."""
    try:
        data = request.get_json()
        if not data or 'channel_id' not in data:
            return jsonify({'error': 'Channel ID is required'}), 400
        
        channel_id = data['channel_id'].strip()
        if not channel_id:
            return jsonify({'error': 'Channel ID cannot be empty'}), 400
        
        # Verify the input channel exists
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY, cache_discovery=False)
        channel_request = youtube.channels().list(
            part='id,snippet',
            id=channel_id
        )
        response = channel_request.execute()
        if not response['items']:
            app.logger.error(f"Channel {channel_id} not found")
            return jsonify({'error': 'Channel not found'}), 404
        input_channel_title = response['items'][0]['snippet']['title']
        
        # Get top videos from the input channel
        top_videos = get_top_videos(channel_id, n=5)
        if not top_videos:
            app.logger.error(f"No top videos found for channel {channel_id}")
            return jsonify({'error': 'No top videos found for the channel'}), 404
        
        # Collect related videos to find similar channels
        related_channel_ids = set()
        for video in top_videos:
            video_id = video['video_id']
            related_videos = get_related_videos(video_id, m=25)
            for vid in related_videos:
                if vid.get('channel_id') and vid['channel_id'] != channel_id and len(related_channel_ids) < 15:
                    related_channel_ids.add(vid['channel_id'])
            app.logger.debug(f"Found {len(related_channel_ids)} unique channel IDs after processing video {video_id}")
        
        # If insufficient channels, fall back to keyword search based on top video titles
        if len(related_channel_ids) < 15:
            app.logger.info(f"Only {len(related_channel_ids)} related channels found for channel {channel_id}, falling back to keyword search")
            for video in top_videos:
                title_words = [word for word in video['title'].split() if len(word) > 3 and word.lower() not in ['the', 'and', 'video', 'in', 'to', 'for']]
                query = ' '.join(title_words[:3])[:50]
                if query:
                    search_vids = search_videos_by_query(query, max_results=25)
                    for vid in search_vids:
                        if vid.get('channel_id') and vid['channel_id'] != channel_id and len(related_channel_ids) < 15:
                            related_channel_ids.add(vid['channel_id'])
                    app.logger.debug(f"Found {len(related_channel_ids)} unique channel IDs after keyword search for query '{query}'")
        
        if not related_channel_ids:
            app.logger.error(f"No similar channels found for channel {channel_id} after related videos and keyword search")
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
                'subscriber_count': int(item['statistics'].get('subscriberCount', 0)),
                'description': item['snippet'].get('description', '')  # Added channel description
            })
        
        if not similar_channels:
            app.logger.error(f"No channel details retrieved for channel IDs: {related_channel_ids}")
            return jsonify({'error': 'No similar channels found after fetching details'}), 404
        
        app.logger.info(f"Found {len(similar_channels)} similar channels for channel {channel_id}")
        return jsonify({
            'success': True,
            'input_channel_id': channel_id,
            'input_channel_title': input_channel_title,
            'total_results': len(similar_channels),
            'similar_channels': similar_channels
        })
    except HttpError as e:
        app.logger.error(f"YouTube API error in similar_channels for channel {channel_id}: {str(e)}")
        return jsonify({'success': False, 'error': f'YouTube API error: {str(e)}'}), 503
    except Exception as e:
        app.logger.error(f"Unexpected error in similar_channels for channel {channel_id}: {str(e)}")
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

@app.route('/api/refine_title', methods=['POST'])
def refine_title():
    """Refine a YouTube title by making it shorter or longer and adjust virality score."""
    try:
        data = request.get_json(force=1, silent=False)  # Add strict=False to handle single quotes
        if not data or 'title' not in data or 'action' not in data or 'previous_virality_score' not in data:
            return jsonify({'error': 'Title, action (shorter/longer), and previous_virality_score are required'}), 400
        
        title = data['title'].strip()
        action = data['action'].strip().lower()
        previous_virality_score = int(data['previous_virality_score'])
        if not title or action not in ['shorter', 'longer'] or previous_virality_score < 0 or previous_virality_score > 100:
            return jsonify({'error': 'Invalid input: title cannot be empty, action must be "shorter" or "longer", and previous_virality_score must be between 0 and 100, make sure to avoid using special characters or excessive punctuation also make sure emotion of sentence remain same.'}), 400

        # Prepare prompt for Gemini, including the previous virality score
        prompt = (
            f"Refine the YouTube video title: '{title}'. "
            f"The previous virality score was {previous_virality_score} out of 100. "
            f"Make it {'shorter' if action == 'shorter' else 'longer'} while keeping it engaging and viral. "
            f"Ensure the refined title retains the core idea, uses attention-grabbing words, and fits YouTube best practices. "
            f"Provide the refined title and a new virality score out of 100 based on its appeal, length adjustment, and aiming to improve or maintain the previous virality score. "
            f"Return the response in valid JSON format wrapped in ```json\n...\n``` with the structure: "
            f"{{ \"refined_title\": \"string\", \"virality_score\": int }}"
        )
        
        client = genai.GenerativeModel('gemini-2.0-flash')
        response = client.generate_content(prompt)
        gemini_response = response.text
        if gemini_response.startswith('```json\n') and gemini_response.endswith('\n```'):
            gemini_response = gemini_response[7:-4]
        gemini_data = json.loads(gemini_response)
        
        return jsonify({
            'success': True,
            'original_title': title,
            'action': action,
            'previous_virality_score': previous_virality_score,
            'refined_title': gemini_data['refined_title'],
            'virality_score': gemini_data['virality_score']
        })
    except json.JSONDecodeError as e:
        app.logger.error(f"JSON decode error in refine_title: {str(e)}")
        return jsonify({'success': False, 'error': 'Invalid JSON format in request body'}), 400
    except Exception as e:
        error_message = str(e)
        app.logger.error(f"Refine title error: {error_message}")
        if "API key" in error_message or "authentication" in error_message.lower():
            return jsonify({
                'success': False,
                'error': 'API key or authentication error. Please renew the API key and restart the application.'
            }), 500
        return jsonify({'success': False, 'error': f'An error occurred: {error_message}'}), 500

@app.route('/api/refine_description', methods=['POST'])
def refine_description():
    """Refine a YouTube video description by making it shorter or longer."""
    try:
        data = request.get_json()
        if not data or 'description' not in data or 'action' not in data:
            return jsonify({'error': 'Description and action (shorter/longer) are required'}), 400
        
        description = data['description'].strip()
        action = data['action'].strip().lower()
        if not description or action not in ['shorter', 'longer']:
            return jsonify({'error': 'Invalid input: description cannot be empty, action must be "shorter" or "longer"'}), 400
        
        # Prepare prompt for Gemini to refine the description
        prompt = (
            f"Refine the YouTube video description: '{description}'. "
            f"Make it {'shorter' if action == 'shorter' else 'longer'} while keeping it engaging, informative, and aligned with YouTube best practices. "
            f"Ensure the refined description retains the core idea and enhances viewer interest. "
            f"Return the response in valid JSON format wrapped in ```json\n...\n``` with the structure: "
            f"{{ 'refined_description': 'string' }}"
        )
        
        client = genai.GenerativeModel('gemini-2.0-flash')
        response = client.generate_content(prompt)
        gemini_response = response.text
        if gemini_response.startswith('```json\n') and gemini_response.endswith('\n```'):
            gemini_response = gemini_response[7:-4]
        gemini_data = json.loads(gemini_response)
        
        return jsonify({
            'success': True,
            'original_description': description,
            'action': action,
            'refined_description': gemini_data['refined_description']
        })
    except Exception as e:
        app.logger.error(f"Refine description error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/generate_script', methods=['POST'])
def generate_script_api():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Validate required fields
        text = data.get('text')
        duration = data.get('duration')  # in minutes
        groups = data.get('groups', [])  # Array of group objects with name and videos

        if not text or not duration:
            return jsonify({'error': 'Text and duration are required'}), 400

        # Convert duration to float and validate
        duration = float(duration)
        if duration <= 0:
            return jsonify({'error': 'Duration must be a positive number'}), 400

        # Optional parameters with defaults
        wpm = data.get('wpm', 145)  # default to 145 WPM
        creator_name = data.get('creator_name', 'YourChannelName')
        audience = data.get('audience', 'beginners')
        language = data.get('language', 'en')  # default to English

        # Prepare data in the format expected by generate_script
        script_data = {
            'text': text,
            'duration': duration,
            'groups': groups
        }

        # Generate the script using the updated generate_script function
        script = generate_script(script_data)

        # Return the generated script as JSON response
        return jsonify({
            'success': True,
            'script': script
        })
    except Exception as e:
        app.logger.error(f"Generate script error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/shorts_outliers', methods=['POST'])
def shorts_outliers():
    """Fetch random trending outlier YouTube Shorts (<= 60 seconds)."""
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
        
        # Fetch trending videos using multiple queries
        trending_queries = [
            "trending shorts", "viral shorts", "popular shorts", "tiktok style", "quick tips", "short challenge"
        ]
        all_videos = []
        for query in trending_queries[:3]:  # Use top 3 queries
            videos = get_random_trending_videos(max_results=150, query=query)
            all_videos.extend(videos)
        
        # Deduplicate videos
        video_id_to_data = {v['video_id']: v for v in all_videos}
        all_videos = list(video_id_to_data.values())
        if not all_videos:
            app.logger.warning("No trending Shorts retrieved, check internet or API key")
            return jsonify({'error': 'No trending Shorts found, check internet connection or API key'}), 503
        
        # Filter to include only Shorts
        similarity_finder = UltraFastYouTubeSimilarity()
        raw_videos = [[v['video_id'], v['title'], v['thumbnail_url']] for v in all_videos]
        filtered_videos = similarity_finder._filter_shorts_only(raw_videos, max_duration=60)
        shorts_videos = [video_id_to_data[video[0]] for video in filtered_videos if video[0] in video_id_to_data]
        
        app.logger.debug(f"Fetched {len(shorts_videos)} trending Shorts after filtering")
        
        # Fallback if fewer than 100 Shorts
        if len(shorts_videos) < 100:
            app.logger.info("Insufficient Shorts, fetching more with fallback query")
            extra_videos = get_random_trending_videos(max_results=150, query="shorts")
            extra_raw = [[v['video_id'], v['title'], v['thumbnail_url']] for v in extra_videos]
            extra_filtered = similarity_finder._filter_shorts_only(extra_raw, max_duration=60)
            for video in extra_filtered:
                video_id = video[0]
                if video_id not in video_id_to_data:
                    video_id_to_data[video_id] = extra_videos[extra_raw.index(video)]
                    shorts_videos.append(video_id_to_data[video_id])
        
        # Process videos for outliers
        video_ids = [v['video_id'] for v in shorts_videos]
        video_stats = get_video_stats(video_ids)
        channel_ids = list(set(v['channel_id'] for v in shorts_videos))
        processed_channels = {}
        for channel_id in channel_ids:
            avg_views, subscriber_count = calculate_channel_average_views(channel_id)
            processed_channels[channel_id] = {'average_views': avg_views, 'subscriber_count': subscriber_count}
        
        threshold = 1.5
        max_attempts = 3
        attempt = 0
        outliers = []
        
        while len(outliers) < 35 and attempt < max_attempts:
            outliers = []
            for video in shorts_videos:
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
                
                if multiplier <= threshold:
                    continue
                
                duration_seconds = parse_duration(stats.get('duration', 'PT0S'))
                try:
                    pub_date = datetime.strptime(video['published_at'], '%Y-%m-%dT%H:%M:%SZ')
                    video_age_days = (datetime.utcnow() - pub_date).days
                except:
                    video_age_days = 0
                
                video_result = {
                    'video_id': video_id,
                    'title': video['title'],
                    'channel_id': channel_id,
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
                    'viral_score': round(multiplier / 10, 2),
                    'engagement_rate': round((stats['likes'] + stats['comments']) / stats['views'] if stats['views'] > 0 else 0, 4),
                    'subscriber_count': channel_data.get('subscriber_count', 0),
                    'language': video.get('language', 'en')
                }
                outliers.append(video_result)
            
            outliers.sort(key=lambda x: x['multiplier'], reverse=True)
            if len(outliers) > 50:
                outliers = outliers[:50]
            
            attempt += 1
            if len(outliers) < 35 and attempt < max_attempts:
                threshold -= 0.2
                app.logger.debug(f"Attempt {attempt}: Found {len(outliers)} outliers, lowering threshold to {threshold}")
        
        # Fallback: Fetch related Shorts if still insufficient
        if len(outliers) < 35:
            app.logger.info("Insufficient outliers, fetching related Shorts")
            top_videos = [v for v in shorts_videos[:5]]  # Use top 5 Shorts by views
            for video in top_videos:
                rel_vids = get_related_videos(video['video_id'], m=25)
                rel_raw = [[v['video_id'], v['title'], v['thumbnail_url']] for v in rel_vids]
                rel_filtered = similarity_finder._filter_shorts_only(rel_raw, max_duration=60)
                for rel_video in rel_filtered:
                    video_id = rel_video[0]
                    if video_id not in video_id_to_data:
                        rel_video_data = next((v for v in rel_vids if v['video_id'] == video_id), None)
                        if rel_video_data:
                            video_id_to_data[video_id] = rel_video_data
                            shorts_videos.append(rel_video_data)
            
            # Re-run outlier detection
            video_ids = [v['video_id'] for v in shorts_videos]
            video_stats = get_video_stats(video_ids)
            channel_ids = list(set(v['channel_id'] for v in shorts_videos))
            for channel_id in channel_ids:
                if channel_id not in processed_channels:
                    avg_views, subscriber_count = calculate_channel_average_views(channel_id)
                    processed_channels[channel_id] = {'average_views': avg_views, 'subscriber_count': subscriber_count}
            
            outliers = []
            for video in shorts_videos:
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
                
                if multiplier <= threshold:
                    continue
                
                duration_seconds = parse_duration(stats.get('duration', 'PT0S'))
                try:
                    pub_date = datetime.strptime(video['published_at'], '%Y-%m-%dT%H:%M:%SZ')
                    video_age_days = (datetime.utcnow() - pub_date).days
                except:
                    video_age_days = 0
                
                video_result = {
                    'video_id': video_id,
                    'title': video['title'],
                    'channel_id': channel_id,
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
                    'viral_score': round(multiplier / 10, 2),
                    'engagement_rate': round((stats['likes'] + stats['comments']) / stats['views'] if stats['views'] > 0 else 0, 4),
                    'subscriber_count': channel_data.get('subscriber_count', 0),
                    'language': video.get('language', 'en')
                }
                outliers.append(video_result)
            
            outliers.sort(key=lambda x: x['multiplier'], reverse=True)
            if len(outliers) > 50:
                outliers = outliers[:50]
        
        # Format response
        outlier_detector = OutlierDetector()
        formatted_outliers = []
        for video in outliers:
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
                'likes': video['likes'],
                'likes_formatted': outlier_detector.format_number(video['likes']),
                'comments': video['comments'],
                'comments_formatted': outlier_detector.format_number(video['comments']),
                'duration': video['duration'],
                'duration_seconds': video['duration_seconds'],
                'url': video['url'],
                'published_at': video['published_at'],
                'viral_score': video['viral_score'],
                'engagement_rate': video['engagement_rate'],
                'thumbnail_url': video['thumbnail_url'],
                'subscriber_count': video['subscriber_count'],
                'language': video['language']
            }
            formatted_outliers.append(formatted_video)
        
        search_history.append({
            'query': 'shorts_outliers',
            'timestamp': datetime.now().isoformat(),
            'results_count': len(formatted_outliers)
        })
        
        app.logger.info(f"Found {len(formatted_outliers)} trending Shorts outliers")
        return jsonify({
            'success': True,
            'total_results': len(formatted_outliers),
            'outliers': formatted_outliers
        })
    except Exception as e:
        app.logger.error(f"Shorts outliers error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/similar_shorts', methods=['POST'])
def similar_shorts():
    """Fetch up to 15 similar YouTube Shorts (≤ 90 seconds) for a given video ID using a title-based search."""
    try:
        # Validate input
        data = request.get_json()
        if not data or 'video_id' not in data:
            return jsonify({'error': 'Video ID is required'}), 400
        
        video_id = data['video_id'].strip()
        if not video_id:
            return jsonify({'error': 'Video ID cannot be empty'}), 400
        
        if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY':
            app.logger.error("YouTube API key not configured.")
            return jsonify({'error': 'YouTube API key not configured'}), 500

        # Initialize YouTube API client
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        
        # Step 1: Fetch input video details
        video_response = youtube.videos().list(
            part='snippet',
            id=video_id
        ).execute()
        
        if not video_response['items']:
            app.logger.error(f"Video not found for ID: {video_id}")
            return jsonify({'error': 'Video not found'}), 404
        
        input_video = video_response['items'][0]
        input_title = input_video['snippet']['title']
        input_channel_id = input_video['snippet']['channelId']
        input_channel_title = input_video['snippet']['channelTitle']
        
        # Step 2: Generate search query from the title
        query = ' '.join([word for word in input_title.split() if len(word) > 3 and word.lower() not in ['the', 'and', 'video']])
        if not query:
            app.logger.error(f"No valid search query generated from title: {input_title}")
            return jsonify({'error': 'No valid search query generated from title'}), 404
        
        # Step 3: Search for similar videos
        search_response = youtube.search().list(
            part='snippet',
            q=query,
            type='video',
            maxResults=25,  # Fetch extra results to allow filtering for Shorts
            order='relevance'
        ).execute()
        
        if not search_response.get('items', []):
            app.logger.error(f"No similar videos found for query: {query}")
            return jsonify({'error': 'No similar videos found'}), 404
        
        # Step 4: Filter results for Shorts (≤ 90 seconds)
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
        
        # Step 5: Fetch additional details for videos
        if not video_ids:
            app.logger.error("No valid video IDs after filtering.")
            return jsonify({'error': 'No similar videos found from different channels'}), 404

        details_response = youtube.videos().list(
            part='snippet,statistics,contentDetails',
            id=','.join(video_ids[:25])  # Limit to 25 IDs per API call
        ).execute()
        
        video_stats = {}
        for item in details_response.get('items', []):
            duration = item['contentDetails']['duration']
            duration_seconds = parse_duration(duration)
            if duration_seconds <= 90:  # Filter for Shorts
                video_stats[item['id']] = {
                    'views': int(item['statistics'].get('viewCount', 0)),
                    'likes': int(item['statistics'].get('likeCount', 0)),
                    'comments': int(item['statistics'].get('commentCount', 0)),
                    'duration': duration,
                    'duration_seconds': duration_seconds,
                    'title': item['snippet']['title'],
                    'channel_id': item['snippet']['channelId'],
                    'channel_title': item['snippet']['channelTitle'],
                    'published_at': item['snippet']['publishedAt'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url'],
                    'language': item['snippet'].get('defaultLanguage', 'en')
                }
        
        # Step 6: Fetch subscriber count for each channel
        channel_ids = list(set(video['channel_id'] for video in video_stats.values()))
        processed_channels = {}
        channel_video_counts = {}  # Track number of videos per channel
        channel_total_views = {}  # Track total views per channel
        for channel_id in channel_ids:
            try:
                channel_response = youtube.channels().list(
                    part='statistics',
                    id=channel_id
                ).execute()
                channel_data = channel_response['items'][0]['statistics'] if channel_response['items'] else {}
                subscriber_count = int(channel_data.get('subscriberCount', 0))
                processed_channels[channel_id] = {'subscriber_count': subscriber_count}
            except Exception as e:
                app.logger.error(f"Error fetching subscriber count for channel {channel_id}: {str(e)}")
                processed_channels[channel_id] = {'subscriber_count': 0}
        
        # Step 7: Calculate channel average views
        for vid, stats in video_stats.items():
            channel_id = stats['channel_id']
            channel_video_counts[channel_id] = channel_video_counts.get(channel_id, 0) + 1
            channel_total_views[channel_id] = channel_total_views.get(channel_id, 0) + stats['views']
        
        for channel_id in channel_video_counts:
            avg_views = channel_total_views[channel_id] / channel_video_counts[channel_id] if channel_video_counts[channel_id] > 0 else 0
            processed_channels[channel_id]['channel_avg_views'] = int(avg_views)
            processed_channels[channel_id]['channel_avg_views_formatted'] = format_number(int(avg_views))
        
        # Step 8: Format the response
        formatted_shorts = []
        for vid, stats in video_stats.items():
            channel_data = processed_channels.get(stats['channel_id'], {'subscriber_count': 0, 'channel_avg_views': 0, 'channel_avg_views_formatted': '0'})
            engagement_rate = (stats['likes'] + stats['comments']) / stats['views'] if stats['views'] > 0 else 0
            multiplier = engagement_rate * (stats['views'] / 1000) if stats['views'] > 0 else 0  # Example multiplier calculation
            formatted_short = {
                'video_id': vid,
                'title': stats['title'],
                'channel_id': stats['channel_id'],
                'channel_title': stats['channel_title'],
                'views': stats['views'],
                'views_formatted': format_number(stats['views']),
                'likes': stats['likes'],
                'likes_formatted': format_number(stats['likes']),
                'comments': stats['comments'],
                'comments_formatted': format_number(stats['comments']),
                'duration': stats['duration'],
                'duration_seconds': stats['duration_seconds'],
                'url': f"https://www.youtube.com/watch?v={vid}",
                'published_at': stats['published_at'],
                'thumbnail_url': stats['thumbnail_url'],
                'engagement_rate': round(engagement_rate, 4),
                'language': stats['language'],
                'multiplier': round(multiplier, 4),
                'channel_avg_views': channel_data['channel_avg_views'],
                'channel_avg_views_formatted': channel_data['channel_avg_views_formatted'],
                'subscriber_count': channel_data['subscriber_count']
            }
            formatted_shorts.append(formatted_short)
        
        # Sort by views (relevance) and limit to 15
        formatted_shorts.sort(key=lambda x: x['views'], reverse=True)
        formatted_shorts = formatted_shorts[:15]
        
        if not formatted_shorts:
            app.logger.error("No valid Shorts after processing.")
            return jsonify({'error': 'No similar Shorts found'}), 404
        
        # Return successful response
        app.logger.info(f"Found {len(formatted_shorts)} similar Shorts for video {video_id}")
        return jsonify({
            'success': True,
            'input_video_id': video_id,
            'input_video_title': input_title,
            'input_channel_title': input_channel_title,
            'total_results': len(formatted_shorts),
            'similar_shorts': formatted_shorts
        })
    
    except HttpError as e:
        app.logger.error(f"YouTube API error in similar_shorts for video {video_id}: {str(e)}")
        return jsonify({'success': False, 'error': f'YouTube API error: {str(e)}'}), 503
    except Exception as e:
        app.logger.error(f"Similar shorts error: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/api/shorts_by_channel_id', methods=['POST'])
def shorts_by_channel_id():
    """Find competitors using last 5 Shorts of input channel, then fetch 10 Shorts (5 latest + 5 popular) per competitor."""
    try:
        data = request.get_json()
        if not data or 'channel_id' not in data:
            return jsonify({'error': 'Channel ID is required'}), 400

        channel_id = data['channel_id'].strip()
        if not channel_id:
            return jsonify({'error': 'Channel ID cannot be empty'}), 400

        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

        # STEP 1: Get input channel details
        channel_resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()
        if not channel_resp.get("items"):
            return jsonify({'error': 'Channel not found'}), 404

        channel_title = channel_resp['items'][0]['snippet']['title']
        uploads_playlist = channel_resp['items'][0]['contentDetails']['relatedPlaylists']['uploads']

        # STEP 2: Get last 5 Shorts from input channel
        playlist_items = youtube.playlistItems().list(
            part="snippet,contentDetails",
            playlistId=uploads_playlist,
            maxResults=15
        ).execute()

        input_shorts = []
        for item in playlist_items.get("items", []):
            video_id = item["contentDetails"]["videoId"]

            video_resp = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            ).execute()
            if not video_resp.get("items"):
                continue
            vid = video_resp["items"][0]

            # check if it's a Short (<=60s)
            duration_str = vid['contentDetails']['duration']
            match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            total_seconds = hours * 3600 + minutes * 60 + seconds
            if total_seconds <= 60:
                input_shorts.append(vid)

            if len(input_shorts) >= 5:
                break

        if not input_shorts:
            return jsonify({'error': 'No Shorts found for input channel'}), 404

        # STEP 3: Search competitors using titles of input Shorts
        competitor_channels = {}
        for short in input_shorts:
            query = short["snippet"]["title"]
            search_results = youtube.search().list(
                part="snippet",
                q=query,
                type="video",
                maxResults=20,
                videoDuration="short"
            ).execute()

            for item in search_results.get("items", []):
                comp_cid = item["snippet"]["channelId"]
                comp_ctitle = item["snippet"]["channelTitle"]

                if comp_cid == channel_id or comp_ctitle == channel_title:
                    continue

                if comp_cid not in competitor_channels:
                    competitor_channels[comp_cid] = comp_ctitle

                if len(competitor_channels) >= 20:
                    break
            if len(competitor_channels) >= 20:
                break

        competitors_data = []

        # STEP 4: For each competitor channel, fetch 10 Shorts
        for comp_id, comp_title in competitor_channels.items():
            comp_channel_resp = youtube.channels().list(
            part="statistics",
            id=comp_id
            ).execute()
            subs_count = int(comp_channel_resp['items'][0]['statistics'].get('subscriberCount', 0))
            # Get latest Shorts
            latest_search = youtube.search().list(
                part="id,snippet",
                channelId=comp_id,
                type="video",
                maxResults=10,
                order="date",
                videoDuration="short"
            ).execute()
            subs_count = int(comp_channel_resp['items'][0]['statistics'].get('subscriberCount', 0))

            latest_ids = [v["id"]["videoId"] for v in latest_search.get("items", [])]

            # Get most popular Shorts
            popular_search = youtube.search().list(
                part="id,snippet",
                channelId=comp_id,
                type="video",
                maxResults=10,
                order="viewCount",
                videoDuration="short"
            ).execute()
            popular_ids = [v["id"]["videoId"] for v in popular_search.get("items", [])]

            video_ids = list(set(latest_ids[:5] + popular_ids[:5]))
            if not video_ids:
                continue

            video_resp = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(video_ids)
            ).execute()

            shorts = []
            for item in video_resp.get("items", []):
                # duration filter
                duration_str = item['contentDetails']['duration']
                match = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", duration_str)
                hours = int(match.group(1) or 0)
                minutes = int(match.group(2) or 0)
                seconds = int(match.group(3) or 0)
                total_seconds = hours * 3600 + minutes * 60 + seconds
                if total_seconds > 60:
                    continue

                views = int(item['statistics'].get('viewCount', 0))
                likes = int(item['statistics'].get('likeCount', 0)) if "likeCount" in item["statistics"] else 0
                comments = int(item['statistics'].get('commentCount', 0)) if "commentCount" in item["statistics"] else 0
                engagement_rate = round((likes + comments) / views, 4) if views > 0 else 0

                shorts.append({
                    "video_id": item['id'],
                    "title": item['snippet']['title'],
                    "views": views,
                    "likes": likes,
                    "comments": comments,
                    "multiplier": round(views / subs_count, 1) if subs_count > 0 else 0,
                    "engagement_rate": engagement_rate,
                    "duration_seconds": total_seconds,
                    "published_at": item['snippet']['publishedAt'],
                    "thumbnail_url": item['snippet']['thumbnails']['high']['url'],
                    "url": f"https://www.youtube.com/watch?v={item['id']}"
                })

            competitors_data.append({
                "channel_id": comp_id,
                "channel_title": comp_title,
                "shorts": shorts[:10]
            })

        return jsonify({
            "success": True,
            "input_channel": {"id": channel_id, "title": channel_title},
            "competitors_count": len(competitors_data),
            "competitors": competitors_data
        })

    except Exception as e:
        app.logger.error(f"Shorts competitors error: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500



@app.route('/api/generate-script-from-title', methods=['POST'])
def generate_script_from_title_endpoint():
    """API endpoint to generate a video script from a title and duration."""
    try:
        data = request.get_json()
        title = data.get('title')
        duration = data.get('duration')
        if not title or not duration:
            return jsonify({'error': 'Title and duration are required'}), 400
        try:
            duration = float(duration)
            if duration <= 0:
                raise ValueError("Duration must be a positive number")
        except (ValueError, TypeError):
            return jsonify({'error': 'Duration must be a valid positive number'}), 400
        # Optional parameters
        wpm = data.get('wpm', 145)
        creator_name = data.get('creator_name', 'YourChannelName')
        audience = data.get('audience', 'beginners')
        language = data.get('language', 'en')
        script, _, error = generate_script_from_title(
            title=title,
            duration=duration,
            wpm=wpm,
            creator_name=creator_name,
            audience=audience,
            language=language
        ), title, None  # Mimic find_similar_videos_enhanced return structure
        if error:
            return jsonify({'success': False, 'error': error}), 400
        result_data = {
            'title': title,
            'duration': duration,
            'script': script,
            'wpm': wpm,
            'creator_name': creator_name,
            'audience': audience,
            'language': language
        }
        return jsonify({'success': True, **result_data})
    except Exception as e:
        print(f"Error in generate_script_from_title_endpoint: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500


@app.route('/api/review-thumbnail', methods=['POST'])
def review_thumbnail_endpoint():
    """API endpoint to review a thumbnail image."""
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        if not image_url:
            return jsonify({'error': 'Image URL is required'}), 400
        review_data, error = review_thumbnail(image_url)
        if error:
            return jsonify({'success': False, 'error': error}), 400
        return jsonify({'success': True, **review_data})
    except Exception as e:
        print(f"Error in review_thumbnail_endpoint: {str(e)}")
        return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
