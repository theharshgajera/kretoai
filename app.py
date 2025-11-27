import traceback
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
import google.generativeai as genai
from googleapiclient.discovery import build
from sklearn import logger
import io
import logging
from isodate import parse_duration
from datetime import datetime
from uuid import uuid4
import json
from youtube_api import get_viral_thumbnails
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import isodate
from script_generate import generate_script_from_titleshorts
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
import uuid
from datetime import datetime
from collections import defaultdict
import threading
from urllib.parse import urlparse, parse_qs
import logging
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import tempfile
import fitz
from pathlib import Path

load_dotenv()

# ========================================
# CONFIGURE UPLOAD FOLDER FIRST
# ========================================
UPLOAD_FOLDER = r'D:\poppy AI\kretoai\tempfolder'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload folder exists BEFORE any other initialization
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder configured: {UPLOAD_FOLDER}")
print(f"Upload folder exists: {os.path.exists(UPLOAD_FOLDER)}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set app config immediately
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")

# Configure logging
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)

# NOW import from script.py (after app config is set)
from script import (
    DocumentProcessor, 
    VideoProcessor, 
    EnhancedScriptGenerator,
    facebook_processor,
    instagram_processor,
    user_data
)


# Initialize processors
document_processor = DocumentProcessor()
video_processor = VideoProcessor()
script_generator = EnhancedScriptGenerator()

# Load API keys
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'your_youtube_api_key')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'your_gemini_api_key')
ACCESS_TOKEN = os.getenv('TRENDING_ACCESS_TOKEN', 'your_secure_token_123')

genai.configure(api_key=GEMINI_API_KEY)



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


def safe_duration(video):
    try:
        return parse_duration(video.get("contentDetails", {}).get("duration", "PT0S"))
    except Exception:
        return 0

def format_number(num):
    return "{:,}".format(num)

@app.route("/api/channel_outliers_by_id", methods=["POST"])
def channel_outliers_by_id():
    try:
        data = request.get_json()
        channel_id = data.get("channel_id", "").strip()
        if not channel_id:
            return jsonify({"error": "Channel ID is required"}), 400

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # 1. Fetch channel info
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

        # 2. Get last 10 uploads, filter medium/long → take 10
        videos_resp = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist,
            maxResults=20
        ).execute()

        candidate_videos = []
        for item in videos_resp.get("items", []):
            vid_id = item["contentDetails"]["videoId"]
            vid_details = youtube.videos().list(
                part="contentDetails,snippet,statistics",
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

        if not candidate_videos:
            return jsonify({"error": "No medium/long videos found for this channel"}), 404

        # 3. Collect competitor channels with frequency count
        competitor_counts = {}
        for video in candidate_videos:
            search_resp = youtube.search().list(
                part="snippet",
                q=video["title"],
                type="video",
                maxResults=50,
                relevanceLanguage="en",
                videoDuration="medium"  # filter medium/long
            ).execute()

            for item in search_resp.get("items", []):
                ch_id = item["snippet"]["channelId"]
                ch_title = item["snippet"]["channelTitle"]

                if ch_id == channel_id:
                    continue

                if ch_id not in competitor_counts:
                    competitor_counts[ch_id] = {"title": ch_title, "count": 0}
                competitor_counts[ch_id]["count"] += 1

        # sort competitors by frequency (descending) and take top 20
        competitors = sorted(
            competitor_counts.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True
        )[:50]

        # 4. For each competitor, fetch top + latest videos with avg views
        all_videos = []
        competitors_meta = []
        for comp_id, comp_data in competitors:
            comp_resp = youtube.channels().list(
                part="statistics,contentDetails",
                id=comp_id
            ).execute()

            if not comp_resp.get("items"):
                continue

            comp_info = comp_resp["items"][0]
            comp_title = comp_data["title"]
            comp_subs = int(comp_info["statistics"].get("subscriberCount", 1))
            comp_uploads = comp_info["contentDetails"]["relatedPlaylists"]["uploads"]

            # Fetch up to 50 recent uploads
            comp_uploads_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=comp_uploads,
                maxResults=50
            ).execute()

            comp_video_ids = [v["contentDetails"]["videoId"] for v in comp_uploads_resp.get("items", [])]
            
            comp_videos_resp = youtube.videos().list(
                part="contentDetails,snippet,statistics",
                id=",".join(comp_video_ids)
            ).execute()
            comp_videos = comp_videos_resp.get("items", [])

            if not comp_videos:
                continue

            # calculate avg recent views
            avg_recent_views = sum(
                int(v["statistics"].get("viewCount", 0)) for v in comp_videos
            ) / len(comp_videos)

            # Sort by views → Top 5
            top_videos = sorted(
                comp_videos,
                key=lambda x: int(x["statistics"].get("viewCount", 0)),
                reverse=True
            )[:5]

            # Latest 5 (with ratio > 1)
            latest_videos = []
            for v in comp_videos[:10]:  # last 10 uploads
                views = int(v["statistics"].get("viewCount", 0))
                if avg_recent_views > 0 and (views / avg_recent_views) > 1:
                    latest_videos.append(v)
                if len(latest_videos) >= 5:
                    break

            # Add competitor metadata
            competitors_meta.append({
                "id": comp_id,
                "title": comp_title,
                "subscriber_count": comp_subs,
                "avg_recent_views": round(avg_recent_views, 2),
                "frequency": comp_data.get("count", 0)
            })

            # Add formatted video data
            for v in top_videos + latest_videos:
                duration = parse_duration(v["contentDetails"]["duration"])
                views = int(v["statistics"].get("viewCount", 0))
                multiplier = round(views / avg_recent_views, 2) if avg_recent_views > 0 else 0

                all_videos.append({
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel_id": comp_id,
                    "channel_title": comp_title,
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "multiplier": multiplier,
                    "avg_recent_views": round(avg_recent_views, 2),
                    "thumbnail_url": v["snippet"]["thumbnails"]["high"]["url"],
                    "url": f"https://www.youtube.com/watch?v={v['id']}"
                })

            if len(all_videos) >= 200:
                break

        return jsonify({
            "success": True,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "competitors": competitors_meta,
            "total_videos": len(all_videos),
            "videos": all_videos[:200]
        })

    except Exception as e:
        app.logger.error(f"Channel outliers error: {str(e)}")
        return jsonify({"error": str(e)}), 500




# ------------------ Fetch Videos ------------------
def get_channel_videos(youtube, channel_id, order_by):
    """Fetch videos from a given channel with proper error handling."""
    videos = []
    next_page_token = None
    total_fetched = 0

    try:
        while len(videos) < 50:  # limit for one channel
            search_response = youtube.search().list(
                part="snippet",
                channelId=channel_id,
                maxResults=50,
                order=order_by,
                type="video",
                pageToken=next_page_token
            ).execute()

            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
            if not video_ids:
                break

            video_response = youtube.videos().list(
                part="snippet,statistics,contentDetails",
                id=",".join(video_ids)
            ).execute()

            for video in video_response.get("items", []):
                snippet = video.get("snippet", {})
                stats = video.get("statistics", {})
                videos.append({
                    "video_id": video["id"],
                    "title": snippet.get("title"),
                    "publishedAt": snippet.get("publishedAt"),
                    "channelId": snippet.get("channelId"),
                    "channelTitle": snippet.get("channelTitle"),
                    "views": int(stats.get("viewCount", 0)),
                    "likes": int(stats.get("likeCount", 0)),
                    "thumbnails": snippet.get("thumbnails", {}),
                    "url": f"https://www.youtube.com/watch?v={video['id']}"
                })

            total_fetched += len(video_ids)
            next_page_token = search_response.get("nextPageToken")
            if not next_page_token:
                break

    except Exception as e:
        print(f"❌ Error fetching videos for {channel_id}: {e}")
        traceback.print_exc()

    print(f"✅ {len(videos)} videos fetched from {channel_id} ({order_by})")
    return videos


# ------------------ Merge Logic ------------------
def merge_videos(latest_videos, popular_videos):
    final = []
    used_ids = set()

    latest_by_views = sorted(latest_videos, key=lambda x: x["views"], reverse=True)
    latest_by_date = sorted(latest_videos, key=lambda x: x["publishedAt"])
    popular_by_views = sorted(popular_videos, key=lambda x: x["views"], reverse=True)
    popular_by_date = sorted(popular_videos, key=lambda x: x["publishedAt"], reverse=True)

    while any([latest_by_views, latest_by_date, popular_by_views, popular_by_date]):
        for lst in [latest_by_views, popular_by_date, latest_by_date, popular_by_views]:
            if lst:
                vid = lst.pop(0) if lst in [latest_by_views, popular_by_date] else lst.pop()
                if vid["video_id"] not in used_ids:
                    final.append(vid)
                    used_ids.add(vid["video_id"])
        if len(final) >= 400:
            break

    # enforce 2 videos per channel per 24 batch
    adjusted = []
    for i in range(0, len(final), 24):
        segment = final[i:i + 24]
        counts = {}
        valid = []
        overflow = []
        for v in segment:
            c = v["channelId"]
            if counts.get(c, 0) < 2:
                valid.append(v)
                counts[c] = counts.get(c, 0) + 1
            else:
                overflow.append(v)
        adjusted.extend(valid)
        adjusted.extend(overflow)
    return adjusted[:400]


# ------------------ Endpoint ------------------
# @app.route("/api/video_outliers", methods=["POST"])
# def video_outliers():
#     try:
#         data = request.get_json()
#         channel_ids = data.get("channel_ids") or data.get("competitors", [])
#         if not channel_ids:
#             return jsonify({"error": "No competitors provided"}), 400

#         youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
#         latest_videos_all = []
#         popular_videos_all = []

#         # ------------------------------------------------------------------
#         # 1. FETCH & CLEAN (unchanged)
#         # ------------------------------------------------------------------
#         for ch_id in channel_ids:
#             ch_resp = youtube.channels().list(part="snippet,statistics", id=ch_id).execute()
#             if not ch_resp.get("items"):
#                 continue
#             ch_info = ch_resp["items"][0]
#             ch_title = ch_info["snippet"]["title"]
#             subs_count = int(ch_info["statistics"].get("subscriberCount", 1))

#             # latest
#             search_resp = youtube.search().list(
#                 part="snippet", channelId=ch_id, type="video",
#                 videoDuration="medium", order="date", maxResults=50
#             ).execute()
#             video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
#             latest_videos = fetch_video_details(youtube, video_ids)

#             # popular
#             search_resp_pop = youtube.search().list(
#                 part="snippet", channelId=ch_id, type="video",
#                 order="viewCount", maxResults=50
#             ).execute()
#             popular_ids = [item["id"]["videoId"] for item in search_resp_pop.get("items", [])]
#             popular_videos = fetch_video_details(youtube, popular_ids)

#             # avg recent views
#             avg_recent_views = (
#                 sum(int(v["statistics"].get("viewCount", 0)) for v in latest_videos) / len(latest_videos)
#                 if latest_videos else 1
#             )

#             def clean_video(v):
#                 views = int(v["statistics"].get("viewCount", 0))
#                 duration = parse_duration(v.get("contentDetails", {}).get("duration", "PT0S"))
#                 return {
#                     "video_id": v["id"],
#                     "title": v["snippet"]["title"],
#                     "channel_id": ch_id,
#                     "channel_title": ch_title,
#                     "views": views,
#                     "subscriber_count": subs_count,
#                     "published_at": v["snippet"].get("publishedAt", ""),
#                     "views_formatted": format_number(views),
#                     "duration_seconds": duration,
#                     "avg_recent_views": round(avg_recent_views, 2),
#                     "multiplier": round(views / avg_recent_views, 2) if avg_recent_views > 0 else 0,
#                     "thumbnail_url": v["snippet"]["thumbnails"].get("high", {}).get("url"),
#                     "url": f"https://www.youtube.com/watch?v={v['id']}"

#                 }

#             latest_videos_all.extend([clean_video(v) for v in latest_videos[:50]])
#             popular_videos_all.extend([clean_video(v) for v in popular_videos[:50]])

#         # ------------------------------------------------------------------
#         # 2. PRE-SORT (unchanged)
#         # ------------------------------------------------------------------
#         latest_videos_all.sort(key=lambda x: x["views"], reverse=True)   # most-viewed first
#         popular_videos_all.sort(key=lambda x: x["views"])               # least-viewed first

#         final_list = []
#         channel_counter = {}      # key = f"{channel_id}_{batch_index}"
#         batch_size = 24
#         i = 0
#         MAX_TOTAL = 200

#         # ------------------------------------------------------------------
#         # 3. ORIGINAL MERGE LOOP (with deadlock detection)
#         # ------------------------------------------------------------------
#         while latest_videos_all or popular_videos_all:
#             added_this_round = 0

#             for lst_type in ("latest", "popular"):
#                 if i >= MAX_TOTAL:
#                     break

#                 src = latest_videos_all if lst_type == "latest" else popular_videos_all
#                 if not src:
#                     continue

#                 v = src.pop(0)

#                 batch_index = i // batch_size
#                 key = f"{v['channel_id']}_{batch_index}"
#                 if channel_counter.get(key, 0) >= 2:
#                     # push back for later attempt
#                     src.append(v)
#                     continue

#                 final_list.append(v)
#                 channel_counter[key] = channel_counter.get(key, 0) + 1
#                 i += 1
#                 added_this_round += 1

#             # ------------------------------------------------------------------
#             # 4. DEAD-LOCK → FALLBACK ROUND-ROBIN
#             # ------------------------------------------------------------------
#             if added_this_round == 0 and (latest_videos_all or popular_videos_all):
#                 # Build per-channel queues that still have videos
#                 from collections import defaultdict, deque

#                 channel_latest = defaultdict(deque)
#                 channel_popular = defaultdict(deque)

#                 for v in latest_videos_all:
#                     channel_latest[v["channel_id"]].append(v)
#                 for v in popular_videos_all:
#                     channel_popular[v["channel_id"]].append(v)

#                 all_channels = set(channel_latest) | set(channel_popular)

#                 while i < MAX_TOTAL and all_channels:
#                     placed = False
#                     for ch in list(all_channels):
#                         # try latest first, then popular
#                         src_q = channel_latest[ch] or channel_popular[ch]
#                         if not src_q:
#                             all_channels.discard(ch)
#                             continue

#                         candidate = src_q.popleft()
#                         batch_index = i // batch_size
#                         key = f"{candidate['channel_id']}_{batch_index}"

#                         if channel_counter.get(key, 0) >= 2:
#                             # put it back and skip this channel for now
#                             src_q.appendleft(candidate)
#                             continue

#                         final_list.append(candidate)
#                         channel_counter[key] = channel_counter.get(key, 0) + 1
#                         i += 1
#                         placed = True
#                         # remove channel if both queues are empty now
#                         if not channel_latest[ch] and not channel_popular[ch]:
#                             all_channels.discard(ch)
#                         break   # one video per outer iteration

#                     if not placed:
#                         # No channel could give a video → true deadlock, break out
#                         break

#             if i >= MAX_TOTAL:
#                 break

#         # ------------------------------------------------------------------
#         # 5. RETURN
#         # ------------------------------------------------------------------
#         return jsonify({
#             "success": True,
#             "total_videos": len(final_list),
#             "videos": final_list
#         })

#     except Exception as e:
#         app.logger.error(f"Video outliers error: {str(e)}")
#         return jsonify({"error": str(e)}), 500


@app.route("/api/video_outliers", methods=["POST"])
def video_outliers():
    try:
        data = request.get_json()
        channel_ids = data.get("channel_ids") or data.get("competitors", [])
        if not channel_ids:
            return jsonify({"error": "No competitors provided"}), 400

        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        latest_videos_all = []
        popular_videos_all = []

        # ✅ Fixed per-channel cap
        per_channel_cap = 20  # Hardcoded limit per channel

        for ch_id in channel_ids:
            # Channel info
            ch_resp = youtube.channels().list(
                part="snippet,statistics", id=ch_id
            ).execute()

            if not ch_resp.get("items"):
                continue

            ch_info = ch_resp["items"][0]
            ch_title = ch_info["snippet"]["title"]
            subs_count = int(ch_info["statistics"].get("subscriberCount", 1))

            # Fetch latest medium-length videos
            search_resp = youtube.search().list(
                part="snippet",
                channelId=ch_id,
                type="video",
                order="date",
                videoDuration="medium",
                maxResults=per_channel_cap
            ).execute()

            video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
            latest_videos = fetch_video_details(youtube, video_ids)

            # Fetch popular medium-length videos
            search_resp_pop = youtube.search().list(
                part="snippet",
                channelId=ch_id,
                type="video",
                order="viewCount",
                videoDuration="medium",
                maxResults=per_channel_cap
            ).execute()

            popular_ids = [item["id"]["videoId"] for item in search_resp_pop.get("items", [])]
            popular_videos = fetch_video_details(youtube, popular_ids)

            # Compute average recent views
            avg_recent_views = (
                sum(int(v["statistics"].get("viewCount", 0)) for v in latest_videos) / len(latest_videos)
                if latest_videos else 1
            )

            def clean_video(v, list_type):
                views = int(v["statistics"].get("viewCount", 0))
                duration = parse_duration(v.get("contentDetails", {}).get("duration", "PT0S"))

                return {
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel_id": ch_id,
                    "channel_title": ch_title,
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "avg_recent_views": round(avg_recent_views, 2),
                    "multiplier": round(views / avg_recent_views, 2) if avg_recent_views else 0,
                    "thumbnail_url": v["snippet"]["thumbnails"].get("high", {}).get("url"),
                    "url": f"https://www.youtube.com/watch?v={v['id']}",
                    "subscriber_count": subs_count,
                    "list_type": list_type,
                    "published_at": v["snippet"].get("publishedAt", "")
                }

            # ✅ If a channel has fewer videos than cap, take all available
            latest_videos_all.extend([clean_video(v, "latest") for v in latest_videos])
            popular_videos_all.extend([clean_video(v, "popular") for v in popular_videos])

        # Sorting
        latest_videos_all.sort(key=lambda x: x["views"], reverse=True)
        popular_videos_all.sort(key=lambda x: x["views"])

        # ✅ Dynamic max_total_videos based on channels and cap
        max_total_videos = len(channel_ids) * per_channel_cap

        final_list = []
        channel_counter = {}
        batch_size = 24
        i = 0

        # ✅ Infinite loop protection
        max_iterations = 10000
        iteration_count = 0

        while (latest_videos_all or popular_videos_all) and iteration_count < max_iterations:
            iteration_count += 1

            for lst_type in ["latest", "popular"]:
                if lst_type == "latest" and latest_videos_all:
                    v = latest_videos_all.pop(0)
                elif lst_type == "popular" and popular_videos_all:
                    v = popular_videos_all.pop(0)
                else:
                    continue

                # Per-channel batching rule
                batch_index = i // batch_size
                key = f"{v['channel_id']}_{batch_index}"

                if channel_counter.get(key, 0) >= 2:
                    # Push back instead of breaking the loop
                    if lst_type == "latest":
                        latest_videos_all.append(v)
                    else:
                        popular_videos_all.append(v)
                    continue

                final_list.append(v)
                channel_counter[key] = channel_counter.get(key, 0) + 1
                i += 1

                # ✅ Stop once we hit the target number of videos (but not before)
                if i >= max_total_videos:
                    break

            if i >= max_total_videos:
                break

        # ✅ Safety fallback if iteration cap hit early
        if iteration_count >= max_iterations and len(final_list) < max_total_videos:
            app.logger.warning(
                f"Max iteration limit reached with {len(final_list)} videos — filling remaining slots randomly"
            )

            remaining_videos = latest_videos_all + popular_videos_all
            final_video_ids = {v["video_id"] for v in final_list}
            remaining_videos = [v for v in remaining_videos if v["video_id"] not in final_video_ids]

            import random
            random.shuffle(remaining_videos)
            needed = max_total_videos - len(final_list)
            final_list.extend(remaining_videos[:needed])

        return jsonify({
            "success": True,
            "total_videos": len(final_list),
            "videos": final_list


        })

    except Exception as e:
        app.logger.error(f"Video outliers error: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route("/api/comp_analysis", methods=["POST"])
def comp_analysis():
    try:
        data = request.get_json()
        channel_id = data.get("channel_id", "").strip()
        if not channel_id:
            return jsonify({"error": "Channel ID is required"}), 400

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # 1️⃣ Fetch channel info (with snippet for thumbnail)
        channel_resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()

        if not channel_resp["items"]:
            return jsonify({"error": "Channel not found"}), 404

        channel_info = channel_resp["items"][0]
        channel_title = channel_info["snippet"]["title"]
        channel_thumbnail = channel_info["snippet"]["thumbnails"]["high"]["url"]
        subs_count = int(channel_info["statistics"].get("subscriberCount", 1))
        uploads_playlist = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]

        # 2️⃣ Fetch up to 50 recent uploads
        videos_resp = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist,
            maxResults=50
        ).execute()

        # Helper to fetch filtered videos by duration
        def get_videos_by_duration(videos_resp, min_dur=None, max_dur=None):
            selected = []
            for item in videos_resp.get("items", []):
                vid_id = item["contentDetails"]["videoId"]
                vid_details = youtube.videos().list(
                    part="contentDetails,snippet,statistics",
                    id=vid_id
                ).execute()

                if not vid_details["items"]:
                    continue

                v = vid_details["items"][0]
                duration = parse_duration(v.get("contentDetails", {}).get("duration", ""))
                if (min_dur is None or duration >= min_dur) and (max_dur is None or duration <= max_dur):
                    selected.append({
                        "id": vid_id,
                        "title": v["snippet"]["title"]
                    })
            return selected

        # 3️⃣ Progressive fallback: medium → long → short
        candidate_videos = get_videos_by_duration(videos_resp, min_dur=240)
        selected_duration_type = "medium"

        if len(candidate_videos) < 20:
            long_videos = get_videos_by_duration(videos_resp, min_dur=600)
            if long_videos:
                candidate_videos.extend([v for v in long_videos if v not in candidate_videos])
                selected_duration_type = "long"

        if len(candidate_videos) < 20:
            short_videos = get_videos_by_duration(videos_resp, max_dur=240)
            if short_videos:
                candidate_videos.extend([v for v in short_videos if v not in candidate_videos])
                selected_duration_type = "short"

        # Limit to 20
        candidate_videos = candidate_videos[:20]

        if not candidate_videos:
            return jsonify({"error": "No suitable videos found for this channel"}), 404

        # 4️⃣ Collect competitor channels with frequency count
        competitor_counts = {}
        for video in candidate_videos:
            search_resp = youtube.search().list(
                part="snippet",
                q=video["title"],
                type="video",
                maxResults=50,
                relevanceLanguage="en",
                videoDuration=selected_duration_type  # match the duration type used
            ).execute()

            for item in search_resp.get("items", []):
                ch_id = item["snippet"]["channelId"]
                ch_title = item["snippet"]["channelTitle"]

                if ch_id == channel_id:
                    continue

                if ch_id not in competitor_counts:
                    competitor_counts[ch_id] = {"title": ch_title, "count": 0}
                competitor_counts[ch_id]["count"] += 1

        # sort competitors by frequency (descending) and take top 50
        competitors = sorted(
            competitor_counts.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True
        )[:50]

        # 5️⃣ For each competitor, fetch latest videos with avg views + channel thumbnail
        competitors_data = []
        for comp_id, comp_data in competitors:
            comp_resp = youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=comp_id
            ).execute()

            if not comp_resp.get("items"):
                continue

            comp_info = comp_resp["items"][0]
            comp_title = comp_data["title"]
            comp_subs = int(comp_info["statistics"].get("subscriberCount", 0))

            # Skip competitors with fewer than 1,000 subs
            if comp_subs < 1000:
                continue

            comp_uploads = comp_info["contentDetails"]["relatedPlaylists"]["uploads"]
            comp_thumbnail = comp_info["snippet"]["thumbnails"]["high"]["url"]

            # Fetch up to 5 recent uploads
            comp_uploads_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=comp_uploads,
                maxResults=5
            ).execute()

            comp_video_ids = [v["contentDetails"]["videoId"] for v in comp_uploads_resp.get("items", [])]
            comp_videos = fetch_video_details(youtube, comp_video_ids)

            if not comp_videos:
                continue

            # calculate avg recent views
            avg_recent_views = sum(
                int(v["statistics"].get("viewCount", 0)) for v in comp_videos
            ) / len(comp_videos)

            latest_videos = comp_videos[:4]

            video_data = []
            for v in latest_videos:
                duration = parse_duration(v.get("contentDetails", {}).get("duration", ""))
                views = int(v["statistics"].get("viewCount", 0))
                multiplier = round(views / avg_recent_views, 2) if avg_recent_views > 0 else 0

                video_data.append({
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel_id": comp_id,
                    "channel_title": comp_title,
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "multiplier": multiplier,
                    "avg_recent_views": round(avg_recent_views, 2),
                    "thumbnail_url": v["snippet"]["thumbnails"]["high"]["url"],
                    "url": f"https://www.youtube.com/watch?v={v['id']}",
                    "published_at": v["snippet"]["publishedAt"]
                })

            competitors_data.append({
                "id": comp_id,
                "title": comp_title,
                "subscriber_count": comp_subs,
                "avg_recent_views": round(avg_recent_views, 2),
                "frequency": comp_data.get("count", 0),
                "thumbnail_url": comp_thumbnail,
                "videos": video_data
            })

            if len(competitors_data) * 4 >= 200:
                break

        return jsonify({
            "success": True,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "channel_thumbnail": channel_thumbnail,
            "video_duration_mode": selected_duration_type,  # ✅ Added for clarity
            "competitors": competitors_data
        })

    except Exception as e:
        app.logger.error(f"Channel comp_analysis error: {str(e)}")
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
            maxResults=50,
            type='video',
            videoDuration="medium"
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
        videos = videos[:50]  # Ensure max 15 videos

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
@app.route('/api/short_ideas_by_channel_id', methods=['POST'])
def short_ideas_by_channel_id():
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
            maxResults=50,
            type='video',
            videoDuration="short"
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
        videos = videos[:50]  # Ensure max 15 videos

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
    
# app.
@app.route('/api/script_generation', methods=['POST'])
def script_generation():
    """Enhanced all-in-one endpoint with proper video file handling"""
    user_id = request.remote_addr
    print(f"\n{'='*60}")
    print(f"Starting complete script generation for user: {user_id}")
    print(f"{'='*60}\n")
    
    try:
        # Handle both JSON and multipart form data
        if request.content_type and 'multipart/form-data' in request.content_type:
            data = {
                'personal_videos': request.form.getlist('personal_videos[]'),
                'inspiration_videos': request.form.getlist('inspiration_videos[]'),
                'prompt': request.form.get('prompt', '').strip(),
                'minutes': request.form.get('minutes', type=int)
            }
            documents = request.files.getlist('documents[]')
            video_files = request.files.getlist('video_files[]')
        else:
            data = request.json or {}
            documents = []
            video_files = []
        
        personal_videos = data.get('personal_videos', [])
        inspiration_videos = data.get('inspiration_videos', [])
        prompt = data.get('prompt', '').strip()
        target_minutes = data.get('minutes')
        
        if not prompt:
            prompt = "Create an engaging, informative YouTube video script on a topic of general interest."
        
        print(f"Inputs received:")
        print(f"  - Personal videos: {len(personal_videos)}")
        print(f"  - Inspiration videos: {len(inspiration_videos)}")
        print(f"  - Documents: {len(documents)}")
        print(f"  - Video files: {len(video_files)}")
        print(f"  - Prompt: {prompt[:100]}...")
        print(f"  - Target minutes: {target_minutes}")
        print(f"  - Upload folder: {UPLOAD_FOLDER}")
        print(f"  - Upload folder exists: {os.path.exists(UPLOAD_FOLDER)}\n")
        
        # Initialize results
        processed_documents = []
        processed_personal = []
        processed_inspiration = []
        errors = []
        
        # ========================================
        # PROCESS DOCUMENTS
        # ========================================
        if documents:
            print(f"\n{'='*60}")
            print(f"PROCESSING {len(documents)} DOCUMENTS")
            print(f"{'='*60}\n")
            
            for idx, file in enumerate(documents, 1):
                if file.filename and document_processor.allowed_file(file.filename):
                    try:
                        filename = secure_filename(file.filename)
                        # Use safe filename for temp storage
                        safe_user_id = user_id.replace('.', '_').replace(':', '_')
                        file_path = os.path.join(
                            UPLOAD_FOLDER, 
                            f"temp_doc_{safe_user_id}_{int(time.time())}_{filename}"
                        )
                        
                        print(f"[{idx}/{len(documents)}] Processing document: {filename}")
                        print(f"  Saving to: {file_path}")
                        
                        file.save(file_path)
                        
                        if not os.path.exists(file_path):
                            raise FileNotFoundError(f"Failed to save file to {file_path}")
                        
                        print(f"  File saved successfully: {os.path.getsize(file_path)} bytes")
                        
                        result = document_processor.process_document(file_path, filename)
                        
                        # Cleanup
                        try:
                            os.remove(file_path)
                            print(f"  Cleaned up temp file")
                        except Exception as cleanup_error:
                            print(f"  Warning: Failed to cleanup {file_path}: {cleanup_error}")
                        
                        if result['error']:
                            error_msg = f"Document {filename}: {result['error']}"
                            errors.append(error_msg)
                            print(f"  ❌ ERROR: {result['error']}")
                        else:
                            processed_documents.append({
                                'filename': filename,
                                'text': result['text'],
                                'stats': result['stats']
                            })
                            print(f"  ✓ Success: {len(result['text'])} characters extracted")
                            
                    except Exception as e:
                        error_msg = f"Document {file.filename}: {str(e)}"
                        errors.append(error_msg)
                        print(f"  ❌ EXCEPTION: {str(e)}")
        
        # ========================================
        # PROCESS UPLOADED VIDEO FILES
        # ========================================
        if video_files:
            print(f"\n{'='*60}")
            print(f"PROCESSING {len(video_files)} UPLOADED VIDEO FILES")
            print(f"{'='*60}\n")
            
            for idx, video_file in enumerate(video_files, 1):
                if video_file.filename and video_processor.is_supported_video_format(video_file.filename):
                    video_path = None
                    try:
                        filename = secure_filename(video_file.filename)
                        # Create safe user ID and timestamp for unique filename
                        safe_user_id = user_id.replace('.', '_').replace(':', '_')
                        timestamp = int(time.time())
                        video_path = os.path.join(
                            UPLOAD_FOLDER, 
                            f"temp_video_{safe_user_id}_{timestamp}_{filename}"
                        )
                        
                        print(f"[{idx}/{len(video_files)}] Processing video: {filename}")
                        print(f"  UPLOAD_FOLDER: {UPLOAD_FOLDER}")
                        print(f"  Full video path: {video_path}")
                        print(f"  Directory exists: {os.path.exists(UPLOAD_FOLDER)}")
                        
                        print(f"  Saving video file...")
                        video_file.save(video_path)
                        
                        # Verify file was saved
                        if not os.path.exists(video_path):
                            raise FileNotFoundError(f"Failed to save video file to {video_path}")
                        
                        file_size = os.path.getsize(video_path)
                        print(f"  ✓ File saved successfully: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
                        
                        # Process the video
                        print(f"  Starting video content processing...")
                        result = video_processor.process_video_content(video_path, 'local')
                        
                        # Cleanup
                        try:
                            if video_path and os.path.exists(video_path):
                                os.remove(video_path)
                                print(f"  Cleaned up temp video file")
                        except Exception as cleanup_error:
                            print(f"  Warning: Failed to cleanup {video_path}: {cleanup_error}")
                        
                        if result['error']:
                            error_msg = f"Video {filename}: {result['error']}"
                            errors.append(error_msg)
                            print(f"  ❌ ERROR: {result['error']}")
                        else:
                            processed_inspiration.append({
                                'source': filename,
                                'transcript': result['transcript'],
                                'stats': result['stats'],
                                'type': 'local_video'
                            })
                            print(f"  ✓ Success: {len(result['transcript'])} characters transcribed")
                            print(f"\n  TRANSCRIPT PREVIEW (first 500 chars):")
                            print(f"  {'-'*56}")
                            print(f"  {result['transcript'][:500]}...")
                            print(f"  {'-'*56}\n")
                            
                    except Exception as e:
                        error_msg = f"Video {video_file.filename}: {str(e)}"
                        errors.append(error_msg)
                        print(f"  ❌ EXCEPTION: {str(e)}")
                        import traceback
                        print(f"  TRACEBACK:\n{traceback.format_exc()}")
                        
                        # Cleanup on error
                        try:
                            if video_path and os.path.exists(video_path):
                                os.remove(video_path)
                                print(f"  Cleaned up temp video file after error")
                        except:
                            pass
                else:
                    error_msg = f"Video {video_file.filename}: Unsupported format or invalid filename"
                    errors.append(error_msg)
                    print(f"  ❌ Skipped: Unsupported format")
        
        # ========================================
        # PROCESS PERSONAL YOUTUBE VIDEOS
        # ========================================
        if personal_videos:
            print(f"\n{'='*60}")
            print(f"PROCESSING {len(personal_videos)} PERSONAL YOUTUBE VIDEOS")
            print(f"{'='*60}\n")
            
            for idx, url in enumerate(personal_videos, 1):
                if url and video_processor.validate_youtube_url(url):
                    print(f"[{idx}/{len(personal_videos)}] Processing: {url}")
                    result = video_processor.process_video_content(url, 'youtube')
                    
                    if result['error']:
                        error_msg = f"Personal video {url}: {result['error']}"
                        errors.append(error_msg)
                        print(f"  ❌ ERROR: {result['error']}")
                    else:
                        processed_personal.append({
                            'url': url,
                            'transcript': result['transcript'],
                            'stats': result['stats'],
                            'type': 'youtube'
                        })
                        print(f"  ✓ Success: {len(result['transcript'])} characters")
        
        # ========================================
        # PROCESS INSPIRATION YOUTUBE VIDEOS
        # ========================================
        if inspiration_videos:
            print(f"\n{'='*60}")
            print(f"PROCESSING {len(inspiration_videos)} INSPIRATION YOUTUBE VIDEOS")
            print(f"{'='*60}\n")
            
            for idx, url in enumerate(inspiration_videos, 1):
                if url and video_processor.validate_youtube_url(url):
                    print(f"[{idx}/{len(inspiration_videos)}] Processing: {url}")
                    result = video_processor.process_video_content(url, 'youtube')
                    
                    if result['error']:
                        error_msg = f"Inspiration video {url}: {result['error']}"
                        errors.append(error_msg)
                        print(f"  ❌ ERROR: {result['error']}")
                    else:
                        processed_inspiration.append({
                            'url': url,
                            'transcript': result['transcript'],
                            'stats': result['stats'],
                            'type': 'youtube'
                        })
                        print(f"  ✓ Success: {len(result['transcript'])} characters")
        
        # ========================================
        # ANALYZE CONTENT
        # ========================================
        print(f"\n{'='*60}")
        print(f"ANALYZING CONTENT")
        print(f"{'='*60}\n")
        
        # Style analysis
        style_profile = "Professional, engaging YouTube style with clear explanations, good pacing, and viewer-friendly language."
        if processed_personal:
            print(f"Analyzing creator style from {len(processed_personal)} personal videos...")
            personal_transcripts = [v['transcript'] for v in processed_personal]
            try:
                style_profile = script_generator.analyze_creator_style(personal_transcripts)
                print(f"✓ Style analysis complete: {len(style_profile)} characters")
            except Exception as e:
                error_msg = f"Style analysis failed: {str(e)}"
                errors.append(error_msg)
                print(f"❌ ERROR: {error_msg}")
        else:
            print("Using default style profile (no personal videos provided)")
        
        # Inspiration analysis
        inspiration_summary = "Creating original, engaging content based on best YouTube practices and viewer engagement strategies."
        if processed_inspiration:
            print(f"\nAnalyzing inspiration content from {len(processed_inspiration)} sources...")
            inspiration_transcripts = [v['transcript'] for v in processed_inspiration]
            try:
                inspiration_summary = script_generator.analyze_inspiration_content(inspiration_transcripts)
                print(f"✓ Inspiration analysis complete: {len(inspiration_summary)} characters")
            except Exception as e:
                error_msg = f"Inspiration analysis failed: {str(e)}"
                errors.append(error_msg)
                print(f"❌ ERROR: {error_msg}")
        else:
            print("Using default inspiration summary (no inspiration sources provided)")
        
        # Document analysis
        document_insights = "Using general knowledge and industry best practices to create informative content."
        if processed_documents:
            print(f"\nAnalyzing {len(processed_documents)} documents...")
            document_texts = [d['text'] for d in processed_documents]
            try:
                document_insights = script_generator.analyze_documents(document_texts)
                print(f"✓ Document analysis complete: {len(document_insights)} characters")
            except Exception as e:
                error_msg = f"Document analysis failed: {str(e)}"
                errors.append(error_msg)
                print(f"❌ ERROR: {error_msg}")
        else:
            print("Using default document insights (no documents provided)")
        
        # Add duration context
        if target_minutes:
            prompt = f"{prompt}\n\n[Target video duration: approximately {target_minutes} minutes]"
            print(f"\n✓ Added duration target to prompt: {target_minutes} minutes")
        
        # ========================================
        # GENERATE SCRIPT
        # ========================================
        print(f"\n{'='*60}")
        print(f"GENERATING FINAL SCRIPT")
        print(f"{'='*60}\n")
        
        try:
            script = script_generator.generate_enhanced_script(
                style_profile,
                inspiration_summary,
                document_insights,
                prompt
            )
            print(f"✓ Script generated successfully: {len(script)} characters\n")
        except Exception as e:
            print(f"❌ Script generation failed: {str(e)}")
            import traceback
            print(f"TRACEBACK:\n{traceback.format_exc()}")
            return jsonify({'error': f'Script generation failed: {str(e)}'}), 500
        
        # ========================================
        # STORE SESSION DATA
        # ========================================
        chat_session_id = str(uuid.uuid4())
        user_data[user_id]['current_script'] = {
            'content': script,
            'style_profile': style_profile,
            'topic_insights': inspiration_summary,
            'document_insights': document_insights,
            'original_prompt': prompt,
            'target_minutes': target_minutes,
            'timestamp': datetime.now().isoformat()
        }
        
        user_data[user_id]['chat_sessions'][chat_session_id] = {
            'messages': [],
            'script_versions': [script],
            'created_at': datetime.now().isoformat()
        }
        
        print(f"✓ Session stored: {chat_session_id}")
        
        # ========================================
        # CALCULATE STATS
        # ========================================
        stats = {
            'personal_videos': len(processed_personal),
            'inspiration_videos': len(processed_inspiration),
            'documents': len(processed_documents),
            'total_sources': len(processed_personal) + len(processed_inspiration) + len(processed_documents),
            'errors_count': len(errors),
            'video_files_processed': len([v for v in processed_inspiration if v.get('type') == 'local_video']),
            'target_duration': target_minutes
        }
        
        print(f"\n{'='*60}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Stats: {json.dumps(stats, indent=2)}")
        if errors:
            print(f"\nErrors encountered ({len(errors)}):")
            for error in errors:
                print(f"  - {error}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'script': script,
            'chat_session_id': chat_session_id,
            'stats': stats,
            'processed_content': {
                'personal_videos': len(processed_personal),
                'inspiration_videos': len(processed_inspiration),
                'documents': len(processed_documents),
                'video_files': len([v for v in processed_inspiration if v.get('type') == 'local_video'])
            },
            'errors': errors if errors else None,
            'analysis_quality': 'premium' if (processed_personal and processed_inspiration and processed_documents) else 'optimal' if any([processed_personal, processed_inspiration, processed_documents]) else 'basic',
            'inputs_provided': {
                'prompt': bool(data.get('prompt')),
                'personal_videos': len(personal_videos),
                'inspiration_videos': len(inspiration_videos),
                'documents': len(documents),
                'video_files': len(video_files),
                'duration_specified': target_minutes is not None
            }
        })
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"CRITICAL ERROR IN SCRIPT GENERATION")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        print(f"{'='*60}\n")
        return jsonify({'error': f'Complete script generation failed: {str(e)}'}), 500

@app.route('/api/chat-modify-script', methods=['POST'])
def chat_modify_script():
    """Enhanced chat modification with document context"""
    user_id = request.remote_addr
    data = request.json
    user_message = data.get('message', '').strip()
    chat_session_id = data.get('chat_session_id')
    
    if not user_message:
        return jsonify({'error': 'Please provide a modification request'}), 400
    
    current_script_data = user_data[user_id].get('current_script')
    if not current_script_data:
        return jsonify({'error': 'No active script to modify'}), 400
    
    try:
        current_script = current_script_data['content']
        style_profile = current_script_data['style_profile']
        topic_insights = current_script_data['topic_insights']
        document_insights = current_script_data.get('document_insights', '')
        
        modification_response = script_generator.modify_script_chat(
            current_script,
            style_profile,
            topic_insights,
            document_insights,
            user_message
        )
        
        if chat_session_id and chat_session_id in user_data[user_id]['chat_sessions']:
            chat_session = user_data[user_id]['chat_sessions'][chat_session_id]
            chat_session['messages'].append({
                'user_message': user_message,
                'ai_response': modification_response,
                'timestamp': datetime.now().isoformat()
            })
        
        return jsonify({
            'success': True,
            'response': modification_response,
            'user_message': user_message,
            'chat_session_id': chat_session_id,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': f'Error modifying script: {str(e)}'}), 500


@app.route('/api/update-script', methods=['POST'])
def update_script():
    """Update the current working script"""
    user_id = request.remote_addr
    data = request.json
    new_script = data.get('script', '').strip()
    chat_session_id = data.get('chat_session_id')
    
    if not new_script:
        return jsonify({'error': 'Script content is required'}), 400
    
    try:
        if user_data[user_id].get('current_script'):
            user_data[user_id]['current_script']['content'] = new_script
            user_data[user_id]['current_script']['timestamp'] = datetime.now().isoformat()
        
        if chat_session_id and chat_session_id in user_data[user_id]['chat_sessions']:
            chat_session = user_data[user_id]['chat_sessions'][chat_session_id]
            chat_session['script_versions'].append(new_script)
        
        return jsonify({
            'success': True,
            'message': 'Script updated successfully'
        })
    except Exception as e:
        return jsonify({'error': f'Error updating script: {str(e)}'}), 500



# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'timestamp': datetime.now().isoformat(),
#         'version': '2.0-simplified',
#         'upload_folder': UPLOAD_FOLDER,
#         'upload_folder_exists': os.path.exists(UPLOAD_FOLDER)
#     })


# @app.route('/api/script_generation', methods=['POST'])
# def generate_complete_script():
#     """Enhanced all-in-one endpoint: ALL inputs optional - upload documents, process videos (YouTube + local), analyze, and generate script"""
#     user_id = request.remote_addr
    
#     try:
#         # Handle both JSON and multipart form data
#         if request.content_type and 'multipart/form-data' in request.content_type:
#             data = {
#                 'personal_videos': request.form.getlist('personal_videos[]'),
#                 'inspiration_videos': request.form.getlist('inspiration_videos[]'),
#                 'prompt': request.form.get('prompt', '').strip(),
#                 'minutes': request.form.get('minutes', type=int)  # NEW: video duration target
#             }
#             documents = request.files.getlist('documents[]')
#             video_files = request.files.getlist('video_files[]')
#         else:
#             data = request.json or {}
#             documents = []
#             video_files = []
        
#         personal_videos = data.get('personal_videos', [])
#         inspiration_videos = data.get('inspiration_videos', [])
#         prompt = data.get('prompt', '').strip()
#         target_minutes = data.get('minutes')  # NEW: optional duration target
        
#         # CHANGED: Prompt now optional - will generate generic script if missing
#         if not prompt:
#             prompt = "Create an engaging, informative YouTube video script on a topic of general interest."
        
#         # Initialize results
#         processed_documents = []
#         processed_personal = []
#         processed_inspiration = []
#         errors = []
        
#         # Process documents if uploaded (OPTIONAL)
#         if documents:
#             logger.info(f"Processing {len(documents)} documents...")
#             for file in documents:
#                 if file.filename and document_processor.allowed_file(file.filename):
#                     try:
#                         filename = secure_filename(file.filename)
#                         file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{user_id.replace('.', '_')}_{filename}")
#                         file.save(file_path)
                        
#                         result = document_processor.process_document(file_path, filename)
                        
#                         try:
#                             os.remove(file_path)
#                         except:
#                             pass
                        
#                         if result['error']:
#                             errors.append(f"Document {filename}: {result['error']}")
#                         else:
#                             processed_documents.append({
#                                 'filename': filename,
#                                 'text': result['text'],
#                                 'stats': result['stats']
#                             })
#                     except Exception as e:
#                         errors.append(f"Document {file.filename}: {str(e)}")
        
#         # Process uploaded video files (OPTIONAL)
#         if video_files:
#             logger.info(f"Processing {len(video_files)} uploaded video files...")
#             for video_file in video_files:
#                 if video_file.filename and video_processor.is_supported_video_format(video_file.filename):
#                     try:
#                         filename = secure_filename(video_file.filename)
#                         video_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_video_{user_id.replace('.', '_')}_{filename}")
#                         video_file.save(video_path)
                        
#                         result = video_processor.process_video_content(video_path, 'local')
                        
#                         try:
#                             os.remove(video_path)
#                         except:
#                             pass
                        
#                         if result['error']:
#                             errors.append(f"Video {filename}: {result['error']}")
#                         else:
#                             processed_inspiration.append({
#                                 'source': filename,
#                                 'transcript': result['transcript'],
#                                 'stats': result['stats'],
#                                 'type': 'local_video'
#                             })
#                     except Exception as e:
#                         errors.append(f"Video {video_file.filename}: {str(e)}")
        
#         # Process personal YouTube videos (OPTIONAL)
#         if personal_videos:
#             logger.info(f"Processing {len(personal_videos)} personal videos...")
#             for url in personal_videos:
#                 if url and video_processor.validate_youtube_url(url):
#                     result = video_processor.process_video_content(url, 'youtube')
                    
#                     if result['error']:
#                         errors.append(f"Personal video {url}: {result['error']}")
#                     else:
#                         # >>> ADD THESE LINES STARTING HERE <<<
#                         print("\n\n" + "="*40)
#                         print(f"EXTRACTED TRANSCRIPT FOR PERSONAL VIDEO: {url}")
#                         print("="*40)
#                         print(result['transcript'])
#                         print("="*40 + "\n\n")
#                         # >>> END OF ADDED LINES <<<

#                         processed_personal.append({
#                             'url': url,
#                             'transcript': result['transcript'],
#                             'stats': result['stats'],
#                             'type': 'youtube'
#                         })

        
#         # Process inspiration YouTube videos (OPTIONAL)
#         if inspiration_videos:
#             logger.info(f"Processing {len(inspiration_videos)} inspiration videos...")
#             for url in inspiration_videos:
#                 if url and video_processor.validate_youtube_url(url):
#                     result = video_processor.process_video_content(url, 'youtube')
                    
#                     if result['error']:
#                         errors.append(f"Inspiration video {url}: {result['error']}")
#                     else:
#                         processed_inspiration.append({
#                             'url': url,
#                             'transcript': result['transcript'],
#                             'stats': result['stats'],
#                             'type': 'youtube'
#                         })
        
#         # CHANGED: No error if no content - will use defaults
#         logger.info("Analyzing all content...")
        
#         # Style analysis (use default if no personal videos)
#         style_profile = "Professional, engaging YouTube style with clear explanations, good pacing, and viewer-friendly language."
#         if processed_personal:
#             personal_transcripts = [v['transcript'] for v in processed_personal]
#             try:
#                 style_profile = script_generator.analyze_creator_style(personal_transcripts)
#             except Exception as e:
#                 logger.error(f"Style analysis error: {str(e)}")
#                 errors.append(f"Style analysis failed: {str(e)}")
        
#         # Inspiration analysis (use default if no inspiration)
#         inspiration_summary = "Creating original, engaging content based on best YouTube practices and viewer engagement strategies."
#         if processed_inspiration:
#             inspiration_transcripts = [v['transcript'] for v in processed_inspiration]
#             try:
#                 inspiration_summary = script_generator.analyze_inspiration_content(inspiration_transcripts)
#             except Exception as e:
#                 logger.error(f"Inspiration analysis error: {str(e)}")
#                 errors.append(f"Inspiration analysis failed: {str(e)}")
        
#         # Document analysis (use default if no documents)
#         document_insights = "Using general knowledge and industry best practices to create informative content."
#         if processed_documents:
#             document_texts = [d['text'] for d in processed_documents]
#             try:
#                 document_insights = script_generator.analyze_documents(document_texts)
#             except Exception as e:
#                 logger.error(f"Document analysis error: {str(e)}")
#                 errors.append(f"Document analysis failed: {str(e)}")
        
#         # NEW: Add duration context to prompt if specified
#         if target_minutes:
#             prompt = f"{prompt}\n\n[Target video duration: approximately {target_minutes} minutes]"
        
#         # Generate script
#         logger.info("Generating final script...")
#         try:
#             script = script_generator.generate_enhanced_script(
#                 style_profile,
#                 inspiration_summary,
#                 document_insights,
#                 prompt
#             )
#         except Exception as e:
#             logger.error(f"Script generation error: {str(e)}")
#             return jsonify({'error': f'Script generation failed: {str(e)}'}), 500
        
#         # Store current script for chat modifications
#         chat_session_id = str(uuid.uuid4())
#         user_data[user_id]['current_script'] = {
#             'content': script,
#             'style_profile': style_profile,
#             'topic_insights': inspiration_summary,
#             'document_insights': document_insights,
#             'original_prompt': prompt,
#             'target_minutes': target_minutes,  # NEW: store duration
#             'timestamp': datetime.now().isoformat()
#         }
        
#         # Initialize chat session
#         user_data[user_id]['chat_sessions'][chat_session_id] = {
#             'messages': [],
#             'script_versions': [script],
#             'created_at': datetime.now().isoformat()
#         }
        
#         # Calculate stats
#         stats = {
#             'personal_videos': len(processed_personal),
#             'inspiration_videos': len(processed_inspiration),
#             'documents': len(processed_documents),
#             'total_sources': len(processed_personal) + len(processed_inspiration) + len(processed_documents),
#             'errors_count': len(errors),
#             'video_files_processed': len([v for v in processed_inspiration if v.get('type') == 'local_video']),
#             'target_duration': target_minutes  # NEW: include in stats
#         }
        
#         logger.info("Complete script generation finished successfully")
        
#         return jsonify({
#             'success': True,
#             'script': script,
#             'chat_session_id': chat_session_id,
#             'stats': stats,
#             'processed_content': {
#                 'personal_videos': len(processed_personal),
#                 'inspiration_videos': len(processed_inspiration),
#                 'documents': len(processed_documents),
#                 'video_files': len([v for v in processed_inspiration if v.get('type') == 'local_video'])
#             },
#             'errors': errors if errors else None,
#             'analysis_quality': 'premium' if (processed_personal and processed_inspiration and processed_documents) else 'optimal' if any([processed_personal, processed_inspiration, processed_documents]) else 'basic',
#             'inputs_provided': {  # NEW: show what was provided
#                 'prompt': bool(data.get('prompt')),
#                 'personal_videos': len(personal_videos),
#                 'inspiration_videos': len(inspiration_videos),
#                 'documents': len(documents),
#                 'video_files': len(video_files),
#                 'duration_specified': target_minutes is not None
#             }
#         })
        
#     except Exception as e:
#         logger.error(f"Complete script generation error: {str(e)}")
#         return jsonify({'error': f'Complete script generation failed: {str(e)}'}), 500

# @app.route('/api/chat-modify-script', methods=['POST'])
# def chat_modify_script():
#     """Enhanced chat modification with document context"""
#     user_id = request.remote_addr  # Use IP as user_id
    
#     data = request.json
#     user_message = data.get('message', '').strip()
#     chat_session_id = data.get('chat_session_id')
    
#     if not user_message:
#         return jsonify({'error': 'Please provide a modification request'}), 400
    
#     current_script_data = user_data[user_id].get('current_script')
#     if not current_script_data:
#         return jsonify({'error': 'No active script to modify'}), 400
    
#     try:
#         logger.info(f"Processing chat modification: {user_message[:50]}...")
        
#         # Get current script and full context
#         current_script = current_script_data['content']
#         style_profile = current_script_data['style_profile']
#         topic_insights = current_script_data['topic_insights']
#         document_insights = current_script_data.get('document_insights', '')
        
#         # Generate modification with full context
#         modification_response = script_generator.modify_script_chat(
#             current_script,
#             style_profile,
#             topic_insights,
#             document_insights,
#             user_message
#         )
        
#         # Update chat session
#         if chat_session_id and chat_session_id in user_data[user_id]['chat_sessions']:
#             chat_session = user_data[user_id]['chat_sessions'][chat_session_id]
#             chat_session['messages'].append({
#                 'user_message': user_message,
#                 'ai_response': modification_response,
#                 'timestamp': datetime.now().isoformat()
#             })
        
#         logger.info("Script modification completed successfully")
        
#         return jsonify({
#             'success': True,
#             'response': modification_response,
#             'user_message': user_message,
#             'timestamp': datetime.now().isoformat()
#         })
        
#     except Exception as e:
#         logger.error(f"Error modifying script via chat: {str(e)}")
#         return jsonify({'error': f'Error modifying script: {str(e)}'}), 500

# @app.route('/api/update-script', methods=['POST'])
# def update_script():
#     """Update the current working script"""
#     user_id = request.remote_addr  # Use IP as user_id
    
#     data = request.json
#     new_script = data.get('script', '').strip()
#     chat_session_id = data.get('chat_session_id')
    
#     if not new_script:
#         return jsonify({'error': 'Script content is required'}), 400
    
#     try:
#         # Update current script
#         if user_data[user_id].get('current_script'):
#             user_data[user_id]['current_script']['content'] = new_script
#             user_data[user_id]['current_script']['timestamp'] = datetime.now().isoformat()
        
#         # Add to chat session history
#         if chat_session_id and chat_session_id in user_data[user_id]['chat_sessions']:
#             chat_session = user_data[user_id]['chat_sessions'][chat_session_id]
#             chat_session['script_versions'].append(new_script)
        
#         return jsonify({
#             'success': True,
#             'message': 'Script updated successfully'
#         })
        
#     except Exception as e:
#         logger.error(f"Error updating script: {str(e)}")
#         return jsonify({'error': f'Error updating script: {str(e)}'}), 500

# @app.route('/health', methods=['GET'])
# def health_check():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy',
#         'timestamp': datetime.now().isoformat(),
#         'version': '2.0-simplified-ytdlp'
#     })

@app.route('/api/similar_videos', methods=['POST'])
def similar_videos():
    """Fetch similar YouTube videos based on input video ID, filtered by duration category."""
    try:
        data = request.get_json()
        video_id = data.get('video_id', '').strip()
        if not video_id:
            return jsonify({'error': 'Video ID is required'}), 400

        # --------------------------
        # 1. Fetch input video details (title and duration) using YouTube Data API
        # --------------------------
        youtube_url = 'https://youtube.googleapis.com/youtube/v3/videos'
        params = {
            'part': 'snippet,contentDetails',
            'id': video_id,
            'key': YOUTUBE_API_KEY
        }
        video_resp = requests.get(youtube_url, params=params)
        video_resp.raise_for_status()
        video_data = video_resp.json()

        if not video_data.get('items'):
            return jsonify({'error': 'Video not found'}), 404

        video_item = video_data['items'][0]
        original_title = video_item['snippet']['title']
        duration_iso = video_item['contentDetails']['duration']  # e.g., PT5M30S

        # Parse ISO 8601 duration to seconds
        def parse_duration(iso_duration):
            duration = isodate.parse_duration(iso_duration)
            return int(duration.total_seconds())

        original_duration = parse_duration(duration_iso)

        # Determine duration category
        if original_duration < 240:  # Less than 4 minutes
            duration_category = 'short'
        elif original_duration <= 1200:  # 4–20 minutes
            duration_category = 'medium'
        else:  # Over 20 minutes
            duration_category = 'long'

        # --------------------------
        # 2. Simplify title using Gemini-2.0-Flash
        # --------------------------
        client = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Take this YouTube video title and make a very short, simple keyword phrase (3-6 words) suitable for searching similar videos. Focus on the core theme and avoid special characters except spaces, commas, periods, question marks, and hyphens:\n\nTitle: {original_title}\n\nShort search phrase:"
        
        gemini_response = client.generate_content(prompt)
        search_phrase = gemini_response.text.strip()
        if not search_phrase:
            return jsonify({'error': 'Failed to generate search phrase'}), 500

        # --------------------------
        # 3. Search similar videos using YouTube Data API with duration filter
        # --------------------------
        search_url = 'https://youtube.googleapis.com/youtube/v3/search'
        search_params = {
            'part': 'snippet',
            'q': search_phrase,
            'type': 'video',
            'videoDuration': duration_category,  # short, medium, or long
            'maxResults': 20,
            'relevanceLanguage': 'en',
            'key': YOUTUBE_API_KEY
        }
        search_resp = requests.get(search_url, params=search_params)
        search_resp.raise_for_status()
        search_data = search_resp.json()

        videos = []
        for item in search_data.get('items', []):
            video_id = item['id'].get('videoId')
            snippet = item.get('snippet', {})
            if video_id:
                videos.append({
                    'video_id': video_id,
                    'title': snippet.get('title'),
                    'channel_title': snippet.get('channelTitle'),
                    'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url'),
                    'url': f'https://www.youtube.com/watch?v={video_id}'
                })

        # Log if no videos found
        if not videos:
            app.logger.warning(f"No videos found for search phrase: {search_phrase}, duration: {duration_category}")

        # --------------------------
        # 4. Return JSON
        # --------------------------
        return jsonify({
            'success': True,
            'original_title': original_title,
            'original_duration': original_duration,
            'duration_category': duration_category,
            'search_phrase': search_phrase,
            'videos': videos
        })

    except requests.exceptions.RequestException as e:
        app.logger.error(f"YouTube API error: {str(e)}")
        return jsonify({'error': f'YouTube API error: {str(e)}'}), 503
    except Exception as e:
        app.logger.error(f"Similar videos error: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
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


@app.route("/api/youtube_search", methods=["POST"])
def youtube_search():
    try:
        data = request.get_json()
        query = data.get("q", "").strip()
        max_results = 50  # fixed at 50 results

        if not query:
            return jsonify({"error": "Search query (q) is required"}), 400

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # 1. Perform search
        search_resp = youtube.search().list(
            part="snippet",
            q=query,
            type="video",
            maxResults=max_results
        ).execute()

        shorts = []
        videos = []

        for item in search_resp.get("items", []):
            video_id = item["id"]["videoId"]
            channel_id = item["snippet"]["channelId"]

            # Fetch video details
            video_resp = youtube.videos().list(
                part="snippet,contentDetails,statistics",
                id=video_id
            ).execute()

            if not video_resp.get("items"):
                continue

            v = video_resp["items"][0]
            stats = v.get("statistics", {})
            content = v.get("contentDetails", {})
            snippet = v.get("snippet", {})

            # Duration parsing
            duration_seconds = parse_duration(content.get("duration", ""))

            # Channel details
            channel_resp = youtube.channels().list(
                part="statistics,snippet,contentDetails",
                id=channel_id
            ).execute()

            if not channel_resp.get("items"):
                continue

            ch = channel_resp["items"][0]
            channel_data = {
                "subscriber_count": int(ch["statistics"].get("subscriberCount", 0)),
                "title": ch["snippet"].get("title", "")
            }

            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))

            # Compute average channel views
            uploads_playlist = ch["contentDetails"]["relatedPlaylists"]["uploads"]
            uploads_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=uploads_playlist,
                maxResults=5
            ).execute()

            upload_video_ids = [u["contentDetails"]["videoId"] for u in uploads_resp.get("items", [])]
            upload_details = fetch_video_details(youtube, upload_video_ids)

            channel_avg = 0
            if upload_details:
                channel_avg = sum(int(u["statistics"].get("viewCount", 0)) for u in upload_details) / len(upload_details)

            multiplier = views / channel_avg if channel_avg > 0 else 0

            video_result = {
                'video_id': video_id,
                'title': snippet.get('title', ''),
                'channel_id': channel_id,
                'channel_title': channel_data["title"],
                'views': views,
                'channel_avg_views': round(channel_avg, 2),
                'multiplier': round(multiplier, 2),
                'likes': likes,
                'comments': comments,
                'duration': content.get("duration", ""),
                'duration_seconds': duration_seconds,
                'url': f"https://www.youtube.com/watch?v={video_id}",
                'published_at': snippet.get("publishedAt", ""),
                'thumbnail_url': snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
                'viral_score': round(multiplier / 10, 2),
                'engagement_rate': round((likes + comments) / views if views > 0 else 0, 4),
                'subscriber_count': channel_data.get('subscriber_count', 0),
                'language': snippet.get("defaultAudioLanguage", "en")
            }

            # Separate Shorts vs Videos
            if duration_seconds <= 60:
                shorts.append(video_result)
            else:
                videos.append(video_result)

        return jsonify({
            "success": True,
            "shorts": shorts,
            "videos": videos
        })

    except Exception as e:
        app.logger.error(f"YouTube search error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    


BYTEPLUS_API_KEY = os.getenv("BYTEPLUS_API_KEY")

THUMBNAIL_CONTEXT = """
Thumbnails are video “movie posters” — decide CTR in 0.5 seconds. They grab attention, communicate topic, trigger emotion/curiosity, and increase clicks.

Core Functions:
- Stop scroll, tell story, trigger emotion.
- CTR benchmarks: 2–4% weak, 4–6% avg, 6–8% strong, 8%+ viral.

Thumbnail Types:
- Face+Emotion (surprise>happy>calm, avoid anger/fear/sad), Face+Text, Object/Product, Before/After, Tutorial, Podcast/Interview, List/Comparison, Minimal curiosity, Story, Challenge, Reaction, Cinematic/Vlog.

3-Part Structure:
1. Hero (face/object, largest, expressive)
2. Supporting Scene (context, product, background, icons)
3. Text (1–4 bold words, power/curiosity/urgency words)

Design Rules:
- Visual hierarchy: Hero → Text → Background
- Zoom hero, blur/darken background
- High contrast colors: red=urgency, yellow=happy, blue=trust, green=growth, orange=energy, purple=luxury, black=power, white=clean
- Use complementary contrast
- Fonts: bold sans-serif, add drop shadow/stroke
- Text top-left/top-center, avoid bottom-right

Curiosity Triggers:
- Moment, Story, Result, Transformation, Novelty

Scroll Stoppers:
- Faces with strong emotions, gestures, big numbers, arrows/circles/question marks

Branding:
- 1–2 fonts, 2–3 colors, layout framework
- Logos small, never bottom-right
- Prioritize curiosity & clarity

Workflow:
1. Understand script/title → emotional core
2. Research competitors → note patterns
3. Choose one strong visual concept
4. Layout: Hero, Supporting, Text
5. Enhance: glow, blur, overlay, arrows
6. Export: 1280×720, JPG/PNG <2MB, test small-screen readability

Analytics & Iteration:
- Monitor CTR, impressions, views
- A/B test one change at a time
- Iterate until CTR improves
"""

@app.route("/api/thumbnail_generation", methods=["POST"])
def thumbnail_generation():
    try:
        data = request.get_json()

        face_images = data.get("face_images", [])
        reference_images = data.get("reference_images", [])[:1]  # Enforce max 1 reference image
        if len(data.get("reference_images", [])) > 1:
            print("Warning: More than 1 reference image provided; using only the first.")

        user_prompt = data.get("prompt", "Generate a YouTube thumbnail")
        size = data.get("size", "1280x720")  # Default to YouTube standard
        model = data.get("model", "seedream-4-0-250828")

        if not face_images and not reference_images and not user_prompt:
            return jsonify({"error": "At least one input (prompt, face_images, or reference_images) is required"}), 400

        # Initialize Gemini client
        gemini_client = genai.GenerativeModel("gemini-2.0-flash")

        # Step 0: Describe faces and references
        face_descriptions = []
        uploaded_files = []
        for url in face_images:
            img_resp = requests.get(url, timeout=10)
            if img_resp.status_code == 200:
                image_bytes = img_resp.content
                mime_type = img_resp.headers.get('Content-Type', 'image/jpeg')
                uploaded_file = genai.upload_file(io.BytesIO(image_bytes), mime_type=mime_type)
                uploaded_files.append(uploaded_file)
                desc_resp = gemini_client.generate_content([
                    "Describe this person's appearance in detail for an image generation prompt: age, gender, ethnicity, facial expression, hair style/color, clothing, notable features. If used in a thumbnail with a reference image, match the pose, lighting, and emotion of any face in the reference for seamless integration. Emphasize expressive emotions (surprise or happy) for high CTR.",
                    uploaded_file
                ])
                face_descriptions.append(desc_resp.text.strip())

        reference_descriptions = []
        for url in reference_images:
            img_resp = requests.get(url, timeout=10)
            if img_resp.status_code == 200:
                image_bytes = img_resp.content
                mime_type = img_resp.headers.get('Content-Type', 'image/jpeg')
                uploaded_file = genai.upload_file(io.BytesIO(image_bytes), mime_type=mime_type)
                uploaded_files.append(uploaded_file)
                desc_resp = gemini_client.generate_content([
                    "Describe this YouTube thumbnail in detail: exact composition, hero element, supporting elements, text (content, placement, style, font, size), colors, background, emotions, faces/objects. Note high-CTR features like curiosity triggers and scroll-stoppers. For face replacement, note the pose, lighting, and emotion of any face to match.",
                    uploaded_file
                ])
                reference_descriptions.append(desc_resp.text.strip())

        # Step 1: Build base prompt for Gemini with enhanced scenario instructions
        scenario_instructions = ""
        if face_images and reference_images:
            scenario_instructions = """
Scenario: Face + Reference + Prompt. Generate an exact replica of the reference thumbnail: identical layout, colors, text (content, placement, style, font), background, supporting elements, and high-CTR features (curiosity triggers, scroll-stoppers). Replace any face in the reference with the provided face(s), matching the exact pose, lighting, and emotion of the original face for seamless integration. Use the user prompt only to guide context or minor text tweaks if specified, but do not alter the reference's design or composition.
"""
        elif face_images:
            scenario_instructions = """
Scenario: Face + Prompt. Feature the described face(s) prominently as the zoomed-in hero with strong, positive emotion (surprise or happy). Build supporting scene and text around the user prompt. Use high-contrast colors, bold sans-serif text with shadows, curiosity triggers like questions/arrows, and scroll-stoppers for max CTR.
"""
        elif reference_images:
            scenario_instructions = """
Scenario: Reference + Prompt. Replicate the reference thumbnail's style, layout, colors, text placement, composition, and CTR elements exactly, but adapt content to the user prompt. Enhance with better curiosity, emotions, and visuals if needed.
"""
        else:
            scenario_instructions = """
Scenario: Prompt Only. Create from scratch: Choose optimal thumbnail type (e.g., Face+Emotion if applicable). Include hero element, supporting scene, bold text (1-4 words), high-contrast colors, curiosity triggers, and scroll-stoppers for viral CTR.
"""

        gemini_base_prompt = f"""
You are a YouTube thumbnail design expert. Generate a detailed, vivid prompt for Seedream 4.0 to create a high-CTR thumbnail. Strictly follow best practices: emotional hero, curiosity text, contrasts, etc.

User prompt: "{user_prompt}"
Face descriptions: {', '.join(face_descriptions) if face_descriptions else 'None'}
Reference descriptions: {', '.join(reference_descriptions) if reference_descriptions else 'None'}

{scenario_instructions}

{THUMBNAIL_CONTEXT}

Output a single, detailed text prompt. Include specifics like 'bold red text "SHOCKING TRUTH!" top-center with shadow', 'high contrast yellow background', 'face with wide-eyed surprise'. End with: 'Generate the image in exact 1280x720 resolution, optimized for YouTube thumbnail (16:9 aspect ratio), high quality, no distortion.'
Respond in plain English, concise but complete, no formatting.
"""

        # Step 2: Generate refined prompt via Gemini
        gemini_resp = gemini_client.generate_content(gemini_base_prompt)
        refined_prompt = gemini_resp.text.strip()

        # Cleanup uploaded files
        for file in uploaded_files:
            genai.delete_file(file.name)

        # Step 3: Prepare Seedream API call
        all_images = face_images + reference_images
        payload = {
            "model": model,
            "prompt": refined_prompt,
            "image": all_images,
            "sequential_image_generation": "disabled",
            "response_format": "url",
            "size": size,
            "stream": False,
            "watermark": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {BYTEPLUS_API_KEY}"
        }

        resp = requests.post(
            "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations",
            headers=headers,
            json=payload,
            timeout=60
        )

        if resp.status_code != 200:
            return jsonify({"error": f"BytePlus API failed: {resp.text}"}), 500

        result = resp.json()
        result["refined_prompt"] = refined_prompt
        return jsonify(result)

    except Exception as e:
        for file in uploaded_files:
            try:
                genai.delete_file(file.name)
            except:
                pass
        return jsonify({"error": str(e)}), 500
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

@app.route("/api/shorts_by_channel_id", methods=["POST"])
def shorts_by_channel_id():
    try:
        data = request.get_json()
        channel_id = data.get("channel_id", "").strip()
        if not channel_id:
            return jsonify({"error": "Channel ID is required"}), 400

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # 1. Fetch channel info
        channel_resp = youtube.channels().list(
            part="snippet,statistics,contentDetails",
            id=channel_id
        ).execute()

        if not channel_resp.get("items"):
            return jsonify({"error": "Channel not found"}), 404

        channel_info = channel_resp["items"][0]
        channel_title = channel_info["snippet"]["title"]
        uploads_playlist = channel_info["contentDetails"]["relatedPlaylists"]["uploads"]

        # 2. Get last 10 uploads, filter shorts (≤60s)
        videos_resp = youtube.playlistItems().list(
            part="contentDetails,snippet",
            playlistId=uploads_playlist,
            maxResults=20
        ).execute()

        candidate_videos = []
        for item in videos_resp.get("items", []):
            vid_id = item["contentDetails"]["videoId"]
            vid_details = youtube.videos().list(
                part="contentDetails,snippet,statistics",
                id=vid_id
            ).execute()

            if not vid_details.get("items"):
                continue

            v = vid_details["items"][0]
            duration = safe_duration(v)
            if duration <= 60:  # shorts only
                candidate_videos.append({
                    "id": vid_id,
                    "title": v["snippet"]["title"]
                })

            if len(candidate_videos) >= 10:
                break

        if not candidate_videos:
            return jsonify({"error": "No Shorts found for this channel"}), 404

        # 3. Collect competitor channels with frequency
        competitor_counts = {}
        for video in candidate_videos:
            search_resp = youtube.search().list(
                part="snippet",
                q=video["title"],
                type="video",
                maxResults=20,
                relevanceLanguage="en",
                videoDuration="short"  # shorts only
            ).execute()

            for item in search_resp.get("items", []):
                ch_id = item["snippet"]["channelId"]
                ch_title = item["snippet"]["channelTitle"]

                if ch_id == channel_id:
                    continue

                if ch_id not in competitor_counts:
                    competitor_counts[ch_id] = {"title": ch_title, "count": 0}
                competitor_counts[ch_id]["count"] += 1

        # Sort competitors by frequency
        competitors = sorted(
            competitor_counts.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True
        )[:20]

        # 4. For each competitor, fetch top + latest shorts with avg views
        all_videos = []
        competitors_meta = []
        for comp_id, comp_data in competitors:
            comp_resp = youtube.channels().list(
                part="statistics,contentDetails",
                id=comp_id
            ).execute()

            if not comp_resp.get("items"):
                continue

            comp_info = comp_resp["items"][0]
            comp_title = comp_data["title"]
            comp_uploads = comp_info["contentDetails"]["relatedPlaylists"]["uploads"]

            # Fetch up to 50 uploads
            comp_uploads_resp = youtube.playlistItems().list(
                part="contentDetails",
                playlistId=comp_uploads,
                maxResults=50
            ).execute()

            comp_video_ids = [v["contentDetails"]["videoId"] for v in comp_uploads_resp.get("items", [])]
            comp_videos = fetch_video_details(youtube, comp_video_ids)

            # Filter only shorts
            comp_videos = [v for v in comp_videos if safe_duration(v) <= 60]

            if not comp_videos:
                continue

            # Calculate avg recent views
            avg_recent_views = sum(
                int(v["statistics"].get("viewCount", 0)) for v in comp_videos
            ) / len(comp_videos)

            competitors_meta.append({
                "id": comp_id,
                "title": comp_title,
                "subscriber_count": int(comp_info["statistics"].get("subscriberCount", 0)),
                "avg_recent_views": round(avg_recent_views, 2),
                "frequency": comp_data.get("count", 0)
            })

            # Top 5 by views
            top_videos = sorted(
                comp_videos,
                key=lambda x: int(x["statistics"].get("viewCount", 0)),
                reverse=True
            )[:5]

            # Latest 5 with multiplier >1
            latest_videos = []
            for v in comp_videos[:10]:
                views = int(v["statistics"].get("viewCount", 0))
                if avg_recent_views > 0 and (views / avg_recent_views) > 1:
                    latest_videos.append(v)
                if len(latest_videos) >= 5:
                    break

            # --- Start: New separate loops to assign 'list_type' and handle duplicates ---

            # 1. Process Top 5 videos
            for v in top_videos:
                duration = safe_duration(v)
                views = int(v["statistics"].get("viewCount", 0))
                multiplier = round(views / avg_recent_views, 2) if avg_recent_views > 0 else 0

                all_videos.append({
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel_id": comp_id,
                    "channel_title": comp_title,
                    "competitor_frequency": comp_data.get("count", 0),
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "multiplier": multiplier,
                    "avg_recent_views": round(avg_recent_views, 2),
                    "thumbnail_url": v["snippet"]["thumbnails"]["high"]["url"],
                    "url": f"https://www.youtube.com/watch?v={v['id']}",
                    "list_type": "top_shorts" # Correct list type
                })

            # Prepare IDs to prevent duplication
            top_video_ids = {v["id"] for v in top_videos}

            # 2. Process Latest Outlier videos (Skip if already in top_videos)
            for v in latest_videos:
                if v["id"] in top_video_ids:
                    continue
                    
                duration = safe_duration(v)
                views = int(v["statistics"].get("viewCount", 0))
                multiplier = round(views / avg_recent_views, 2) if avg_recent_views > 0 else 0

                all_videos.append({
                    "video_id": v["id"],
                    "title": v["snippet"]["title"],
                    "channel_id": comp_id,
                    "channel_title": comp_title,
                    "competitor_frequency": comp_data.get("count", 0),
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "multiplier": multiplier,
                    "avg_recent_views": round(avg_recent_views, 2),
                    "thumbnail_url": v["snippet"]["thumbnails"]["high"]["url"],
                    "url": f"https://www.youtube.com/watch?v={v['id']}",
                    "list_type": "latest_outlier_shorts" # Correct list type
                })
            if len(all_videos) >= 200:
                break

        return jsonify({
            "success": True,
            "channel_id": channel_id,
            "channel_title": channel_title,
            "competitors": competitors_meta,
            "total_videos": len(all_videos),
            "videos": all_videos[:200]
        })

    except Exception as e:
        app.logger.error(f"Shorts outliers error: {str(e)}")
        return jsonify({"error": str(e)}), 500


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


@app.route('/api/whole_script', methods=['POST'])
def whole_script():
    """ULTRA-OPTIMIZED: Fast script generation with parallel processing + ALL MEDIA TYPES"""
    user_id = request.remote_addr
    print(f"\n{'='*60}")
    print(f"FAST SCRIPT GENERATION: {user_id}")
    print(f"{'='*60}\n")
    
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import requests
        import base64
        
        # Parse request
        content_type = request.content_type or ''
        folders = []
        prompt = ""
        target_minutes = None
        
        if 'application/json' in content_type:
            data = request.json or {}
            prompt = data.get('prompt', '').strip()
            target_minutes = data.get('minutes')
            
            # ========================================
            # HANDLE JSON SCHEMA - URLs and Text
            # ========================================
            
            # YouTube URLs
            youtube_urls = data.get('youtube_urls', [])
            if youtube_urls and isinstance(youtube_urls, list):
                yt_folder = {'name': 'YouTube', 'type': 'inspiration', 'items': []}
                for url in youtube_urls:
                    if url and isinstance(url, str) and url.strip():
                        yt_folder['items'].append({'type': 'youtube_url', 'url': url.strip()})
                if yt_folder['items']:
                    folders.append(yt_folder)
                    print(f"Added {len(yt_folder['items'])} YouTube URLs")
            
            # Instagram URLs
            instagram_urls = data.get('instagram_urls', [])
            if instagram_urls and isinstance(instagram_urls, list):
                insta_folder = {'name': 'Instagram', 'type': 'inspiration', 'items': []}
                for url in instagram_urls:
                    if url and isinstance(url, str) and url.strip():
                        insta_folder['items'].append({'type': 'instagram_url', 'url': url.strip()})
                if insta_folder['items']:
                    folders.append(insta_folder)
                    print(f"Added {len(insta_folder['items'])} Instagram URLs")
            
            # Facebook URLs
            facebook_urls = data.get('facebook_urls', [])
            if facebook_urls and isinstance(facebook_urls, list):
                fb_folder = {'name': 'Facebook', 'type': 'inspiration', 'items': []}
                for url in facebook_urls:
                    if url and isinstance(url, str) and url.strip():
                        fb_folder['items'].append({'type': 'facebook_url', 'url': url.strip()})
                if fb_folder['items']:
                    folders.append(fb_folder)
                    print(f"Added {len(fb_folder['items'])} Facebook URLs")
            
            # TikTok URLs
            tiktok_urls = data.get('tiktok_urls', [])
            if tiktok_urls and isinstance(tiktok_urls, list):
                tiktok_folder = {'name': 'TikTok', 'type': 'inspiration', 'items': []}
                for url in tiktok_urls:
                    if url and isinstance(url, str) and url.strip():
                        tiktok_folder['items'].append({'type': 'tiktok_url', 'url': url.strip()})
                if tiktok_folder['items']:
                    folders.append(tiktok_folder)
                    print(f"Added {len(tiktok_folder['items'])} TikTok URLs")
            
            # Video Files (Download from URLs)
            video_files = data.get('video_files', [])
            if video_files and isinstance(video_files, list):
                video_folder = {'name': 'Videos', 'type': 'inspiration', 'items': []}
                for idx, video_url in enumerate(video_files):
                    if video_url and isinstance(video_url, str) and video_url.strip():
                        try:
                            print(f"Downloading video {idx+1}/{len(video_files)}: {video_url[:50]}...")
                            response = requests.get(video_url, timeout=60)
                            if response.status_code == 200:
                                filename = video_url.split('/')[-1].split('?')[0] or f'video_{idx+1}.mp4'
                                video_data = base64.b64encode(response.content).decode('utf-8')
                                video_folder['items'].append({
                                    'type': 'video_file',
                                    'filename': filename,
                                    'data': video_data
                                })
                                print(f"  ✓ Downloaded: {len(response.content):,} bytes")
                            else:
                                print(f"  ✗ Failed: HTTP {response.status_code}")
                        except Exception as e:
                            print(f"  ✗ Error downloading video {video_url}: {e}")
                if video_folder['items']:
                    folders.append(video_folder)
                    print(f"Added {len(video_folder['items'])} video files")
            
            # Audio Files (Download from URLs)
            audio_files = data.get('audio_files', [])
            if audio_files and isinstance(audio_files, list):
                audio_folder = {'name': 'Audio Files', 'type': 'inspiration', 'items': []}
                for idx, audio_url in enumerate(audio_files):
                    if audio_url and isinstance(audio_url, str) and audio_url.strip():
                        try:
                            print(f"Downloading audio {idx+1}/{len(audio_files)}: {audio_url[:50]}...")
                            response = requests.get(audio_url, timeout=60)
                            if response.status_code == 200:
                                filename = audio_url.split('/')[-1].split('?')[0] or f'audio_{idx+1}.mp3'
                                audio_data = base64.b64encode(response.content).decode('utf-8')
                                audio_folder['items'].append({
                                    'type': 'audio_file',
                                    'filename': filename,
                                    'data': audio_data
                                })
                                print(f"  ✓ Downloaded: {len(response.content):,} bytes")
                            else:
                                print(f"  ✗ Failed: HTTP {response.status_code}")
                        except Exception as e:
                            print(f"  ✗ Error downloading audio {audio_url}: {e}")
                if audio_folder['items']:
                    folders.append(audio_folder)
                    print(f"Added {len(audio_folder['items'])} audio files")
            
            # Image Files (Download from URLs)
            image_files = data.get('image_files', [])
            if image_files and isinstance(image_files, list):
                image_folder = {'name': 'Images', 'type': 'document', 'items': []}
                for idx, image_url in enumerate(image_files):
                    if image_url and isinstance(image_url, str) and image_url.strip():
                        try:
                            print(f"Downloading image {idx+1}/{len(image_files)}: {image_url[:50]}...")
                            response = requests.get(image_url, timeout=30)
                            if response.status_code == 200:
                                filename = image_url.split('/')[-1].split('?')[0] or f'image_{idx+1}.jpg'
                                image_data = base64.b64encode(response.content).decode('utf-8')
                                image_folder['items'].append({
                                    'type': 'image_file',
                                    'filename': filename,
                                    'data': image_data
                                })
                                print(f"  ✓ Downloaded: {len(response.content):,} bytes")
                            else:
                                print(f"  ✗ Failed: HTTP {response.status_code}")
                        except Exception as e:
                            print(f"  ✗ Error downloading image {image_url}: {e}")
                if image_folder['items']:
                    folders.append(image_folder)
                    print(f"Added {len(image_folder['items'])} image files")
            
            # Documents (Download from URLs)
            documents = data.get('documents', [])
            if documents and isinstance(documents, list):
                doc_folder = {'name': 'Documents', 'type': 'document', 'items': []}
                for idx, doc_url in enumerate(documents):
                    if doc_url and isinstance(doc_url, str) and doc_url.strip():
                        try:
                            print(f"Downloading document {idx+1}/{len(documents)}: {doc_url[:50]}...")
                            response = requests.get(doc_url, timeout=60)
                            if response.status_code == 200:
                                filename = doc_url.split('/')[-1].split('?')[0] or f'document_{idx+1}.pdf'
                                doc_data = base64.b64encode(response.content).decode('utf-8')
                                doc_folder['items'].append({
                                    'type': 'document',
                                    'filename': filename,
                                    'data': doc_data
                                })
                                print(f"  ✓ Downloaded: {len(response.content):,} bytes")
                            else:
                                print(f"  ✗ Failed: HTTP {response.status_code}")
                        except Exception as e:
                            print(f"  ✗ Error downloading document {doc_url}: {e}")
                if doc_folder['items']:
                    folders.append(doc_folder)
                    print(f"Added {len(doc_folder['items'])} documents")
            
            # ========================================
            # TEXT INPUTS - NEW SCHEMA
            # ========================================
        
            text_inputs_data = data.get('text_inputs')

            if text_inputs_data:
                text_folder = {'name': 'Text Inputs', 'type': 'document', 'items': []}
                
                # Case 1: Array of objects [{"content": "..."}, {"content": "..."}]
                if isinstance(text_inputs_data, list):
                    for idx, text_item in enumerate(text_inputs_data):
                        if isinstance(text_item, dict):
                            text_content = text_item.get('content', '').strip()
                            if text_content:
                                text_folder['items'].append({
                                    'type': 'text_input',
                                    'content': text_content,
                                    'name': f"Text Input {idx + 1}"
                                })
                                print(f"Added text input {idx+1} (object format): {len(text_content)} chars")
                        elif isinstance(text_item, str) and text_item.strip():
                            # Support direct strings in array too
                            text_folder['items'].append({
                                'type': 'text_input',
                                'content': text_item.strip(),
                                'name': f"Text Input {idx + 1}"
                            })
                            print(f"Added text input {idx+1} (string format): {len(text_item)} chars")
                
                # Case 2: Object with content array {"content": ["...", "..."]}
                elif isinstance(text_inputs_data, dict):
                    text_content_array = text_inputs_data.get('content', [])
                    
                    if isinstance(text_content_array, list):
                        for idx, text_content in enumerate(text_content_array):
                            if text_content and isinstance(text_content, str) and text_content.strip():
                                text_folder['items'].append({
                                    'type': 'text_input',
                                    'content': text_content.strip(),
                                    'name': f"Text Input {idx + 1}"
                                })
                                print(f"Added text input {idx+1} (nested array format): {len(text_content)} chars")
                
                # Case 3: Direct string (single text input)
                elif isinstance(text_inputs_data, str) and text_inputs_data.strip():
                    text_folder['items'].append({
                        'type': 'text_input',
                        'content': text_inputs_data.strip(),
                        'name': "Text Input 1"
                    })
                    print(f"Added text input 1 (direct string): {len(text_inputs_data)} chars")
                
                # Add folder if we got any items
                if text_folder['items']:
                    folders.append(text_folder)
                    print(f"✓ Added {len(text_folder['items'])} text inputs total")
        
        elif 'multipart/form-data' in content_type:
            prompt = request.form.get('prompt', '').strip()
            target_minutes = request.form.get('minutes', type=int)
            
            # Build folders from uploads
            uploaded_videos = request.files.getlist('video_files[]')
            uploaded_audio = request.files.getlist('audio_files[]')
            uploaded_docs = request.files.getlist('documents[]')
            uploaded_images = request.files.getlist('image_files[]')
            
            youtube_urls = request.form.getlist('youtube_urls[]')
            instagram_urls = request.form.getlist('instagram_urls[]')
            facebook_urls = request.form.getlist('facebook_urls[]')
            tiktok_urls = request.form.getlist('tiktok_urls[]')
            
            text_inputs = request.form.getlist('text_inputs[]')
            
            # Process uploaded video files
            if uploaded_videos:
                video_folder = {'name': 'Videos', 'type': 'inspiration', 'items': []}
                for vf in uploaded_videos:
                    if vf.filename:
                        video_folder['items'].append({
                            'type': 'video_file',
                            'filename': vf.filename,
                            'data': base64.b64encode(vf.read()).decode('utf-8')
                        })
                if video_folder['items']:
                    folders.append(video_folder)
            
            # Process uploaded audio files
            if uploaded_audio:
                audio_folder = {'name': 'Audio Files', 'type': 'inspiration', 'items': []}
                for af in uploaded_audio:
                    if af.filename:
                        audio_folder['items'].append({
                            'type': 'audio_file',
                            'filename': af.filename,
                            'data': base64.b64encode(af.read()).decode('utf-8')
                        })
                if audio_folder['items']:
                    folders.append(audio_folder)
            
            # Process uploaded documents
            if uploaded_docs:
                doc_folder = {'name': 'Documents', 'type': 'document', 'items': []}
                for df in uploaded_docs:
                    if df.filename:
                        doc_folder['items'].append({
                            'type': 'document',
                            'filename': df.filename,
                            'data': base64.b64encode(df.read()).decode('utf-8')
                        })
                if doc_folder['items']:
                    folders.append(doc_folder)
            
            # Process uploaded image files
            if uploaded_images:
                image_folder = {'name': 'Images', 'type': 'document', 'items': []}
                for img in uploaded_images:
                    if img.filename:
                        image_folder['items'].append({
                            'type': 'image_file',
                            'filename': img.filename,
                            'data': base64.b64encode(img.read()).decode('utf-8')
                        })
                if image_folder['items']:
                    folders.append(image_folder)
            
            # Process YouTube URLs
            if youtube_urls:
                yt_folder = {'name': 'YouTube', 'type': 'inspiration', 'items': []}
                for url in youtube_urls:
                    if url and url.strip():
                        yt_folder['items'].append({'type': 'youtube_url', 'url': url.strip()})
                if yt_folder['items']:
                    folders.append(yt_folder)
            
            # Process Instagram URLs
            if instagram_urls:
                insta_folder = {'name': 'Instagram', 'type': 'inspiration', 'items': []}
                for url in instagram_urls:
                    if url and url.strip():
                        insta_folder['items'].append({'type': 'instagram_url', 'url': url.strip()})
                if insta_folder['items']:
                    folders.append(insta_folder)
            
            # Process Facebook URLs
            if facebook_urls:
                fb_folder = {'name': 'Facebook', 'type': 'inspiration', 'items': []}
                for url in facebook_urls:
                    if url and url.strip():
                        fb_folder['items'].append({'type': 'facebook_url', 'url': url.strip()})
                if fb_folder['items']:
                    folders.append(fb_folder)
            
            # Process TikTok URLs
            if tiktok_urls:
                tiktok_folder = {'name': 'TikTok', 'type': 'inspiration', 'items': []}
                for url in tiktok_urls:
                    if url and url.strip():
                        tiktok_folder['items'].append({'type': 'tiktok_url', 'url': url.strip()})
                if tiktok_folder['items']:
                    folders.append(tiktok_folder)
            
            # Process Direct Text Inputs
            if text_inputs:
                text_folder = {'name': 'Text Inputs', 'type': 'document', 'items': []}
                for idx, text_content in enumerate(text_inputs):
                    if text_content and text_content.strip():
                        text_folder['items'].append({
                            'type': 'text_input',
                            'content': text_content.strip(),
                            'name': f"Text Input {idx + 1}"
                        })
                if text_folder['items']:
                    folders.append(text_folder)
        
        else:
            return jsonify({'error': 'Unsupported content type'}), 415
        
        if not prompt:
            prompt = "Create an engaging YouTube video script."
        
        print(f"Folders: {len(folders)}, Prompt: {prompt[:50]}...")
        
        # Results storage
        processed_personal = []
        processed_inspiration = []
        processed_documents = []
        errors = []
        
        # ========================================
        # PARALLEL PROCESSING OF ALL ITEMS
        # ========================================
        
        def process_item(folder_name, folder_type, item, item_idx):
            """Process a single item (video/audio/doc/youtube/instagram/facebook/tiktok/image/text)"""
            item_type = item.get('type')
            
            try:
                # YOUTUBE
                if item_type == 'youtube_url':
                    url = item.get('url', '').strip()
                    if url and video_processor.validate_youtube_url(url):
                        print(f"\n[{folder_name}] Processing YouTube: {url}")
                        result = video_processor.process_video_content(url, 'youtube')
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] YouTube {url}: {result['error']}")
                        
                        transcript = result['transcript']
                        print(f"\n{'='*80}")
                        print(f"YOUTUBE TRANSCRIPT EXTRACTED: {url}")
                        print(f"{'='*80}")
                        print(f"Length: {len(transcript):,} characters")
                        print(f"Words: {result['stats'].get('word_count', 0):,}")
                        print(f"\nPREVIEW (first 500 chars):")
                        print(f"{'-'*80}")
                        print(transcript[:500])
                        if len(transcript) > 500:
                            print(f"... (truncated, {len(transcript) - 500:,} more characters)")
                        print(f"{'='*80}\n")
                        
                        return (folder_type if folder_type == 'personal' else 'inspiration', {
                            'folder_name': folder_name,
                            'url': url,
                            'transcript': transcript,
                            'stats': result['stats'],
                            'type': 'youtube'
                        })
                    return ('error', f"[{folder_name}] Invalid YouTube URL")
                
                # INSTAGRAM
                elif item_type == 'instagram_url':
                    url = item.get('url', '').strip()
                    if url and instagram_processor.validate_instagram_url(url):
                        print(f"\n[{folder_name}] Processing Instagram: {url}")
                        result = instagram_processor.process_instagram_url(url)
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] Instagram {url}: {result['error']}")
                        
                        transcript = result['transcript']
                        
                        return (folder_type if folder_type == 'personal' else 'inspiration', {
                            'folder_name': folder_name,
                            'url': url,
                            'transcript': transcript,
                            'stats': result['stats'],
                            'type': 'instagram'
                        })
                    return ('error', f"[{folder_name}] Invalid Instagram URL")
                
                # FACEBOOK
                elif item_type == 'facebook_url':
                    url = item.get('url', '').strip()
                    if url and facebook_processor.validate_facebook_url(url):
                        print(f"\n[{folder_name}] Processing Facebook: {url}")
                        result = facebook_processor.process_facebook_url(url)
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] Facebook {url}: {result['error']}")
                        
                        transcript = result['transcript']
                        
                        return (folder_type if folder_type == 'personal' else 'inspiration', {
                            'folder_name': folder_name,
                            'url': url,
                            'transcript': transcript,
                            'stats': result['stats'],
                            'type': 'facebook'
                        })
                    return ('error', f"[{folder_name}] Invalid Facebook URL")
                
                # TIKTOK
                elif item_type == 'tiktok_url':
                    url = item.get('url', '').strip()
                    # if url and tiktok_processor.validate_tiktok_url(url):
                    #     print(f"\n[{folder_name}] Processing TikTok: {url}")
                    #     result = tiktok_processor.process_tiktok_url(url)
                        
                    #     if result['error']:
                    #         return ('error', f"[{folder_name}] TikTok {url}: {result['error']}")
                        
                    #     transcript = result['transcript']
                        
                    #     return (folder_type if folder_type == 'personal' else 'inspiration', {
                    #         'folder_name': folder_name,
                    #         'url': url,
                    #         'transcript': transcript,
                    #         'stats': result['stats'],
                    #         'type': 'tiktok'
                    #     })
                    return ('error', f"[{folder_name}] TikTok processing not implemented")
                
                # AUDIO FILE
                elif item_type == 'audio_file':
                    filename = item.get('filename', 'audio.mp3')
                    file_data = item.get('data')
                    
                    if not audio_processor.is_supported_audio_format(filename):
                        return ('error', f"[{folder_name}] Unsupported audio format: {filename}")
                    
                    if not file_data:
                        return ('error', f"[{folder_name}] No data: {filename}")
                    
                    print(f"\n[{folder_name}] Processing Audio File: {filename}")
                    
                    safe_user_id = user_id.replace('.', '_').replace(':', '_')
                    audio_path = os.path.join(
                        UPLOAD_FOLDER,
                        f"temp_{safe_user_id}_{int(time.time())}_{item_idx}_{secure_filename(filename)}"
                    )
                    
                    try:
                        audio_bytes = base64.b64decode(file_data)
                        with open(audio_path, 'wb') as f:
                            f.write(audio_bytes)
                        
                        print(f"  Saved: {len(audio_bytes):,} bytes ({len(audio_bytes)/(1024*1024):.2f} MB)")
                        
                        result = audio_processor.process_audio_file(audio_path, filename)
                        
                        try:
                            os.remove(audio_path)
                        except:
                            pass
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] Audio {filename}: {result['error']}")
                        
                        transcript = result['transcript']
                        
                        return (folder_type if folder_type == 'personal' else 'inspiration', {
                            'folder_name': folder_name,
                            'source': filename,
                            'transcript': transcript,
                            'stats': result['stats'],
                            'type': 'audio_file'
                        })
                    
                    except Exception as e:
                        if os.path.exists(audio_path):
                            try:
                                os.remove(audio_path)
                            except:
                                pass
                        return ('error', f"[{folder_name}] Audio error {filename}: {str(e)}")
                
                # VIDEO FILE
                elif item_type == 'video_file':
                    filename = item.get('filename', 'video.mp4')
                    file_data = item.get('data')
                    
                    if not video_processor.is_supported_video_format(filename):
                        return ('error', f"[{folder_name}] Unsupported: {filename}")
                    
                    if not file_data:
                        return ('error', f"[{folder_name}] No data: {filename}")
                    
                    print(f"\n[{folder_name}] Processing Video File: {filename}")
                    
                    safe_user_id = user_id.replace('.', '_').replace(':', '_')
                    video_path = os.path.join(
                        UPLOAD_FOLDER,
                        f"temp_{safe_user_id}_{int(time.time())}_{item_idx}_{secure_filename(filename)}"
                    )
                    
                    try:
                        video_bytes = base64.b64decode(file_data)
                        with open(video_path, 'wb') as f:
                            f.write(video_bytes)
                        
                        print(f"  Saved: {len(video_bytes):,} bytes ({len(video_bytes)/(1024*1024):.2f} MB)")
                        
                        result = video_processor.process_video_content(video_path, 'local')
                        
                        try:
                            os.remove(video_path)
                        except:
                            pass
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] Video {filename}: {result['error']}")
                        
                        transcript = result['transcript']
                        
                        return (folder_type if folder_type == 'personal' else 'inspiration', {
                            'folder_name': folder_name,
                            'source': filename,
                            'transcript': transcript,
                            'stats': result['stats'],
                            'type': 'local_video'
                        })
                    
                    except Exception as e:
                        if os.path.exists(video_path):
                            try:
                                os.remove(video_path)
                            except:
                                pass
                        return ('error', f"[{folder_name}] Video error {filename}: {str(e)}")
                
                # DOCUMENT
                elif item_type == 'document':
                    filename = item.get('filename', 'doc.pdf')
                    file_data = item.get('data')
                    
                    if not document_processor.allowed_file(filename):
                        return ('error', f"[{folder_name}] Unsupported doc: {filename}")
                    
                    if not file_data:
                        return ('error', f"[{folder_name}] No data: {filename}")
                    
                    print(f"\n[{folder_name}] Processing Document: {filename}")
                    
                    safe_user_id = user_id.replace('.', '_').replace(':', '_')
                    file_path = os.path.join(
                        UPLOAD_FOLDER,
                        f"temp_{safe_user_id}_{int(time.time())}_{item_idx}_{secure_filename(filename)}"
                    )
                    
                    try:
                        doc_bytes = base64.b64decode(file_data)
                        with open(file_path, 'wb') as f:
                            f.write(doc_bytes)
                        
                        print(f"  Saved: {len(doc_bytes):,} bytes ({len(doc_bytes)/(1024*1024):.2f} MB)")
                        
                        result = document_processor.process_document(file_path, filename)
                        
                        try:
                            os.remove(file_path)
                        except:
                            pass
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] Doc {filename}: {result['error']}")
                        
                        doc_text = result['text']
                        
                        return ('document', {
                            'folder_name': folder_name,
                            'filename': filename,
                            'text': doc_text,
                            'stats': result['stats']
                        })
                    
                    except Exception as e:
                        if os.path.exists(file_path):
                            try:
                                os.remove(file_path)
                            except:
                                pass
                        return ('error', f"[{folder_name}] Doc error {filename}: {str(e)}")
                
                # IMAGE FILE
                elif item_type == 'image_file':
                    filename = item.get('filename', 'image.jpg')
                    file_data = item.get('data')
                    
                    if not image_processor.is_supported_image_format(filename):
                        return ('error', f"[{folder_name}] Unsupported image format: {filename}")
                    
                    if not file_data:
                        return ('error', f"[{folder_name}] No data: {filename}")
                    
                    print(f"\n[{folder_name}] Processing Image: {filename}")
                    
                    safe_user_id = user_id.replace('.', '_').replace(':', '_')
                    image_path = os.path.join(
                        UPLOAD_FOLDER,
                        f"temp_{safe_user_id}_{int(time.time())}_{item_idx}_{secure_filename(filename)}"
                    )
                    
                    try:
                        image_bytes = base64.b64decode(file_data)
                        with open(image_path, 'wb') as f:
                            f.write(image_bytes)
                        
                        print(f"  Saved: {len(image_bytes):,} bytes ({len(image_bytes)/(1024*1024):.2f} MB)")
                        
                        result = image_processor.process_image_with_gemini(image_path, filename)
                        
                        try:
                            os.remove(image_path)
                        except:
                            pass
                        
                        if result['error']:
                            return ('error', f"[{folder_name}] Image {filename}: {result['error']}")
                        
                        extracted_text = result['text']
                        
                        return ('document', {
                            'folder_name': folder_name,
                            'filename': filename,
                            'text': extracted_text,
                            'stats': result['stats']
                        })
                    
                    except Exception as e:
                        if os.path.exists(image_path):
                            try:
                                os.remove(image_path)
                            except:
                                pass
                        return ('error', f"[{folder_name}] Image error {filename}: {str(e)}")
                
                # TEXT INPUT
                elif item_type == 'text_input':
                    text_content = item.get('content', '').strip()
                    text_name = item.get('name', 'Text Input')
                    
                    if not text_content:
                        return ('error', f"[{folder_name}] Empty text input")
                    
                    print(f"\n[{folder_name}] Processing Text: {text_name}")
                    
                    result = text_processor.process_text_input(text_content, text_name)
                    
                    if result['error']:
                        return ('error', f"[{folder_name}] Text {text_name}: {result['error']}")
                    
                    processed_text = result['text']
                    
                    return ('document', {
                        'folder_name': folder_name,
                        'source_name': text_name,
                        'text': processed_text,
                        'stats': result['stats']
                    })
                
                return ('error', f"[{folder_name}] Unknown type: {item_type}")
            
            except Exception as e:
                return ('error', f"[{folder_name}] Item error: {str(e)}")
        
        # Process all items in parallel
        print(f"\n{'='*60}")
        print(f"PARALLEL PROCESSING START")
        print(f"{'='*60}\n")
        
        all_tasks = []
        for folder in folders:
            folder_name = folder.get('name', 'Unnamed')
            folder_type = folder.get('type', 'inspiration')
            items = folder.get('items', [])
            
            for idx, item in enumerate(items):
                all_tasks.append((folder_name, folder_type, item, idx))
        
        print(f"Total tasks: {len(all_tasks)}")
        
        # Process with ThreadPoolExecutor
        max_workers = min(5, len(all_tasks)) if all_tasks else 1
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_item, fn, ft, item, idx): (fn, ft, item, idx)
                for fn, ft, item, idx in all_tasks
            }
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                result_type, result_data = future.result()
                
                if result_type == 'error':
                    errors.append(result_data)
                    print(f"[{completed}/{len(all_tasks)}] ❌ Error")
                elif result_type == 'personal':
                    processed_personal.append(result_data)
                    print(f"[{completed}/{len(all_tasks)}] ✓ Personal video")
                elif result_type == 'inspiration':
                    processed_inspiration.append(result_data)
                    print(f"[{completed}/{len(all_tasks)}] ✓ Inspiration")
                elif result_type == 'document':
                    processed_documents.append(result_data)
                    print(f"[{completed}/{len(all_tasks)}] ✓ Document")
        
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Personal: {len(processed_personal)}")
        print(f"Inspiration: {len(processed_inspiration)}")
        print(f"Documents: {len(processed_documents)}")
        print(f"Errors: {len(errors)}\n")
        
        # ========================================
        # PARALLEL ANALYSIS
        # ========================================
        
        print(f"{'='*60}")
        print(f"PARALLEL ANALYSIS")
        print(f"{'='*60}\n")
        
        style_profile = "Professional YouTube style."
        inspiration_summary = "Best practices for engaging content."
        document_insights = "General knowledge for informative content."
        
        def analyze_style():
            if processed_personal:
                transcripts = [v['transcript'] for v in processed_personal]
                result = script_generator.analyze_creator_style(transcripts)
                
                print(f"\n{'='*80}")
                print(f"STYLE ANALYSIS COMPLETE")
                print(f"{'='*80}")
                print(result[:1000] if len(result) > 1000 else result)
                if len(result) > 1000:
                    print(f"... (truncated, {len(result) - 1000} more chars)")
                print(f"{'='*80}\n")
                
                return result
            return style_profile
        
        def analyze_inspiration():
            if processed_inspiration:
                transcripts = [v['transcript'] for v in processed_inspiration]
                result = script_generator.analyze_inspiration_content(transcripts)
                
                print(f"\n{'='*80}")
                print(f"INSPIRATION ANALYSIS COMPLETE")
                print(f"{'='*80}")
                print(result[:1000] if len(result) > 1000 else result)
                if len(result) > 1000:
                    print(f"... (truncated, {len(result) - 1000} more chars)")
                print(f"{'='*80}\n")
                
                return result
            return inspiration_summary
        
        def analyze_documents():
            if processed_documents:
                texts = [d['text'] for d in processed_documents]
                result = script_generator.analyze_documents(texts)
                
                print(f"\n{'='*80}")
                print(f"DOCUMENT ANALYSIS COMPLETE")
                print(f"{'='*80}")
                print(result[:1000] if len(result) > 1000 else result)
                if len(result) > 1000:
                    print(f"... (truncated, {len(result) - 1000} more chars)")
                print(f"{'='*80}\n")
                
                return result
            return document_insights
        
        # Run analyses in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            style_future = executor.submit(analyze_style)
            inspiration_future = executor.submit(analyze_inspiration)
            docs_future = executor.submit(analyze_documents)
            
            style_profile = style_future.result()
            inspiration_summary = inspiration_future.result()
            document_insights = docs_future.result()
        
        print("✓ All analyses complete\n")
        
        # ========================================
        # GENERATE SCRIPT
        # ========================================
        
        print(f"{'='*60}")
        print(f"GENERATING SCRIPT")
        print(f"{'='*60}\n")
        
        script = script_generator.generate_enhanced_script(
            style_profile,
            inspiration_summary,
            document_insights,
            prompt,
            target_minutes
        )
        
        print(f"\n{'='*80}")
        print(f"FINAL SCRIPT GENERATED")
        print(f"{'='*80}")
        print(f"Length: {len(script):,} characters")
        print(f"\nFULL SCRIPT:")
        print(f"{'-'*80}")
        print(script)
        print(f"{'='*80}\n")
        
        print("✓ Script generated\n")
        
        # Store session
        chat_session_id = str(uuid.uuid4())
        user_data[user_id]['current_script'] = {
            'content': script,
            'style_profile': style_profile,
            'topic_insights': inspiration_summary,
            'document_insights': document_insights,
            'original_prompt': prompt,
            'target_minutes': target_minutes,
            'timestamp': datetime.now().isoformat()
        }
        
        user_data[user_id]['chat_sessions'][chat_session_id] = {
            'messages': [],
            'script_versions': [script],
            'created_at': datetime.now().isoformat()
        }
        
        # Response
        stats = {
            'folders_processed': len(folders),
            'personal_videos': len(processed_personal),
            'inspiration_sources': len(processed_inspiration),
            'documents': len(processed_documents),
            'total_sources': len(processed_personal) + len(processed_inspiration) + len(processed_documents),
            'errors_count': len(errors),
            'target_duration': target_minutes
        }
        
        print(f"\n{'='*60}")
        print(f"✓ GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Stats: {json.dumps(stats, indent=2)}")
        print(f"{'='*60}\n")
        
        return jsonify({
            'success': True,
            'script': script,
            'chat_session_id': chat_session_id,
            'stats': stats,
            'processed_content': {
                'personal_videos': len(processed_personal),
                'inspiration_sources': len(processed_inspiration),
                'documents': len(processed_documents),
                'video_files': len([v for v in processed_inspiration + processed_personal if v.get('type') == 'local_video']),
                'audio_files': len([v for v in processed_inspiration + processed_personal if v.get('type') == 'audio_file']),
                'instagram_reels': len([v for v in processed_inspiration + processed_personal if v.get('type') == 'instagram']),
                'facebook_videos': len([v for v in processed_inspiration + processed_personal if v.get('type') == 'facebook']),
                'tiktok_videos': len([v for v in processed_inspiration + processed_personal if v.get('type') == 'tiktok']),
                'images': len([d for d in processed_documents if d.get('stats', {}).get('source_type') == 'image']),
                'text_inputs': len([d for d in processed_documents if d.get('stats', {}).get('source_type') == 'text_input'])
            },
            'errors': errors if errors else None,
            'analysis_quality': 'premium' if (processed_personal and processed_inspiration and processed_documents) else 
                               'optimal' if any([processed_personal, processed_inspiration, processed_documents]) else 
                               'basic',
            'folder_summary': [
                {
                    'name': folder.get('name'),
                    'type': folder.get('type'),
                    'items_count': len(folder.get('items', [])),
                    'processed_successfully': len([
                        item for item in folder.get('items', [])
                        if not any(folder.get('name') in err for err in errors)
                    ])
                }
                for folder in folders
            ]
        })
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"CRITICAL ERROR")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        import traceback
        print(f"Traceback:\n{traceback.format_exc()}")
        print(f"{'='*60}\n")
        return jsonify({'error': f'Script generation failed: {str(e)}'}), 500

@app.route("/api/shorts_videos", methods=["POST"])
def shorts_videos():
    try:
        data = request.get_json()
        channel_ids = data.get("channel_ids", [])
        if not channel_ids:
            return jsonify({"error": "Channel IDs list is required"}), 400

        youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        channel_videos = []

        for ch_id in channel_ids:
            # Fetch channel info
            ch_resp = youtube.channels().list(
                part="snippet,statistics",
                id=ch_id
            ).execute()

            if not ch_resp.get("items"):
                continue

            ch_info = ch_resp["items"][0]
            ch_title = ch_info["snippet"]["title"]
            subs_count = int(ch_info["statistics"].get("subscriberCount", 1))

            # Fetch Shorts only (under 60s)
            search_resp = youtube.search().list(
                part="snippet",
                channelId=ch_id,
                type="video",
                order="date",
                videoDuration="short",
                maxResults=50
            ).execute()

            video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
            if not video_ids:
                continue

            comp_videos = fetch_video_details(youtube, video_ids)
            if not comp_videos:
                continue

            avg_recent_views = sum(
                int(v["statistics"].get("viewCount", 0)) for v in comp_videos
            ) / len(comp_videos)

            # Top 10 popular Shorts
            popular_videos = sorted(
                comp_videos,
                key=lambda x: int(x["statistics"].get("viewCount", 0)),
                reverse=True
            )[:10]

            # ---------------------------------------------------------
            # CHANGE 1: Get Latest Videos (Simple slice, no calculation)
            # ---------------------------------------------------------
            latest_videos = comp_videos[:10]

            def format_video(v, list_type):
                duration_str = v.get("contentDetails", {}).get("duration", "")
                duration = parse_duration(duration_str)
                views = int(v["statistics"].get("viewCount", 0))
                multiplier = round(views / avg_recent_views, 2) if avg_recent_views > 0 else 0

                snippet = v.get("snippet", {}) or {}
                language = snippet.get("defaultAudioLanguage") or snippet.get("defaultLanguage") or snippet.get("language") or None

                published_at = snippet.get("publishedAt")
                published_date_friendly = None
                if published_at:
                    try:
                        if published_at.endswith("Z"):
                            try:
                                dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                            except ValueError:
                                dt = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%S.%fZ")
                        else:
                            dt = datetime.fromisoformat(published_at)
                        published_date_friendly = dt.strftime("%Y-%m-%d")
                    except Exception:
                        published_date_friendly = published_at

                return {
                    "video_id": v["id"],
                    "title": snippet.get("title"),
                    "channel_id": ch_id,
                    "channel_title": ch_title,
                    "subscriber_count": subs_count,
                    "views": views,
                    "views_formatted": format_number(views),
                    "duration_seconds": duration,
                    "multiplier": multiplier,
                    "avg_recent_views": round(avg_recent_views, 2),
                    "channel_avg_views_formatted": format_number(round(avg_recent_views, 0)),
                    "thumbnail_url": snippet.get("thumbnails", {}).get("high", {}).get("url"),
                    "url": f"https://www.youtube.com/watch?v={v['id']}",
                    "language": language,
                    "published_date": published_at,
                    "published_date_friendly": published_date_friendly,
                    "list_type": list_type
                }

            channel_data = {
                "channel_id": ch_id,
                "channel_title": ch_title,
                "subscriber_count": subs_count,
                "avg_recent_views": round(avg_recent_views, 2),
                "avg_recent_views_formatted": format_number(round(avg_recent_views, 0)),
                # ---------------------------------------------------------
                # CHANGE 2: Use "latest" key and pass "latest" list type
                # ---------------------------------------------------------
                "latest": [format_video(v, "latest") for v in latest_videos],
                "popular": [format_video(v, "popular") for v in popular_videos] 
            }

            channel_videos.append(channel_data)

        # Interleave latest/popular Shorts across channels
        all_videos = []
        max_videos_per_type = 10 

        for i in range(max_videos_per_type):
            for j, channel in enumerate(channel_videos):
                # ---------------------------------------------------------
                # CHANGE 3: Look for "latest" in channel dict, NOT "trending"
                # ---------------------------------------------------------
                if j % 2 == 0 and i < len(channel["latest"]):
                    all_videos.append(channel["latest"][i])
                elif j % 2 == 1 and i < len(channel["popular"]):
                    all_videos.append(channel["popular"][i])
                if len(all_videos) >= 200:
                    break
            if len(all_videos) >= 200:
                break

        return jsonify({
            "success": True,
            "total_videos": len(all_videos),
            "videos": all_videos[:200]
        })

    except Exception as e:
        app.logger.error(f"Shorts outliers error: {str(e)}")
        return jsonify({"error": str(e)}), 500
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
