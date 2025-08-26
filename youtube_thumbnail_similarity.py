import random
from datetime import datetime, timedelta
import googleapiclient.discovery
from isodate import parse_duration
import re
import numpy as np
from flask import current_app
import requests
from io import BytesIO
from PIL import Image
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import threading
from functools import lru_cache
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file")

# Mock OutlierDetector for formatting numbers (replace with your actual implementation)
class OutlierDetector:
    @staticmethod
    def format_number(number):
        """Format large numbers into human-readable strings (e.g., 1000 -> '1K')."""
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number / 1_000:.1f}K"
        return str(number)

outlier_detector = OutlierDetector()

# Existing global variables and class setup (unchanged)
tf.config.experimental.enable_memory_growth = True
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

_model_instance = None
_model_lock = threading.Lock()

class UltraFastYouTubeSimilarity:
    def __init__(self):
        self._session = self._create_optimized_session()
        self._model = self._get_or_create_model()
        self._feature_cache = {}
        self._batch_size = 64
        self._max_workers = 16

    def _create_optimized_session(self):
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            max_retries=1,
            pool_connections=50,
            pool_maxsize=50,
            pool_block=False
        )
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _get_or_create_model(self):
        global _model_instance, _model_lock
        if _model_instance is None:
            with _model_lock:
                if _model_instance is None:
                    _model_instance = self._create_ultra_fast_model()
        return _model_instance

    def _create_ultra_fast_model(self):
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras import Model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)
        model = Model(inputs=base_model.input, outputs=x)
        model.compile(optimizer='adam')
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        model.predict(dummy_input, verbose=0)
        print("🚀 Ultra-fast EfficientNetB0 model loaded and warmed up!")
        return model

    def _get_video_durations_batch(self, video_ids):
        """Get durations and statistics for multiple videos in a single API call."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        details_map = {}
        for i in range(0, len(video_ids), 50):
            batch_ids = video_ids[i:i+50]
            try:
                request = youtube.videos().list(
                    part='contentDetails,snippet,statistics',
                    id=','.join(batch_ids)
                )
                response = request.execute()
                for item in response['items']:
                    video_id = item['id']
                    duration = item['contentDetails']['duration']
                    duration_seconds = self._parse_duration(duration)
                    details_map[video_id] = {
                        'duration': duration,
                        'duration_seconds': duration_seconds,
                        'channel_id': item['snippet']['channelId'],
                        'channel_title': item['snippet']['channelTitle'],
                        'views': int(item['statistics'].get('viewCount', 0)),
                        'likes': int(item['statistics'].get('likeCount', 0)),
                        'comments': int(item['statistics'].get('commentCount', 0)),
                        'published_at': item['snippet']['publishedAt'],
                        'language': item['snippet'].get('defaultLanguage', 'en')
                    }
            except Exception as e:
                print(f"❌ Error getting details for batch: {e}")
                for video_id in batch_ids:
                    details_map[video_id] = {
                        'duration': 'PT0S',
                        'duration_seconds': 0,
                        'channel_id': '',
                        'channel_title': '',
                        'views': 0,
                        'likes': 0,
                        'comments': 0,
                        'published_at': '',
                        'language': 'en'
                    }
        return details_map

    def _parse_duration(self, duration):
        """Parse ISO 8601 duration to seconds."""
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        return 0

    def _get_channel_details(self, channel_id):
        """Get channel details including subscriber count and average views."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        try:
            request = youtube.channels().list(
                part='statistics,contentDetails',
                id=channel_id
            )
            response = request.execute()
            if response['items']:
                stats = response['items'][0]['statistics']
                subscriber_count = int(stats.get('subscriberCount', 0))
                uploads_playlist_id = response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
                # Get recent videos to calculate average views
                playlist_request = youtube.playlistItems().list(
                    part='contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=10
                )
                playlist_response = playlist_request.execute()
                video_ids = [item['contentDetails']['videoId'] for item in playlist_response['items']]
                if video_ids:
                    videos_request = youtube.videos().list(
                        part='statistics',
                        id=','.join(video_ids)
                    )
                    videos_response = videos_request.execute()
                    view_counts = [
                        int(item['statistics'].get('viewCount', 0))
                        for item in videos_response['items']
                    ]
                    avg_views = sum(view_counts) / len(view_counts) if view_counts else 0
                else:
                    avg_views = 0
                return subscriber_count, avg_views
            return 0, 0
        except Exception as e:
            print(f"❌ Error getting channel details for {channel_id}: {e}")
            return 0, 0

    def _filter_shorts_strict(self, videos, max_duration=60):
        """Strictly filter out YouTube Shorts based on duration."""
        print(f"🔍 Strictly filtering Shorts (≤{max_duration}s) from {len(videos)} videos...")
        if not videos:
            return []
        video_ids = [video[0] for video in videos]
        details_map = self._get_video_durations_batch(video_ids)
        filtered_videos = []
        shorts_count = 0
        for video in videos:
            video_id = video[0]
            details = details_map.get(video_id, {})
            duration_seconds = details.get('duration_seconds', 0)
            if duration_seconds > max_duration:
                filtered_videos.append(video + [details])
            else:
                shorts_count += 1
        print(f"🚫 Filtered out {shorts_count} YouTube Shorts (≤{max_duration}s)")
        print(f"✅ Kept {len(filtered_videos)} regular videos (>{max_duration}s)")
        return filtered_videos

    def _get_channel_videos_no_shorts(self, channel_id, max_results=50):
        """Get videos from a specific channel, strictly excluding Shorts."""
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
            raw_videos = []
            next_page_token = None
            while len(raw_videos) < max_results * 3:
                playlist_request = youtube.playlistItems().list(
                    part='snippet',
                    playlistId=uploads_playlist_id,
                    maxResults=50,
                    pageToken=next_page_token
                )
                playlist_response = playlist_request.execute()
                for item in playlist_response['items']:
                    video_id = item['snippet']['resourceId']['videoId']
                    title = item['snippet']['title']
                    thumbnail_url = item['snippet']['thumbnails']['high']['url']
                    raw_videos.append([video_id, title, thumbnail_url])
                next_page_token = playlist_response.get('nextPageToken')
                if not next_page_token:
                    break
            print(f"📺 Found {len(raw_videos)} raw videos from channel")
            filtered_videos = self._filter_shorts_strict(raw_videos)
            videos = filtered_videos[:max_results]
        except Exception as e:
            print(f"❌ Error getting channel videos: {e}")
        print(f"📺 Final channel videos: {len(videos)} (no Shorts)")
        return videos

    def _search_videos_no_shorts(self, youtube, query, max_results):
        """Search for videos with strict Shorts filtering."""
        try:
            search_results = max_results * 2
            request = youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=min(search_results, 50),
                order='relevance',
                regionCode='US',
                videoDuration='medium'
            )
            response = request.execute()
            raw_videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                thumbnail_url = item['snippet']['thumbnails']['high']['url']
                raw_videos.append([video_id, title, thumbnail_url])
            filtered_videos = self._filter_shorts_strict(raw_videos)
            return filtered_videos[:max_results]
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return []

    def _get_smart_related_videos_no_shorts(self, video_title, input_video_id, channel_id=None, max_results=200):
        """Smart video fetching with strict Shorts filtering and reduced title-based searching."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        all_videos = []
        video_ids_seen = {input_video_id}
        if channel_id:
            print(f"🎯 Searching in same channel first...")
            channel_videos = self._get_channel_videos_no_shorts(channel_id, max_results=50)
            for video in channel_videos:
                video_id = video[0]
                if video_id not in video_ids_seen:
                    video_ids_seen.add(video_id)
                    all_videos.append(video)
            print(f"📺 Found {len(channel_videos)} videos from same channel (no Shorts)")
        remaining_results = max_results - len(all_videos)
        if remaining_results > 0:
            print(f"🌍 Searching globally for {remaining_results} more videos...")
            search_strategies = [
                ' '.join([w for w in video_title.split() 
                         if len(w) > 3 and w.lower() not in [
                             'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                             'with', 'by', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 
                             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                             'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                             'might', 'must', 'can', 'video', 'youtube', 'watch', 'episode',
                             'part', 'full', 'complete', 'official'
                         ]])[:30],
                next((w for w in video_title.split() if len(w) > 4), video_title.split()[0] if video_title.split() else ''),
                ' '.join([w for w in video_title.split() 
                         if any(char.isupper() for char in w) and len(w) > 3])[:25]
            ]
            search_strategies = [s.strip() for s in search_strategies if s.strip()]
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_query = {}
                for query in search_strategies[:3]:
                    if len(all_videos) >= max_results:
                        break
                    remaining = min(25, remaining_results // len(search_strategies))
                    future = executor.submit(self._search_videos_no_shorts, youtube, query, remaining)
                    future_to_query[future] = query
                for future in concurrent.futures.as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        videos = future.result()
                        for video in videos:
                            if video[0] not in video_ids_seen and len(all_videos) < max_results:
                                video_ids_seen.add(video[0])
                                all_videos.append(video)
                    except Exception as e:
                        print(f"❌ Search failed for query '{query}': {e}")
        print(f"✅ Final video count: {len(all_videos)} (strictly no Shorts)")
        return all_videos

    def _preprocess_image_ultra_fast(self, image_url):
        cache_key = hash(image_url)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        try:
            if image_url.startswith('http'):
                response = self._session.get(
                    image_url, 
                    timeout=5,
                    headers={'User-Agent': 'Mozilla/5.0'},
                    stream=True
                )
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = (img_array - 0.5) * 2.0
                self._feature_cache[cache_key] = img_array
                return img_array
            else:
                img = Image.open(image_url)
                img = img.convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = (img_array - 0.5) * 2.0
                return img_array
        except Exception as e:
            print(f"❌ Error processing {image_url}: {e}")
            return None

    def _extract_features_batch(self, image_urls):
        features = []
        valid_urls = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_url = {executor.submit(self._preprocess_image_ultra_fast, url): url for url in image_urls}
            batch_images = []
            batch_urls = []
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                img_array = future.result()
                if img_array is not None:
                    batch_images.append(img_array)
                    batch_urls.append(url)
                    if len(batch_images) >= self._batch_size:
                        batch_features = self._extract_features_from_batch(batch_images)
                        features.extend(batch_features)
                        valid_urls.extend(batch_urls)
                        batch_images = []
                        batch_urls = []
            if batch_images:
                batch_features = self._extract_features_from_batch(batch_images)
                features.extend(batch_features)
                valid_urls.extend(batch_urls)
        return np.array(features), valid_urls

    def _extract_features_from_batch(self, batch_images):
        if not batch_images:
            return []
        batch_array = np.array(batch_images)
        features = self._model.predict(batch_array, verbose=0)
        return features.tolist()

    def _compute_pure_visual_similarity(self, query_features, database_features):
        if len(query_features) == 0 or len(database_features) == 0:
            return []
        query_features = np.array(query_features).reshape(1, -1)
        database_features = np.array(database_features)
        cosine_sim = cosine_similarity(query_features, database_features)[0]
        query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)
        database_norm = database_features / (np.linalg.norm(database_features, axis=1, keepdims=True) + 1e-8)
        dot_sim = np.dot(database_norm, query_norm.T).flatten()
        combined_similarity = 0.8 * cosine_sim + 0.2 * dot_sim
        return combined_similarity

@lru_cache(maxsize=200)
def extract_video_id(url):
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

@lru_cache(maxsize=200)
def get_video_details(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.videos().list(part='snippet,statistics', id=video_id)
        response = request.execute()
        if response['items']:
            snippet = response['items'][0]['snippet']
            stats = response['items'][0]['statistics']
            title = snippet['title']
            thumbnail_url = snippet['thumbnails']['high']['url']
            channel_id = snippet['channelId']
            channel_title = snippet['channelTitle']
            published_at = snippet['publishedAt']
            views = int(stats.get('viewCount', 0))
            likes = int(stats.get('likeCount', 0))
            comments = int(stats.get('commentCount', 0))
            language = snippet.get('defaultLanguage', 'en')
            return (title, thumbnail_url, channel_id, channel_title, published_at, 
                    views, likes, comments, language)
        else:
            raise ValueError(f"Video {video_id} not found")
    except Exception as e:
        raise ValueError(f"Failed to get video details: {e}")

def find_similar_videos_enhanced(input_url, max_results=150, top_similar=50, similarity_threshold=0.70):
    similarity_finder = UltraFastYouTubeSimilarity()
    try:
        print("🚀 Starting ENHANCED similarity analysis (Pure Thumbnail, No Shorts)...")
        start_time = time.time()
        input_video_id = extract_video_id(input_url)
        (input_title, input_thumbnail_url, channel_id, channel_title, 
         input_published_at, input_views, input_likes, input_comments, 
         input_language) = get_video_details(input_video_id)
        print(f"🎯 Analyzing: {input_title}")
        print(f"📺 Channel ID: {channel_id}")
        related_videos = similarity_finder._get_smart_related_videos_no_shorts(
            input_title, input_video_id, channel_id, max_results
        )
        if not related_videos:
            return None, None, "No related videos found for analysis."
        print(f"✅ Found {len(related_videos)} videos for analysis (guaranteed no Shorts)")
        print("🧠 Extracting visual features from input thumbnail...")
        input_features, _ = similarity_finder._extract_features_batch([input_thumbnail_url])
        if len(input_features) == 0:
            return None, None, "Failed to process input video thumbnail."
        print("⚡ Processing related video thumbnails with batch extraction...")
        related_thumbnail_urls = [video[2] for video in related_videos]
        related_features, valid_thumbnail_urls = similarity_finder._extract_features_batch(
            related_thumbnail_urls
        )
        if len(related_features) == 0:
            return None, None, "Failed to process any related video thumbnails."
        print(f"🎯 Successfully processed {len(related_features)} thumbnails")
        print("🧮 Computing pure visual similarity scores...")
        similarities = similarity_finder._compute_pure_visual_similarity(
            input_features[0], related_features
        )
        print("📊 Ranking results by visual similarity only...")
        url_to_video = {video[2]: video for video in related_videos}
        subscriber_count, channel_avg_views = similarity_finder._get_channel_details(channel_id)
        similar_videos_data = []
        for i, (thumbnail_url, similarity_score) in enumerate(zip(valid_thumbnail_urls, similarities)):
            if similarity_score >= similarity_threshold and thumbnail_url in url_to_video:
                video_data = url_to_video[thumbnail_url]
                video_id = video_data[0]
                details = video_data[3]
                video_subscriber_count, video_channel_avg_views = similarity_finder._get_channel_details(
                    details['channel_id']
                )
                views = details['views']
                likes = details['likes']
                comments = details['comments']
                total_engagement = likes + comments
                engagement_rate = total_engagement / views if views > 0 else 0
                multiplier = views / video_channel_avg_views if video_channel_avg_views > 0 else 1
                viral_score = (similarity_score * 0.4 + engagement_rate * 0.3 + multiplier * 0.3) * 100
                similar_videos_data.append({
                    'video_id': video_id,
                    'title': video_data[1],
                    'channel_id': details['channel_id'],
                    'channel_title': details['channel_title'],
                    'views': views,
                    'views_formatted': outlier_detector.format_number(views),
                    'channel_avg_views': video_channel_avg_views,
                    'channel_avg_views_formatted': outlier_detector.format_number(video_channel_avg_views),
                    'multiplier': round(multiplier, 2),
                    'likes': likes,
                    'likes_formatted': outlier_detector.format_number(likes),
                    'comments': comments,
                    'comments_formatted': outlier_detector.format_number(comments),
                    'duration': details['duration'],
                    'duration_seconds': details['duration_seconds'],
                    'url': f"https://www.youtube.com/watch?v={video_id}",
                    'published_at': details['published_at'],
                    'viral_score': round(viral_score, 2),
                    'engagement_rate': round(engagement_rate, 4),
                    'thumbnail_url': thumbnail_url,
                    'subscriber_count': video_subscriber_count,
                    'language': details['language']
                })
        similar_videos_data.sort(key=lambda x: x['viral_score'], reverse=True)
        similar_videos_data = similar_videos_data[:top_similar]
        input_engagement = input_likes + input_comments
        input_engagement_rate = input_engagement / input_views if input_views > 0 else 0
        input_multiplier = input_views / channel_avg_views if channel_avg_views > 0 else 1
        input_viral_score = (input_engagement_rate * 0.5 + input_multiplier * 0.5) * 100
        result_data = {
            'input_video': {
                'video_id': input_video_id,
                'title': input_title,
                'channel_id': channel_id,
                'channel_title': channel_title,
                'views': input_views,
                'views_formatted': outlier_detector.format_number(input_views),
                'channel_avg_views': channel_avg_views,
                'channel_avg_views_formatted': outlier_detector.format_number(channel_avg_views),
                'multiplier': round(input_multiplier, 2),
                'likes': input_likes,
                'likes_formatted': outlier_detector.format_number(input_likes),
                'comments': input_comments,
                'comments_formatted': outlier_detector.format_number(input_comments),
                'duration': 'PT0S',  # Placeholder
                'duration_seconds': 0,
                'url': input_url,
                'published_at': input_published_at,
                'viral_score': round(input_viral_score, 2),
                'engagement_rate': round(input_engagement_rate, 4),
                'thumbnail_url': input_thumbnail_url,
                'subscriber_count': subscriber_count,
                'language': input_language
            },
            'similar_videos': similar_videos_data,
            'processing_stats': {
                'total_videos_analyzed': len(related_videos),
                'thumbnails_processed': len(related_features),
                'processing_time': round(time.time() - start_time, 2),
                'average_visual_similarity': round(np.mean([v['viral_score'] for v in similar_videos_data]), 3) if similar_videos_data else 0,
                'shorts_strictly_filtered': True,
                'thumbnail_only_analysis': True,
                'channel_priority': True
            }
        }
        processing_time = time.time() - start_time
        print(f"🎉 ENHANCED analysis complete!")
        print(f"📊 Found {len(similar_videos_data)} visually similar videos (no Shorts, thumbnail-only)")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"🚀 Speed: {len(related_features)/processing_time:.1f} thumbnails/second")
        if similar_videos_data:
            avg_similarity = np.mean([v['viral_score'] for v in similar_videos_data])
            print(f"🎯 Average viral score: {avg_similarity:.3f}")
        return result_data, input_title, None
    except Exception as e:
        print(f"❌ Error in enhanced analysis: {str(e)}")
        return None, None, f"Enhanced analysis failed: {str(e)}"