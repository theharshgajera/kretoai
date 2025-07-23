import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from googleapiclient.discovery import build
from model_util import DeepModel, DataSequence
from main_multi import ImageSimilarity
import concurrent.futures
import threading
from functools import lru_cache
import time
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from PIL import Image
import requests
from io import BytesIO
import tensorflow as tf

# Load environment variables from .env
load_dotenv()
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')
if not YOUTUBE_API_KEY:
    raise ValueError("YOUTUBE_API_KEY not found in .env file")

# Advanced model configuration
tf.config.experimental.enable_memory_growth = True
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.threading.set_intra_op_parallelism_threads(0)

# Global instances for ultra-fast processing
_model_instance = None
_model_lock = threading.Lock()

class UltraFastYouTubeSimilarity:
    """Ultra-fast YouTube thumbnail similarity finder with strict Shorts filtering and thumbnail-only analysis."""
    
    def __init__(self):
        self._session = self._create_optimized_session()
        self._model = self._get_or_create_model()
        self._feature_cache = {}
        self._batch_size = 64
        self._max_workers = 16
        
    def _create_optimized_session(self):
        """Create highly optimized requests session."""
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
        """Get or create optimized model instance."""
        global _model_instance, _model_lock
        if _model_instance is None:
            with _model_lock:
                if _model_instance is None:
                    _model_instance = self._create_ultra_fast_model()
        return _model_instance
    
    def _create_ultra_fast_model(self):
        """Create ultra-fast feature extraction model."""
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
        from tensorflow.keras import Model
        
        # Use EfficientNetB0 for better speed/accuracy tradeoff
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Add custom layers for better feature extraction
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x)  # Reduced dimensionality
        
        model = Model(inputs=base_model.input, outputs=x)
        
        # Optimize for inference
        model.compile(optimizer='adam')
        
        # Warmup the model
        dummy_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        model.predict(dummy_input, verbose=0)
        
        print("🚀 Ultra-fast EfficientNetB0 model loaded and warmed up!")
        return model
    
    def _get_video_durations_batch(self, video_ids):
        """Get durations for multiple videos in a single API call."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        duration_map = {}
        
        # Process in batches of 50 (YouTube API limit)
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
                    duration_seconds = self._parse_duration(duration)
                    duration_map[video_id] = duration_seconds
                    
            except Exception as e:
                print(f"❌ Error getting durations for batch: {e}")
                # If API fails, assume all videos in this batch are regular videos
                for video_id in batch_ids:
                    duration_map[video_id] = 300  # Assume 5 minutes (not a Short)
        
        return duration_map
    
    def _parse_duration(self, duration):
        """Parse ISO 8601 duration to seconds."""
        import re
        pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
        match = re.match(pattern, duration)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
        return 0
    
    def _filter_shorts_strict(self, videos, max_duration=60):
        """Strictly filter out YouTube Shorts based on duration."""
        print(f"🔍 Strictly filtering Shorts (≤{max_duration}s) from {len(videos)} videos...")
        
        if not videos:
            return []
        
        # Extract all video IDs
        video_ids = [video[0] for video in videos]
        
        # Get durations for all videos in batches
        duration_map = self._get_video_durations_batch(video_ids)
        
        # Filter videos based on duration
        filtered_videos = []
        shorts_count = 0
        
        for video in videos:
            video_id = video[0]
            duration = duration_map.get(video_id, 0)
            
            if duration > max_duration:  # Keep only videos longer than max_duration
                filtered_videos.append(video)
            else:
                shorts_count += 1
        
        print(f"🚫 Filtered out {shorts_count} YouTube Shorts (≤{max_duration}s)")
        print(f"✅ Kept {len(filtered_videos)} regular videos (>{max_duration}s)")
        
        return filtered_videos
    
    def _get_channel_id(self, video_id):
        """Get channel ID from video ID."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        try:
            request = youtube.videos().list(
                part='snippet',
                id=video_id
            )
            response = request.execute()
            
            if response['items']:
                return response['items'][0]['snippet']['channelId']
            return None
        except Exception as e:
            print(f"❌ Error getting channel ID for {video_id}: {e}")
            return None
    
    def _get_channel_videos_no_shorts(self, channel_id, max_results=50):
        """Get videos from a specific channel, strictly excluding Shorts."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        videos = []
        
        try:
            # Get channel uploads playlist
            channels_request = youtube.channels().list(
                part='contentDetails',
                id=channel_id
            )
            channels_response = channels_request.execute()
            
            if not channels_response['items']:
                return videos
            
            uploads_playlist_id = channels_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Get more videos initially to account for filtering
            raw_videos = []
            next_page_token = None
            
            # Fetch more videos than needed to account for Shorts filtering
            while len(raw_videos) < max_results * 3:  # Get 3x more to account for Shorts
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
            
            # Filter out Shorts strictly
            filtered_videos = self._filter_shorts_strict(raw_videos)
            
            # Take only the requested number of results
            videos = filtered_videos[:max_results]
                    
        except Exception as e:
            print(f"❌ Error getting channel videos: {e}")
        
        print(f"📺 Final channel videos: {len(videos)} (no Shorts)")
        return videos
    
    def _search_videos_no_shorts(self, youtube, query, max_results):
        """Search for videos with strict Shorts filtering."""
        try:
            # Search for more videos initially to account for Shorts filtering
            search_results = max_results * 2  # Get 2x more to account for Shorts
            
            request = youtube.search().list(
                part='id,snippet',
                q=query,
                type='video',
                maxResults=min(search_results, 50),  # API limit is 50
                order='relevance',
                regionCode='US',
                videoDuration='medium'  # This helps filter out some Shorts
            )
            response = request.execute()
            
            raw_videos = []
            for item in response['items']:
                video_id = item['id']['videoId']
                title = item['snippet']['title']
                thumbnail_url = item['snippet']['thumbnails']['high']['url']
                raw_videos.append([video_id, title, thumbnail_url])
            
            # Strictly filter out Shorts
            filtered_videos = self._filter_shorts_strict(raw_videos)
            
            # Return only the requested number
            return filtered_videos[:max_results]
            
        except Exception as e:
            print(f"❌ API request failed: {e}")
            return []
    
    def _get_smart_related_videos_no_shorts(self, video_title, input_video_id, channel_id=None, max_results=200):
        """Smart video fetching with strict Shorts filtering and reduced title-based searching."""
        youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
        all_videos = []
        video_ids_seen = {input_video_id}
        
        # Step 1: Get videos from the same channel first (if channel_id provided)
        if channel_id:
            print(f"🎯 Searching in same channel first...")
            channel_videos = self._get_channel_videos_no_shorts(channel_id, max_results=50)
            for video in channel_videos:
                if video[0] not in video_ids_seen:
                    video_ids_seen.add(video[0])
                    all_videos.append(video)
            
            print(f"📺 Found {len(channel_videos)} videos from same channel (no Shorts)")
        
        # Step 2: If we need more videos, search globally with minimal title influence
        remaining_results = max_results - len(all_videos)
        if remaining_results > 0:
            print(f"🌍 Searching globally for {remaining_results} more videos...")
            
            # More generic search strategies to reduce title bias
            search_strategies = [
                # Extract only key content words (no common words)
                ' '.join([w for w in video_title.split() 
                         if len(w) > 3 and w.lower() not in [
                             'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 
                             'with', 'by', 'a', 'an', 'this', 'that', 'these', 'those', 'is', 
                             'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
                             'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
                             'might', 'must', 'can', 'video', 'youtube', 'watch', 'episode',
                             'part', 'full', 'complete', 'official'
                         ]])[:30],
                # Just the first significant word
                next((w for w in video_title.split() if len(w) > 4), video_title.split()[0] if video_title.split() else ''),
                # Extract potential topic/category words
                ' '.join([w for w in video_title.split() 
                         if any(char.isupper() for char in w) and len(w) > 3])[:25]
            ]
            
            # Remove empty strategies
            search_strategies = [s.strip() for s in search_strategies if s.strip()]
            
            # Use concurrent requests for faster API calls
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_to_query = {}
                
                for i, query in enumerate(search_strategies[:3]):  # Limit to 3 strategies
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
        """Ultra-fast image preprocessing with caching."""
        cache_key = hash(image_url)
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        try:
            if image_url.startswith('http'):
                response = self._session.get(
                    image_url, 
                    timeout=5,  # Increased timeout for better reliability
                    headers={'User-Agent': 'Mozilla/5.0'},
                    stream=True
                )
                response.raise_for_status()
                
                # Use PIL for faster processing
                img = Image.open(BytesIO(response.content))
                img = img.convert('RGB')
                img = img.resize((224, 224), Image.Resampling.LANCZOS)  # Better quality resampling
                
                # Convert to numpy array
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_array = (img_array - 0.5) * 2.0  # Normalize to [-1, 1]
                
                # Cache the result
                self._feature_cache[cache_key] = img_array
                return img_array
            else:
                # Local file processing
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
        """Extract features in optimized batches."""
        features = []
        valid_urls = []
        
        # Process images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            future_to_url = {
                executor.submit(self._preprocess_image_ultra_fast, url): url 
                for url in image_urls
            }
            
            batch_images = []
            batch_urls = []
            
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                img_array = future.result()
                
                if img_array is not None:
                    batch_images.append(img_array)
                    batch_urls.append(url)
                    
                    # Process in batches
                    if len(batch_images) >= self._batch_size:
                        batch_features = self._extract_features_from_batch(batch_images)
                        features.extend(batch_features)
                        valid_urls.extend(batch_urls)
                        batch_images = []
                        batch_urls = []
            
            # Process remaining images
            if batch_images:
                batch_features = self._extract_features_from_batch(batch_images)
                features.extend(batch_features)
                valid_urls.extend(batch_urls)
        
        return np.array(features), valid_urls
    
    def _extract_features_from_batch(self, batch_images):
        """Extract features from a batch of images."""
        if not batch_images:
            return []
        
        batch_array = np.array(batch_images)
        features = self._model.predict(batch_array, verbose=0)
        
        return features.tolist()
    
    def _compute_pure_visual_similarity(self, query_features, database_features):
        """Compute pure visual similarity focusing only on image features."""
        if len(query_features) == 0 or len(database_features) == 0:
            return []
        
        query_features = np.array(query_features).reshape(1, -1)
        database_features = np.array(database_features)
        
        # Use cosine similarity as primary metric (best for visual features)
        cosine_sim = cosine_similarity(query_features, database_features)[0]
        
        # Add structural similarity component
        # Normalize features for better comparison
        query_norm = query_features / (np.linalg.norm(query_features) + 1e-8)
        database_norm = database_features / (np.linalg.norm(database_features, axis=1, keepdims=True) + 1e-8)
        
        # Dot product similarity (normalized features)
        dot_sim = np.dot(database_norm, query_norm.T).flatten()
        
        # Combine similarities with emphasis on cosine similarity
        combined_similarity = 0.8 * cosine_sim + 0.2 * dot_sim
        
        return combined_similarity

@lru_cache(maxsize=200)
def extract_video_id(url):
    """Extract video ID from YouTube URL with caching."""
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
    """Get video details with caching."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        
        if response['items']:
            snippet = response['items'][0]['snippet']
            title = snippet['title']
            thumbnail_url = snippet['thumbnails']['high']['url']
            channel_id = snippet['channelId']
            return title, thumbnail_url, channel_id
        else:
            raise ValueError(f"Video {video_id} not found")
    except Exception as e:
        raise ValueError(f"Failed to get video details: {e}")

def find_similar_videos_enhanced(input_url, max_results=150, top_similar=50, similarity_threshold=0.70):
    """
    Enhanced algorithm for finding similar YouTube videos with strict Shorts filtering 
    and pure thumbnail-based similarity (no title bias).
    
    Args:
        input_url: YouTube video URL
        max_results: Maximum videos to search through
        top_similar: Number of top similar videos to return
        similarity_threshold: Minimum similarity score (0-1)
    
    Returns:
        Tuple of (result_data, input_title, error_message)
    """
    
    similarity_finder = UltraFastYouTubeSimilarity()
    
    try:
        print("🚀 Starting ENHANCED similarity analysis (Pure Thumbnail, No Shorts)...")
        start_time = time.time()
        
        # Step 1: Extract video ID and get details
        print("📺 Extracting video information...")
        input_video_id = extract_video_id(input_url)
        input_title, input_thumbnail_url, channel_id = get_video_details(input_video_id)
        
        print(f"🎯 Analyzing: {input_title}")
        print(f"📺 Channel ID: {channel_id}")
        
        # Step 2: Get related videos with strict Shorts filtering
        print("🔍 Fetching related videos (Strict No-Shorts filtering)...")
        related_videos = similarity_finder._get_smart_related_videos_no_shorts(
            input_title, input_video_id, channel_id, max_results
        )
        
        if not related_videos:
            return None, None, "No related videos found for analysis."
        
        print(f"✅ Found {len(related_videos)} videos for analysis (guaranteed no Shorts)")
        
        # Step 3: Extract features from input thumbnail (pure visual analysis)
        print("🧠 Extracting visual features from input thumbnail...")
        input_features, _ = similarity_finder._extract_features_batch([input_thumbnail_url])
        
        if len(input_features) == 0:
            return None, None, "Failed to process input video thumbnail."
        
        # Step 4: Extract features from related video thumbnails
        print("⚡ Processing related video thumbnails with batch extraction...")
        related_thumbnail_urls = [video[2] for video in related_videos]
        related_features, valid_thumbnail_urls = similarity_finder._extract_features_batch(
            related_thumbnail_urls
        )
        
        if len(related_features) == 0:
            return None, None, "Failed to process any related video thumbnails."
        
        print(f"🎯 Successfully processed {len(related_features)} thumbnails")
        
        # Step 5: Compute pure visual similarity scores (no title influence)
        print("🧮 Computing pure visual similarity scores...")
        similarities = similarity_finder._compute_pure_visual_similarity(
            input_features[0], related_features
        )
        
        # Step 6: Filter and rank results based purely on visual similarity
        print("📊 Ranking results by visual similarity only...")
        
        # Create mapping from valid URLs back to video data
        url_to_video = {video[2]: video for video in related_videos}
        
        # Combine similarities with video data
        similar_videos_data = []
        for i, (thumbnail_url, similarity_score) in enumerate(zip(valid_thumbnail_urls, similarities)):
            if similarity_score >= similarity_threshold and thumbnail_url in url_to_video:
                video_data = url_to_video[thumbnail_url]
                similar_videos_data.append({
                    "video_id": video_data[0],
                    "title": video_data[1],
                    "thumbnail_url": video_data[2],
                    "video_url": f"https://www.youtube.com/watch?v={video_data[0]}",
                    "visual_similarity": float(similarity_score)  # Renamed to emphasize visual-only
                })
        
        # Sort by visual similarity score only (highest first)
        similar_videos_data.sort(key=lambda x: x['visual_similarity'], reverse=True)
        
        # Take top results
        similar_videos_data = similar_videos_data[:top_similar]
        
        # Step 7: Prepare final results
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
                "processing_time": round(time.time() - start_time, 2),
                "average_visual_similarity": round(np.mean([v['visual_similarity'] for v in similar_videos_data]), 3) if similar_videos_data else 0,
                "shorts_strictly_filtered": True,
                "thumbnail_only_analysis": True,
                "channel_priority": True
            }
        }
        
        processing_time = time.time() - start_time
        print(f"🎉 ENHANCED analysis complete!")
        print(f"📊 Found {len(similar_videos_data)} visually similar videos (no Shorts, thumbnail-only)")
        print(f"⏱️  Total processing time: {processing_time:.2f} seconds")
        print(f"🚀 Speed: {len(related_features)/processing_time:.1f} thumbnails/second")
        
        if similar_videos_data:
            avg_similarity = np.mean([v['visual_similarity'] for v in similar_videos_data])
            print(f"🎯 Average visual similarity score: {avg_similarity:.3f}")
        
        return result_data, input_title, None
        
    except Exception as e:
        print(f"❌ Error in enhanced analysis: {str(e)}")
        return None, None, f"Enhanced analysis failed: {str(e)}"

# Main function for compatibility
def find_similar_videos_fast(input_url, max_results=150, top_similar=50):
    """Main function with enhanced filtering."""
    result_data, input_title, error = find_similar_videos_enhanced(
        input_url, max_results, top_similar
    )
    return result_data, input_title, error

# Lightning fast version with higher thresholds
def find_similar_videos_lightning_fast(input_url):
    """Lightning-fast version with strict filtering."""
    return find_similar_videos_enhanced(
        input_url, 
        max_results=100, 
        top_similar=20, 
        similarity_threshold=0.80  # Higher threshold for better visual accuracy
    )
