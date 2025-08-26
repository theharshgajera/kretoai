import random
import os
from datetime import datetime, timedelta
import googleapiclient.discovery
from isodate import parse_duration
from flask import current_app
from dotenv import load_dotenv

load_dotenv()

# Replace with your actual YouTube API key
YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY', 'your_youtube_api_key')

def get_video_details_batch(video_ids):
    """Fetch video details for a list of video IDs."""
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.videos().list(
            part="contentDetails,snippet,statistics",
            id=",".join(video_ids)
        )
        response = request.execute()
        details = {}
        for item in response.get('items', []):
            video_id = item['id']
            details[video_id] = {
                'views': int(item['statistics'].get('viewCount', 0)),
                'likes': int(item['statistics'].get('likeCount', 0)),
                'comments': int(item['statistics'].get('commentCount', 0)),
                'duration': item['contentDetails']['duration']
            }
        return details
    except Exception as e:
        current_app.logger.error(f"Error fetching video details: {str(e)}")
        return {}

def search_videos(query, max_results=50, region_code='US', published_after=None, order='viewCount'):
    """Search YouTube videos using the API with randomization."""
    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        # Randomize order occasionally to get varied results
        possible_orders = ['viewCount', 'relevance', 'date']
        selected_order = random.choice(possible_orders) if random.random() < 0.5 else order
        request = youtube.search().list(
            part='id,snippet',
            q=query,
            type='video',
            maxResults=min(max_results, 50),
            order=selected_order,
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
        # Shuffle videos to introduce randomness
        random.shuffle(videos)
        current_app.logger.debug(f"Fetched {len(videos)} videos for query: {query}, region: {region_code}, order: {selected_order}")
        return videos
    except Exception as e:
        current_app.logger.error(f"YouTube search error: {str(e)}")
        return []

def get_viral_thumbnails(query):
    """Fetch 10 viral long-video thumbnail URLs from YouTube with randomization."""
    if YOUTUBE_API_KEY == 'YOUR_YOUTUBE_API_KEY':
        return {'error': 'YouTube API key not configured'}, 500

    youtube = googleapiclient.discovery.build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    long_videos = []
    next_page_token = None
    regions = ['US', 'GB', 'IN', 'CA', 'AU', 'DE', 'FR']
    region_code = random.choice(regions)  # Randomize region
    days_back = random.randint(7, 30)  # Randomize time range (7-30 days)
    published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat("T") + "Z"
    max_attempts = 3
    attempt = 0

    while len(long_videos) < 10 and attempt < max_attempts:
        # Search for a batch of videos with randomized parameters
        search_request = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            order=random.choice(['viewCount', 'relevance', 'date']),  # Randomize order
            maxResults=50,
            pageToken=next_page_token,
            regionCode=region_code,
            publishedAfter=published_after
        )
        search_response = search_request.execute()

        if not search_response.get('items'):
            current_app.logger.debug("No more results to process")
            break

        video_ids = [item['id']['videoId'] for item in search_response['items']]

        # Get content details (including duration) for the batch of videos
        videos_request = youtube.videos().list(
            part="contentDetails,snippet",
            id=",".join(video_ids)
        )
        videos_response = videos_request.execute()

        # Filter for videos longer than 180 seconds
        for item in videos_response.get('items', []):
            duration_iso = item['contentDetails']['duration']
            try:
                duration_seconds = parse_duration(duration_iso).total_seconds()
            except Exception as e:
                current_app.logger.error(f"Failed to parse duration '{duration_iso}': {e}")
                duration_seconds = 0

            if duration_seconds > 180:
                long_videos.append({
                    'title': item['snippet']['title'],
                    'thumbnail_url': item['snippet']['thumbnails']['high']['url']
                })

        # Randomly select up to 10 videos
        random.shuffle(long_videos)
        long_videos = long_videos[:10]

        next_page_token = search_response.get('nextPageToken')
        if not next_page_token:
            current_app.logger.debug("No next page available")
            break

        attempt += 1
        # Adjust parameters for next attempt
        region_code = random.choice(regions)
        days_back = random.randint(7, 30)
        published_after = (datetime.utcnow() - timedelta(days=days_back)).isoformat("T") + "Z"

    if not long_videos:
        current_app.logger.error(f"No long videos found for query: {query}")
        return {'error': 'No long videos found for the query'}, 404

    long_videos = long_videos[:10]
    current_app.logger.info(f"Found {len(long_videos)} viral thumbnails for query: {query}")
    return {
        'success': True,
        'query': query,
        'results_count': len(long_videos),
        'thumbnails': long_videos
    }