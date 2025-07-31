import os
import re
from googleapiclient.discovery import build
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
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

def get_video_info(video_id):
    """Fetches title and description for a YouTube video ID."""
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    try:
        request = youtube.videos().list(part='snippet', id=video_id)
        response = request.execute()
        if response['items']:
            snippet = response['items'][0]['snippet']
            return snippet['title'], snippet['description']
        raise ValueError(f"Video {video_id} not found")
    except Exception as e:
        raise ValueError(f"Failed to get video details: {str(e)}")

def generate_script(text, duration, style_links, content_links, wpm=145, creator_name="YourChannelName", audience="beginners", language="en"):
    """Generates a professional video script tailored for a YouTube creator in the specified language."""
    # Calculate target word count
    target_word_count = int(duration * wpm)

    # Fetch style video info
    style_info = []
    for link in style_links:
        video_id = extract_video_id(link)
        title, description = get_video_info(video_id)
        style_info.append(f"Title: {title}\nDescription: {description}")

    # Fetch content video info
    content_info = []
    for link in content_links:
        video_id = extract_video_id(link)
        title, description = get_video_info(video_id)
        content_info.append(f"Title: {title}\nDescription: {description}")

    # Construct detailed prompt for a pro content writer with language specification
    style_text = "\n\n".join(style_info)
    content_text = "\n\n".join(content_info)
    prompt = (
        f"You are a professional content writer crafting a video script for a YouTube creator named {creator_name}, "
        f"targeting an audience of {audience}. The script is based on the initial text: '{text}'. "
        f"It should be approximately {target_word_count} words long, suitable for a {duration}-minute video at a speaking rate of {wpm} words per minute. "
        f"Generate the script in {language} language, reflecting the creator's unique style, tone, and personality as seen in the following videos:\n{style_text}\n\n"
        f"Draw content inspiration and context from these related videos:\n{content_text}\n\n"
        f"Incorporate YouTube SEO best practices, including keyword-rich phrases related to '{text}' in {language}, a strong hook within the first 10 seconds, "
        f"calls-to-action (e.g., 'like', 'subscribe', 'comment' translated to {language}), and timestamps for sections. Structure the script with an introduction, main content, "
        f"and conclusion. Use engaging storytelling, relatable examples, and a conversational tone to retain viewer interest. Ensure the script is optimized "
        f"for watch time and encourages engagement in {language}."
    )

    # Call Gemini API with consistent style
    model = genai.GenerativeModel('gemini-2.5-flash')  # Updated model name
    response = model.generate_content(prompt)
    script = response.text

    return script
def generate_script_from_title(title, duration, wpm=145, creator_name="YourChannelName", audience="beginners", language="en"):
    """Generate a professional video script based on the provided title and duration. Provide ONLY the raw script content with no introductory text, explanations, or formatting notes. The output should begin immediately with the script's first line of dialogue or action. Do not include any meta-commentary, section headers, or production notes - only the pure script content that would be used for filming."""
    # Use empty lists for style and content links to rely solely on the title
    style_links = []
    content_links = []
    
    # Call the existing generate_script function with default parameters
    script = generate_script(
        text=title,
        duration=duration,
        style_links=style_links,
        content_links=content_links,
        wpm=wpm,
        creator_name=creator_name,
        audience=audience,
        language=language
    )
    
    return script