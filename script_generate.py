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

def generate_script(data):
    """Generates a professional video script tailored for a YouTube creator using the standardized JSON format with groups."""
    # Extract required fields with defaults
    text = data.get("text", "")
    duration = float(data.get("duration", 5))  # Default to 5 minutes if not provided
    groups = data.get("groups", [])
    wpm = 145  # Default words per minute
    creator_name = "YourChannelName"  # Default creator name
    audience = "beginners"  # Default audience
    language = "en"  # Default language

    # Calculate target word count
    target_word_count = int(duration * wpm)

    # Parse text to extract group usage instructions
    style_group = None
    content_group = None
    text_content = text
    if "use group" in text.lower():
        instructions = re.findall(r"use group '([^']+)' for (\w+)", text, re.IGNORECASE)
        for group_name, purpose in instructions:
            if purpose.lower() == "style":
                style_group = group_name
            elif purpose.lower() == "content":
                content_group = group_name
        text_content = re.sub(r"use group '[^']+' for \w+", "", text).strip()

    # Fetch style video info from the specified style group
    style_info = []
    if style_group:
        for group in groups:
            if group.get("name") == style_group:
                for video in group.get("videos", []):
                    link = video.get("link", "")
                    if link:
                        try:
                            video_id = extract_video_id(link)
                            title, description = get_video_info(video_id)
                            style_info.append(f"Title: {title}\nDescription: {description}")
                        except ValueError as e:
                            print(f"Warning: Invalid style video link {link}: {str(e)}")

    # Fetch content video info from the specified content group
    content_info = []
    if content_group:
        for group in groups:
            if group.get("name") == content_group:
                for video in group.get("videos", []):
                    link = video.get("link", "")
                    if link:
                        try:
                            video_id = extract_video_id(link)
                            title, description = get_video_info(video_id)
                            content_info.append(f"Title: {title}\nDescription: {description}")
                        except ValueError as e:
                            print(f"Warning: Invalid content video link {link}: {str(e)}")

    # Use default groups if no specific instructions are provided
    if not style_info and groups:
        style_info = [f"Title: No specific style group provided\nDescription: Using default style"]
        for group in groups[:1]:  # Use the first group as default style
            for video in group.get("videos", []):
                link = video.get("link", "")
                if link:
                    try:
                        video_id = extract_video_id(link)
                        title, description = get_video_info(video_id)
                        style_info[0] = f"Title: {title}\nDescription: {description}"
                        break
                    except ValueError as e:
                        print(f"Warning: Invalid default style video link {link}: {str(e)}")
    if not content_info and groups:
        content_info = [f"Title: No specific content group provided\nDescription: Using default content"]
        for group in groups[1:2] or groups[:1]:  # Use the second group or first as default content
            for video in group.get("videos", []):
                link = video.get("link", "")
                if link:
                    try:
                        video_id = extract_video_id(link)
                        title, description = get_video_info(video_id)
                        content_info[0] = f"Title: {title}\nDescription: {description}"
                        break
                    except ValueError as e:
                        print(f"Warning: Invalid default content video link {link}: {str(e)}")

    # Construct detailed prompt for a pro content writer with language specification
    style_text = "\n\n".join(style_info) if style_info else "No style videos provided"
    content_text = "\n\n".join(content_info) if content_info else "No content videos provided"
    prompt = (
        f"You are a professional content writer crafting a video script for a YouTube creator named {creator_name}, "
        f"targeting an audience of {audience}. The script is based on the initial text: '{text_content}'. "
        f"It should be approximately {target_word_count} words long, suitable for a {duration}-minute video at a speaking rate of {wpm} words per minute. "
        f"Generate the script in {language} language, reflecting the creator's unique style, tone, and personality as seen in the following videos:\n{style_text}\n\n"
        f"Draw content inspiration and context from these related videos:\n{content_text}\n\n"
        f"Incorporate YouTube SEO best practices, including keyword-rich phrases related to '{text_content}' in {language}, a strong hook within the first 10 seconds, "
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
    # Prepare data in the standardized JSON format
    data = {
        "text": title,
        "duration": duration,
        "groups": []
    }
    # Call the generate_script function with the standardized data
    script = generate_script(data)
    return script