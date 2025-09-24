import re
import os
from youtube_transcript_api import YouTubeTranscriptApi
from werkzeug.utils import secure_filename
import docx
import PyPDF2
import google.generativeai as genai

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def extract_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    regex = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(regex, url)
    return match.group(1) if match else None

def fetch_transcript(video_id: str) -> str:
    """Fetch transcript text from a YouTube video"""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join([t["text"] for t in transcript])
    except Exception:
        return ""

def extract_text_from_file(file):
    """Extract text from txt, pdf, docx"""
    filename = secure_filename(file.filename)
    ext = filename.split(".")[-1].lower()

    if ext == "txt":
        return file.read().decode("utf-8", errors="ignore")

    elif ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    elif ext == "docx":
        doc = docx.Document(file)
        return "\n".join([para.text for para in doc.paragraphs])

    return ""

def generate_script(context: dict) -> str:
    """Generate script using Gemini"""
    system_prompt = f"""You are a professional YouTube script writer.
    - Use the personal video transcripts to mimic writing style.
    - Use the inspiration video transcripts for tone & structure.
    - Use the docs for factual accuracy.
    - The final script MUST be about {context['minutes']} minutes long
      (around {context['target_words']} words).
    - Respond in plain text only (no JSON, no markdown, no code blocks)."""

    user_content = f"""
    User Prompt: {context['user_prompt']}

    Personal Video Styles:
    {" ".join(context['personal_samples'])[:4000]}

    Inspiration Video Styles:
    {" ".join(context['inspiration_samples'])[:4000]}

    Reference Docs:
    {" ".join(context['docs_samples'])[:4000]}
    """

    client = genai.GenerativeModel('gemini-2.0-flash')
    response = client.generate_content(f"{system_prompt}\n\n{user_content}")

    return response.text.strip() if response and response.text else "No script generated."

