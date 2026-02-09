from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import requests
from google import genai
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from moviepy.config import change_settings
import shutil
from google.genai import types, Client
from google.genai.types import HttpOptions, Part

import os
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import time
import re
from pytubefix import YouTube
import whisper_timestamped as whisper
import uuid
import json
from datetime import datetime
import logging
from urllib.parse import urlparse, parse_qs
import threading
from collections import defaultdict
import subprocess
import speech_recognition as sr
import moviepy.editor as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import base64
# Document processing imports
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import tempfile
import fitz  # PyMuPDF for better PDF handling
from pathlib import Path

# ADD ANTHROPIC IMPORT
import anthropic

# Load environment variables
load_dotenv()
client = genai.Client()
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found. Check your .env file!")
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ========================================
# CONFIGURATION (will be set by app.py)
# ========================================

# File upload configuration
UPLOAD_FOLDER = r'D:\poppy AI\kretoai\tempfolder'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx', 'mp3', 'wav', 'm4a', 'flac', 'ogg', 'opus', 'webm', 'wma', 'aac'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# PERFORMANCE CONFIGURATION
# ========================================

# Threading for parallel processing
MAX_WORKERS = min(5, (multiprocessing.cpu_count() or 1) + 2)

# Video processing optimization
QUICK_PROCESS_THRESHOLD = 300  # 5 minutes - single chunk
CHUNK_SIZE = 240  # 4-minute chunks for parallel processing
MAX_PARALLEL_CHUNKS = 3  # Process 3 chunks simultaneously

print(f"✓ Script module loaded")
print(f"✓ Performance config: {MAX_WORKERS} workers, {CHUNK_SIZE}s chunks")
print(f"✓ Claude API configured for script generation")

# ========================================
# USER DATA STORAGE
# ========================================

# Simplified in-memory storage - only what's needed for all-in-one approach
user_data = defaultdict(lambda: {
    'chat_sessions': {},
    'current_script': None
})

# ========================================
# WHISPER MODEL CONFIGURATION
# ========================================

# Custom model storage path (downloaded once, used forever)
WHISPER_MODEL_DIR = r"D:\poppy AI\kretoai\model"
WHISPER_MODEL_NAME = "base"  # or "base" for better accuracy

# Global cached model (loaded once per server restart)
_WHISPER_MODEL_CACHE = None

def load_whisper_model():
    """
    Load Whisper model from custom directory
    Model is loaded ONCE and reused for all transcriptions
    """
    global _WHISPER_MODEL_CACHE
    
    if _WHISPER_MODEL_CACHE is not None:
        return _WHISPER_MODEL_CACHE
    
    try:
        import whisper
        
        print(f"Loading Whisper model from: {WHISPER_MODEL_DIR}")
        
        # Check if model exists
        model_file = os.path.join(WHISPER_MODEL_DIR, f"{WHISPER_MODEL_NAME}.pt")
        if not os.path.exists(model_file):
            print(f"❌ ERROR: Model not found at {model_file}")
            print(f"   Please run: python download_whisper_model.py")
            return None
        
        # Load model from custom directory
        _WHISPER_MODEL_CACHE = whisper.load_model(
            WHISPER_MODEL_NAME,
            download_root=WHISPER_MODEL_DIR
        )
        
        print(f"✓ Whisper model loaded successfully from {WHISPER_MODEL_DIR}")
        return _WHISPER_MODEL_CACHE
        
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        print(f"❌ Failed to load Whisper model: {str(e)}")
        return None

print(f"✓ Whisper config: Model directory = {WHISPER_MODEL_DIR}")

"""
COMPLETE DocumentProcessor CLASS - READY TO USE
Copy this entire class and replace your existing DocumentProcessor class
"""

class DocumentProcessor:
    def __init__(self):
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        self.max_file_size = 32 * 1024 * 1024  # 32MB limit
        
    def process_document(self, url, filename=None):
        """
        Process document from URL using Claude API
        Args:
            url: Document URL (must start with http/https)
            filename: Optional filename for reference
        """
        print(f"\n{'='*60}")
        print(f"DOCUMENT PROCESSING (CLAUDE): {filename or url}")
        print(f"{'='*60}\n")
        
        try:
            # Validate URL
            if not url or not url.strip():
                return {
                    "error": "Empty URL provided",
                    "text": None,
                    "stats": None
                }
            
            if not url.startswith(('http://', 'https://')):
                return {
                    "error": f"Invalid URL format (must start with http/https): {url}",
                    "text": None,
                    "stats": None
                }
            
            # Download document
            print(f"Downloading document from URL: {url}")
            import requests
            response = requests.get(url, timeout=60, allow_redirects=True)
            
            if response.status_code != 200:
                return {
                    "error": f"Failed to download: HTTP {response.status_code}",
                    "text": None,
                    "stats": None
                }
            
            print(f"✓ Downloaded: {len(response.content):,} bytes ({len(response.content)/(1024*1024):.2f} MB)")
            
            # Validate size
            if len(response.content) > self.max_file_size:
                return {
                    "error": f"File too large ({len(response.content)/(1024*1024):.1f}MB). Max: 32MB",
                    "text": None,
                    "stats": None
                }
            
            # Save to temp file
            import tempfile
            file_extension = os.path.splitext(url.split('?')[0])[1] or '.pdf'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_file.write(response.content)
            temp_file.close()
            
            print(f"✓ Saved to temp: {temp_file.name}")
            
            # Extract text based on file type
            if file_extension == '.pdf':
                extracted_text = self._extract_pdf_text(temp_file.name)
            elif file_extension in ['.docx', '.doc']:
                extracted_text = self._extract_docx_text(temp_file.name)
            elif file_extension == '.txt':
                extracted_text = self._extract_txt_text(temp_file.name)
            else:
                os.remove(temp_file.name)
                return {
                    "error": f"Unsupported file type: {file_extension}",
                    "text": None,
                    "stats": None
                }
            
            # Cleanup temp file
            os.remove(temp_file.name)
            
            if not extracted_text:
                return {
                    "error": "Could not extract text from document",
                    "text": None,
                    "stats": None
                }
            
            print(f"✓ Text extracted: {len(extracted_text):,} characters")
            
            # Analyze with Claude
            print("Analyzing document with Claude...")
            analysis = self._analyze_with_claude(extracted_text, filename)
            
            if not analysis:
                return {
                    "error": "Claude analysis failed",
                    "text": None,
                    "stats": None
                }
            
            word_count = len(analysis.split())
            
            print(f"\n{'='*60}")
            print(f"✓ DOCUMENT PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Analysis: {len(analysis):,} chars, {word_count:,} words")
            print(f"{'='*60}\n")
            
            return {
                "error": None,
                "text": analysis,
                "stats": {
                    "word_count": word_count,
                    "char_count": len(analysis),
                    "source_type": "document",
                    "filename": filename or "Document"
                },
                "filename": filename or "Document"
            }
            
        except Exception as e:
            error_msg = f"Document processing failed: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            print(traceback.format_exc())
            return {"error": error_msg, "text": None, "stats": None}
    
    def _extract_pdf_text(self, pdf_path):
        """Extract text from PDF"""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_parts.append(text)
            
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            return None
    
    def _extract_docx_text(self, docx_path):
        """Extract text from DOCX"""
        try:
            import docx
            doc = docx.Document(docx_path)
            text_parts = [para.text for para in doc.paragraphs if para.text.strip()]
            return "\n\n".join(text_parts)
        except Exception as e:
            print(f"❌ DOCX extraction failed: {e}")
            return None
    
    def _extract_txt_text(self, txt_path):
        """Extract text from TXT"""
        try:
            with open(txt_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            print(f"❌ TXT reading failed: {e}")
            return None
    
    def _analyze_with_claude(self, document_text, filename):
        """Analyze document text using Claude API"""
        try:
            # Truncate if too long (Claude has token limits)
            max_chars = 100000
            if len(document_text) > max_chars:
                chunk_size = max_chars // 2
                document_text = (
                    f"{document_text[:chunk_size]}\n\n"
                    f"[...MIDDLE CONTENT OMITTED...]\n\n"
                    f"{document_text[-chunk_size:]}"
                )
            
            # ✅ FIXED: Proper prompt formatting without extra indentation
            prompt = f"""Analyze this document and extract key insights for YouTube content creation.

Extract:
1. Core concepts, arguments, and themes
2. Specific data, facts, and statistics
3. Technical details and processes
4. Actionable insights and best practices

**DOCUMENT: {filename or 'Untitled'}**

{document_text}

Provide a comprehensive knowledge base for script writing."""

            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            analysis = message.content[0].text.strip()
            
            print(f"✓ Claude analysis complete: {len(analysis):,} characters")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Claude API error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None
class VideoProcessor:
    """
    High-performance video processor using Gemini Native Video Analysis.
    Now unified: Both YouTube and Local files are processed via API upload 
    to avoid local FFMPEG dependencies and improve accuracy.
    """
    
    def __init__(self, client=None, api_key=None):
        """
        Initialize the VideoProcessor.
        :param client: An instance of google.genai.Client
        :param api_key: Optional API key to initialize a client if one isn't provided.
        """
        # Priority 1: Use provided client instance
        self.client = client 
        
        # Priority 2: Use provided API key to create a client
        if self.client is None and api_key:
            try:
                self.client = Client(api_key=api_key, http_options={'api_version': 'v1alpha'})
            except Exception as e:
                logger.error(f"Failed to initialize GenAI Client with provided API key: {e}")

        # Priority 3: Explicitly check environment variable loaded by load_dotenv()
        env_key = os.getenv("GEMINI_API_KEY")
        if self.client is None and env_key:
            try:
                self.client = Client(api_key=env_key, http_options={'api_version': 'v1alpha'})
            except Exception as e:
                logger.error(f"Failed to initialize GenAI Client from .env GEMINI_API_KEY: {e}")

        # Final Fallback: Attempt default initialization
        if self.client is None:
            try:
                self.client = Client(http_options={'api_version': 'v1alpha'})
            except Exception as e:
                logger.warning(f"Default client initialization failed. Please ensure GEMINI_API_KEY is in .env: {e}")

        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        self.max_video_size = 500 * 1024 * 1024  # 500MB
        self.max_duration = 3600  # 1 hour
        self.last_api_call = 0
        self.rate_limit_delay = 1.0

    def _normalize_path(self, path):
        """Helper to ensure paths are absolute and OS-compliant."""
        if not path:
            return path
        clean_path = str(path).strip().strip('"').strip("'")
        normalized = Path(clean_path).expanduser().resolve()
        return str(normalized)

    def process_video_content(self, source, source_type='youtube'):
        """Universal entry point for all video types using Cloud Processing"""
        try:
            # Critical Check: Ensure client exists before processing
            if self.client is None:
                return {
                    "error": "GenAI Client not initialized. Please ensure GEMINI_API_KEY is set in your .env file or passed to the constructor.", 
                    "transcript": None, 
                    "stats": None
                }

            if source_type == 'youtube':
                if not self.validate_youtube_url(source):
                    return {"error": "Invalid YouTube URL", "transcript": None, "stats": None}
                return self.extract_youtube_transcript_details(source)
            
            elif source_type == 'local':
                source = self._normalize_path(source)
                if not self.is_supported_video_format(source):
                    return {"error": f"Unsupported format: {Path(source).suffix}", "transcript": None, "stats": None}
                
                if not os.path.exists(source):
                    return {"error": f"File not found at: {source}", "transcript": None, "stats": None}
                
                filename = os.path.basename(source)
                return self.process_local_video_via_api(source, filename)
            
            return {"error": f"Unknown source type: {source_type}", "transcript": None, "stats": None}
        except Exception as e:
            logger.error(f"Top-level processing error: {str(e)}")
            return {"error": f"Processing Exception: {str(e)}", "transcript": None, "stats": None}

    def is_supported_video_format(self, filename):
        return Path(filename).suffix.lower() in self.supported_video_formats

    def process_local_video_via_api(self, video_path, filename):
        """
        DEDICATED LOCAL VIDEO PIPELINE:
        Uploads local video to Gemini File API and performs analysis.
        """
        print(f"\n{'='*60}\nUPLOADING LOCAL VIDEO TO GEMINI: {filename}\n{'='*60}")
        try:
            start_time = time.time()
            
            # 1. Create a v1 client for file operations
            import google.generativeai as genai
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            
            # Upload using standard genai library
            video_file = genai.upload_file(path=video_path, display_name=filename)
            
            # 2. Wait for processing
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name != "ACTIVE":
                return {"error": f"Video processing failed: {video_file.state.name}", "transcript": None}

            # 3. Analyze with standard model
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content([
                video_file,
                "Provide a detailed transcription and a comprehensive summary of this video content."
            ])
            
            transcript = response.text.strip()
            
            # 4. Cleanup
            try: genai.delete_file(video_file.name)
            except: pass
            
            process_time = time.time() - start_time
            stats = self._calculate_transcript_stats(transcript)
            stats.update({
                'filename': filename,
                'source_type': 'local_native_analysis',
                'processing_time': round(process_time, 1)
            })
            
            return {"error": None, "transcript": transcript, "stats": stats, "source": filename}
            
        except Exception as e:
            logger.error(f"Cloud-based local processing error: {str(e)}")
            return {"error": f"Local processing error: {str(e)}", "transcript": None}
    def _call_gemini_local_analysis(self, video_file):
        """
        Dedicated API call function for local files.
        Uses the uploaded file reference correctly.
        """
        try:
            analysis_prompt = "Provide a detailed transcription and a comprehensive summary of this video content."
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    {
                        "parts": [
                            {"file_data": {"file_uri": video_file.uri, "mime_type": video_file.mime_type}},
                            {"text": analysis_prompt}
                        ]
                    }
                ]
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini local analysis error: {str(e)}")
            raise Exception(f"from_file_data") from e

    def validate_youtube_url(self, url):
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
        try:
            parsed_url = urlparse(url)
            return any(domain in parsed_url.netloc for domain in youtube_domains)
        except: return False

    def extract_youtube_transcript_details(self, youtube_video_url):
        """
        DEDICATED YOUTUBE PIPELINE:
        Processes YouTube video via native URI support in Gemini.
        """
        print(f"\n{'='*60}\nPROCESSING YOUTUBE VIA GEMINI API: {youtube_video_url}\n{'='*60}")
        try:
            transcript = self._call_gemini_youtube_analysis(youtube_video_url)
            
            return {
                "error": None,
                "transcript": transcript, 
                "stats": {
                    'char_count': len(transcript),
                    'word_count': len(transcript.split()),
                    'source_type': 'gemini_native_video_analysis',
                    'url': youtube_video_url
                }
            }
        except Exception as e:
            logger.error(f"YouTube processing error: {str(e)}")
            return {"error": f"Video processing failed: {str(e)}", "transcript": None, "stats": None}

    def _call_gemini_youtube_analysis(self, youtube_url):
        """Dedicated API call function for YouTube URLs."""
        summary_prompt = "Generate a detailed transcript and a comprehensive 500-word summary of this video."
        response = self.client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_uri(file_uri=youtube_url, mime_type="video/mp4"),
                summary_prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2000,
            )
        )
        return response.text.strip()

    def _calculate_transcript_stats(self, transcript_text):
        if not transcript_text:
            return {'char_count': 0, 'word_count': 0, 'estimated_duration': 0, 'estimated_read_time': 0}
        
        char_count = len(transcript_text)
        word_count = len(transcript_text.split())
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'estimated_duration': max(1, word_count // 150),
            'estimated_read_time': max(1, word_count // 200)
        }
        
class AudioProcessor:
    """Process audio files with Gemini for transcription"""
    
    def __init__(self):
        self.supported_audio_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm', '.wma', '.aac'}
        self.max_audio_size = 500 * 1024 * 1024  # 500MB
        self.temp_folder = UPLOAD_FOLDER
        
    def is_supported_audio_format(self, filename):
        """Check if audio format is supported"""
        return Path(filename).suffix.lower() in self.supported_audio_formats
    
    def validate_audio_file(self, file_path):
        """Validate audio file (simple validation)"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False, f"File not found: {file_path}"
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_audio_size:
                return False, f"Audio too large ({file_size/(1024*1024):.1f}MB). Max: {self.max_audio_size // (1024*1024)}MB"
            
            if file_size == 0:
                return False, "Audio file is empty"
            
            return True, None
                
        except Exception as e:
            return False, f"Invalid audio file: {str(e)}"
    
    def transcribe_with_gemini(self, audio_path):
        """
        Transcribe audio using Gemini API
        Returns: (transcript_text, error)
        """
        try:
            print(f"Uploading audio to Gemini: {audio_path}")
            
            # Determine MIME type from file extension
            extension = Path(audio_path).suffix.lower()
            mime_type_map = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav',
                '.m4a': 'audio/mp4',
                '.flac': 'audio/flac',
                '.ogg': 'audio/ogg',
                '.opus': 'audio/opus',
                '.webm': 'audio/webm',
                '.wma': 'audio/x-ms-wma',
                '.aac': 'audio/aac'
            }
            mime_type = mime_type_map.get(extension, 'audio/mpeg')
            
            # Upload file to Gemini (CORRECT SYNTAX)
            print("Uploading audio file...")
            with open(audio_path, 'rb') as f:
                audio_file = client.files.upload(
                    file=f,
                    config={'mime_type': mime_type} 
                ) # ✅ Use 'file' not 'path'
            
            print(f"✓ Uploaded: {audio_file.name}")
            
            # Wait for processing (if needed)
            print("Waiting for Gemini to process audio...")
            max_wait = 180  # 3 minutes
            wait_time = 0
            while audio_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                audio_file = client.files.get(name=audio_file.name)
                if wait_time % 10 == 0:
                    print(f"  Processing... {wait_time}s")
            
            if audio_file.state.name == "FAILED":
                return None, "Audio upload failed"
            
            if wait_time >= max_wait:
                return None, "Audio processing timeout"
            
            # Transcribe with Gemini
            print("Generating transcription with Gemini...")
            model_id = "gemini-2.0-flash"
            
            transcription_prompt = """Transcribe this audio file completely and accurately. 
            
Output ONLY the transcription text with no preamble, no explanation, and no formatting. 
Just provide the raw transcript of what is being said in the audio."""

            response = client.models.generate_content(
                model=model_id,
                contents=[
                    types.Part.from_uri(
                        file_uri=audio_file.uri,
                        mime_type=mime_type,
                    ),
                    transcription_prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Low temperature for accuracy
                    max_output_tokens=8000,
                )
            )
            
            transcript = response.text.strip()
            
            # Cleanup uploaded file
            try:
                client.files.delete(name=audio_file.name)
                print("Cleaned up Gemini uploaded file")
            except:
                pass
            
            if transcript and len(transcript) > 20:
                print(f"✓ Transcription complete: {len(transcript)} characters")
                return transcript, None
            
            return None, "Transcript too short or empty"
                
        except Exception as e:
            print(f"❌ Gemini transcription error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None, f"Transcription failed: {str(e)}"
    
    def process_audio_file(self, audio_path, filename):
        """
        Complete audio processing pipeline using Gemini
        Returns: {error, transcript, stats}
        """
        print(f"\n{'='*60}")
        print(f"AUDIO FILE PROCESSING (GEMINI): {filename}")
        print(f"{'='*60}\n")
        
        total_start = time.time()
        
        try:
            # Validate audio file
            is_valid, error_msg = self.validate_audio_file(audio_path)
            if not is_valid:
                return {
                    "error": error_msg,
                    "transcript": None,
                    "stats": None
                }
            
            # Get file size for stats
            file_size = os.path.getsize(audio_path)
            print(f"File size: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
            
            # Transcribe with Gemini
            transcribe_start = time.time()
            transcript, transcribe_error = self.transcribe_with_gemini(audio_path)
            transcribe_time = time.time() - transcribe_start
            
            if transcribe_error:
                return {
                    "error": transcribe_error,
                    "transcript": None,
                    "stats": None
                }
            
            total_time = time.time() - total_start
            
            # Calculate stats
            word_count = len(transcript.split())
            stats = {
                'char_count': len(transcript),
                'word_count': word_count,
                'file_size': file_size,
                'processing_time': round(total_time, 1),
                'transcribe_time': round(transcribe_time, 1),
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'audio_file',
                'filename': filename
            }
            
            print(f"\n{'='*60}")
            print(f"✓ AUDIO PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Transcript: {len(transcript):,} chars")
            print(f"Words: {word_count:,}")
            print(f"Total Time: {total_time:.1f}s")
            print(f"{'='*60}\n")
            
            # PRINT TRANSCRIPT PREVIEW
            print(f"\n{'='*80}")
            print(f"AUDIO TRANSCRIPT EXTRACTED: {filename}")
            print(f"{'='*80}")
            print(f"Length: {len(transcript):,} characters")
            print(f"Words: {word_count:,}")
            print(f"\nPREVIEW (first 500 chars):")
            print(f"{'-'*80}")
            print(transcript[:500])
            if len(transcript) > 500:
                print(f"... (truncated, {len(transcript) - 500:,} more characters)")
            print(f"{'='*80}\n")
            
            return {
                "error": None,
                "transcript": transcript,
                "stats": stats,
                "source": filename
            }
            
        except Exception as e:
            print(f"❌ Audio processing error: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {
                "error": f"Processing error: {str(e)}",
                "transcript": None,
                "stats": None
            }

class InstagramProcessor:
    """Process Instagram videos by downloading and uploading to Gemini using instaloader with session cookies"""
    
    def __init__(self, client=None, session_file=None):
        self.supported_domains = ['instagram.com', 'www.instagram.com']
        self.temp_folder = tempfile.gettempdir()
        
        # Setup Gemini
        if client is None:
            import google.generativeai as genai_config
            genai_config.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.genai = genai_config
        else:
            self.client = client
        
        # Setup Instaloader
        try:
            import instaloader
            self.instaloader = instaloader.Instaloader(download_videos=True, download_pictures=False)
            self.session_file = session_file  # Path to previously saved session from CLI login
            
            if self.session_file and os.path.exists(self.session_file):
                self.instaloader.load_session_from_file(username=None, filename=self.session_file)
                print(f"✓ Loaded Instagram session from {self.session_file}")
            else:
                print("⚠️ Session file not provided or not found. Only public posts may work.")
        except ImportError:
            raise ImportError("Please install instaloader: pip install instaloader")

    def validate_instagram_url(self, url):
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in self.supported_domains)
        except:
            return False

    def download_instagram_video(self, instagram_url):
        try:
            from instaloader import Post
            
            shortcode = instagram_url.rstrip("/").split("/")[-1]
            post = Post.from_shortcode(self.instaloader.context, shortcode)
            
            if not post.is_video:
                return None, "This Instagram post does not contain a video."
            
            video_url = post.video_url
            video_path = os.path.join(
                self.temp_folder,
                f"instagram_{shortcode}_{int(time.time()*1000)}.mp4"
            )
            
            print(f"Downloading Instagram video: {instagram_url}")
            r = requests.get(video_url, stream=True, headers={"User-Agent": "Mozilla/5.0"})
            with open(video_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if not os.path.exists(video_path) or os.path.getsize(video_path) == 0:
                return None, "Failed to download video. File is empty."
            
            print(f"✓ Downloaded: {os.path.getsize(video_path)/(1024*1024):.2f} MB")
            return video_path, None
        except Exception as e:
            return None, f"Download error: {str(e)}"

    def process_instagram_url(self, instagram_url):
        print(f"\n{'='*60}")
        print(f"PROCESSING INSTAGRAM: {instagram_url}")
        print(f"{'='*60}\n")
        
        video_path = None
        try:
            if not self.validate_instagram_url(instagram_url):
                return {"error": "Invalid Instagram URL. Expected format: https://www.instagram.com/p/...",
                        "transcript": None, "stats": None}
            
            video_path, download_error = self.download_instagram_video(instagram_url)
            if download_error:
                print(f"❌ Download failed: {download_error}")
                return {"error": download_error, "transcript": None, "stats": None}
            
            # Upload to Gemini
            print(f"Uploading to Gemini for analysis...")
            video_file = self.genai.upload_file(path=video_path, display_name="instagram_video.mp4")
            
            # Wait for processing
            max_wait = 120
            wait_time = 0
            while video_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                video_file = self.genai.get_file(video_file.name)
                if wait_time % 10 == 0:
                    print(f"  Processing... {wait_time}s")
            
            if video_file.state.name != "ACTIVE":
                try:
                    os.remove(video_path)
                    self.genai.delete_file(video_file.name)
                except:
                    pass
                return {"error": f"Gemini processing failed: {video_file.state.name}",
                        "transcript": None, "stats": None}
            
            # Generate transcript/summary
            model = self.genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content([
                video_file,
                "Generate a detailed transcript and a comprehensive 500-word summary of this Instagram video content."
            ])
            transcript = response.text.strip()
            
            try:
                os.remove(video_path)
                self.genai.delete_file(video_file.name)
                print("✓ Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
            
            if not transcript or len(transcript) < 50:
                return {"error": "Could not extract content from video. Video may have no audio or visual content.",
                        "transcript": None, "stats": None}
            
            word_count = len(transcript.split())
            stats = {
                'char_count': len(transcript),
                'word_count': word_count,
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'instagram',
                'url': instagram_url
            }
            
            return {"error": None, "transcript": transcript, "stats": stats, "source": instagram_url}
        
        except Exception as e:
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except:
                    pass
            import logging, traceback
            logging.error(f"Instagram processing error: {str(e)}")
            print(f"❌ Instagram error: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Processing error: {str(e)}", "transcript": None, "stats": None}
class FacebookProcessor:
    """Process Facebook videos by downloading and uploading to Gemini"""
    
    def __init__(self, client=None):
        self.supported_domains = ['facebook.com', 'fb.com', 'www.facebook.com', 'fb.watch', 'm.facebook.com']
        if client is None:
            import google.generativeai as genai_config
            genai_config.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.genai = genai_config
        else:
            self.client = client
        self.temp_folder = tempfile.gettempdir()
        self._check_ytdlp()
        
    def _check_ytdlp(self):
        """Check if yt-dlp is available"""
        self.ytdlp_available = shutil.which('yt-dlp') is not None
        if not self.ytdlp_available:
            print("⚠️  WARNING: yt-dlp not found in system PATH")
            print("   Facebook processing will fail until yt-dlp is installed")
            print("   Install with: pip install yt-dlp")
        else:
            print("✓ yt-dlp found and ready for Facebook")
        
    def validate_facebook_url(self, url):
        """Validate Facebook URL"""
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in self.supported_domains)
        except:
            return False
    
    def download_facebook_video(self, facebook_url):
        """
        Download Facebook video using yt-dlp
        Returns: (video_path, error)
        """
        # Check if yt-dlp is available
        if not self.ytdlp_available:
            error_msg = (
                "yt-dlp is not installed or not in PATH.\n"
                "Please install with: pip install yt-dlp\n"
                "Then restart your application."
            )
            return None, error_msg
        
        try:
            video_path = os.path.join(
                self.temp_folder,
                f"facebook_{int(time.time()*1000)}.mp4"
            )
            
            print(f"Downloading Facebook video: {facebook_url}")
            
            # Download video with yt-dlp
            cmd = [
                'yt-dlp',
                '-f', 'best',  # Best quality
                '--no-playlist',
                '--no-warnings',
                '-o', video_path,
                facebook_url
            ]
            
            print(f"Running command: {' '.join(cmd[:5])}... {facebook_url}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes for Facebook (can be slower)
                shell=False
            )
            
            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                print(f"yt-dlp error output: {error_msg}")
                
                # Check for common errors
                if "HTTP Error 429" in error_msg:
                    return None, "Facebook rate limit reached. Please try again later."
                elif "Private video" in error_msg or "login" in error_msg.lower():
                    return None, "Video is private or requires login. Please use a public video."
                elif "Video unavailable" in error_msg:
                    return None, "Video is unavailable or has been deleted."
                else:
                    return None, f"Download failed: {error_msg[:200]}"
            
            # Verify file was created
            if not os.path.exists(video_path):
                return None, "Video file was not created. Video may be unavailable or private."
            
            file_size = os.path.getsize(video_path)
            if file_size == 0:
                os.remove(video_path)
                return None, "Downloaded file is empty. Video may be unavailable."
            
            print(f"✓ Downloaded: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB)")
            
            return video_path, None
            
        except subprocess.TimeoutExpired:
            return None, "Download timeout (3 minutes). Video may be too long or connection is slow."
        except FileNotFoundError:
            return None, (
                "yt-dlp command not found. Please install with: pip install yt-dlp\n"
                "Make sure to restart your terminal/application after installation."
            )
        except Exception as e:
            return None, f"Download error: {str(e)}"
    
    def process_facebook_url(self, facebook_url):
        """
        Process Facebook video: Download → Upload to Gemini → Analyze
        Returns: {error, transcript, stats}
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING FACEBOOK: {facebook_url}")
        print(f"{'='*60}\n")
        
        video_path = None
        
        try:
            # Validate URL
            if not self.validate_facebook_url(facebook_url):
                return {
                    "error": "Invalid Facebook URL. Expected format: https://www.facebook.com/watch/?v=...",
                    "transcript": None,
                    "stats": None
                }
            
            # Step 1: Download video
            video_path, download_error = self.download_facebook_video(facebook_url)
            
            if download_error:
                print(f"❌ Download failed: {download_error}")
                return {
                    "error": download_error,
                    "transcript": None,
                    "stats": None
                }
            
            # Step 2: Upload to Gemini
            print(f"Uploading to Gemini for analysis...")
            video_file = self.genai.upload_file(path=video_path, display_name="facebook_video.mp4")
            
            # Wait for processing
            print("Waiting for Gemini to process...")
            max_wait = 120
            wait_time = 0
            while video_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                video_file = self.genai.get_file(video_file.name)
                if wait_time % 10 == 0:
                    print(f"  Processing... {wait_time}s")
            
            if video_file.state.name != "ACTIVE":
                # Cleanup
                try:
                    os.remove(video_path)
                    self.genai.delete_file(video_file.name)
                except:
                    pass
                return {
                    "error": f"Gemini processing failed: {video_file.state.name}",
                    "transcript": None,
                    "stats": None
                }
            
            # Step 3: Generate analysis
            print("Generating 500-word summary...")
            model = self.genai.GenerativeModel("gemini-2.0-flash")
            
            response = model.generate_content([
                video_file,
                "Generate a detailed transcript and a comprehensive 500-word summary of this Facebook video content."
            ])
            
            transcript = response.text.strip()
            
            # Cleanup
            try:
                os.remove(video_path)
                self.genai.delete_file(video_file.name)
                print("✓ Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {e}")
            
            if not transcript or len(transcript) < 50:
                return {
                    "error": "Could not extract content from video. Video may have no audio or visual content.",
                    "transcript": None,
                    "stats": None
                }
            
            # Calculate stats
            word_count = len(transcript.split())
            stats = {
                'char_count': len(transcript),
                'word_count': word_count,
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'facebook',
                'url': facebook_url
            }
            
            print(f"\n{'='*60}")
            print(f"✓ FACEBOOK PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Transcript: {len(transcript):,} chars")
            print(f"Words: {word_count:,}")
            print(f"{'='*60}\n")
            
            # PRINT TRANSCRIPT PREVIEW
            print(f"\n{'='*80}")
            print(f"FACEBOOK TRANSCRIPT: {facebook_url}")
            print(f"{'='*80}")
            print(f"Length: {len(transcript):,} characters")
            print(f"Words: {word_count:,}")
            print(f"\nPREVIEW (first 500 chars):")
            print(f"{'-'*80}")
            print(transcript[:500])
            if len(transcript) > 500:
                print(f"... (truncated, {len(transcript) - 500:,} more characters)")
            print(f"{'='*80}\n")
            
            return {
                "error": None,
                "transcript": transcript,
                "stats": stats,
                "source": facebook_url
            }
            
        except Exception as e:
            # Cleanup on error
            if video_path and os.path.exists(video_path):
                try:
                    os.remove(video_path)
                except:
                    pass
            
            import logging
            import traceback
            logging.error(f"Facebook processing error: {str(e)}")
            print(f"❌ Facebook error: {str(e)}")
            print(traceback.format_exc())
            
            return {
                "error": f"Processing error: {str(e)}",
                "transcript": None,
                "stats": None
            }

# ADD THESE CLASSES TO script.py (after FacebookProcessor class)

# class TikTokProcessor:
#     """OPTIMIZED: Process TikTok videos with direct audio download + Whisper"""
    
#     def __init__(self):
#         self.supported_domains = ['tiktok.com', 'www.tiktok.com', 'vm.tiktok.com', 'm.tiktok.com']
#         self.max_duration = 600  # 10 minutes max
#         self.temp_folder = UPLOAD_FOLDER
        
#     def validate_tiktok_url(self, url):
#         """Validate TikTok URL"""
#         try:
#             parsed = urlparse(url)
#             return any(domain in parsed.netloc for domain in self.supported_domains)
#         except:
#             return False
    
#     def download_tiktok_audio_direct(self, tiktok_url):
#         """
#         OPTIMIZED: Download ONLY audio from TikTok (no video) - 5-10x faster
#         Returns: (audio_path, duration, error)
#         """
#         try:
#             audio_path = os.path.join(
#                 self.temp_folder,
#                 f"tiktok_audio_{int(time.time()*1000)}.mp3"
#             )
            
#             print(f"Downloading TikTok AUDIO ONLY from: {tiktok_url}")
            
#             # yt-dlp: Extract ONLY audio (no video download)
#             cmd = [
#                 'yt-dlp',
#                 '-x',  # Extract audio only
#                 '--audio-format', 'mp3',
#                 '--audio-quality', '5',
#                 '--no-playlist',
#                 '--no-warnings',
#                 '--postprocessor-args', '-ar 16000',  # 16kHz for Whisper
#                 '-o', audio_path.replace('.mp3', '.%(ext)s'),
#                 tiktok_url
#             ]
            
#             start_time = time.time()
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 timeout=120
#             )
#             download_time = time.time() - start_time
            
#             if result.returncode != 0:
#                 print(f"yt-dlp error: {result.stderr}")
#                 return None, None, f"Audio download failed: {result.stderr[:200]}"
            
#             # Find actual audio file
#             actual_audio_path = audio_path
#             if not os.path.exists(audio_path):
#                 base_path = audio_path.replace('.mp3', '')
#                 for ext in ['.mp3', '.m4a', '.opus', '.webm']:
#                     test_path = base_path + ext
#                     if os.path.exists(test_path):
#                         actual_audio_path = test_path
#                         break
            
#             if not os.path.exists(actual_audio_path):
#                 return None, None, "Audio file not created"
            
#             file_size = os.path.getsize(actual_audio_path)
#             print(f"✓ Audio downloaded: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB) in {download_time:.1f}s")
            
#             # Get duration
#             try:
#                 from moviepy.editor import AudioFileClip
#                 audio_clip = AudioFileClip(actual_audio_path)
#                 duration = audio_clip.duration
#                 audio_clip.close()
#                 print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
#             except Exception as e:
#                 print(f"  Warning: Could not get duration: {e}")
#                 duration = None
            
#             return actual_audio_path, duration, None
            
#         except subprocess.TimeoutExpired:
#             return None, None, "Audio download timeout (2 minutes)"
#         except Exception as e:
#             return None, None, f"Download error: {str(e)}"
    
#     def transcribe_with_whisper_optimized(self, audio_path):
#         """Transcribe with Whisper using cached model"""
#         try:
#             print(f"Transcribing with Whisper (using cached model)...")
            
#             model = load_whisper_model()
#             if model is None:
#                 return None, "Whisper model not loaded. Please run download_whisper_model.py first."
            
#             result = model.transcribe(
#                 audio_path,
#                 language=None,
#                 fp16=False,
#                 verbose=False
#             )
            
#             transcript = result["text"].strip()
#             detected_language = result.get("language", "unknown")
            
#             if len(transcript) < 20:
#                 return None, "Transcript too short or empty"
            
#             print(f"✓ Transcription complete: {len(transcript)} chars, language: {detected_language}")
#             return transcript, None
            
#         except Exception as e:
#             logger.error(f"Whisper transcription error: {str(e)}")
#             return None, f"Transcription failed: {str(e)}"
    
#     def process_tiktok_url(self, tiktok_url):
#         """
#         Complete TikTok processing with audio-only download
#         Returns: {error, transcript, stats}
#         """
#         print(f"\n{'='*60}")
#         print(f"FAST TIKTOK PROCESSING: {tiktok_url}")
#         print(f"{'='*60}\n")
        
#         audio_path = None
#         total_start = time.time()
        
#         try:
#             if not self.validate_tiktok_url(tiktok_url):
#                 return {"error": "Invalid TikTok URL", "transcript": None, "stats": None}
            
#             # Download audio only
#             audio_path, duration, download_error = self.download_tiktok_audio_direct(tiktok_url)
            
#             if download_error:
#                 return {"error": download_error, "transcript": None, "stats": None}
            
#             if duration and duration > self.max_duration:
#                 try:
#                     os.remove(audio_path)
#                 except:
#                     pass
#                 return {
#                     "error": f"Audio too long ({duration/60:.1f} min). Max: {self.max_duration/60} min",
#                     "transcript": None,
#                     "stats": None
#                 }
            
#             # Transcribe
#             transcribe_start = time.time()
#             transcript, transcribe_error = self.transcribe_with_whisper_optimized(audio_path)
#             transcribe_time = time.time() - transcribe_start
            
#             # Cleanup
#             try:
#                 os.remove(audio_path)
#             except:
#                 pass
            
#             if transcribe_error:
#                 return {"error": transcribe_error, "transcript": None, "stats": None}
            
#             total_time = time.time() - total_start
            
#             word_count = len(transcript.split())
#             stats = {
#                 'char_count': len(transcript),
#                 'word_count': word_count,
#                 'actual_duration': duration,
#                 'processing_time': round(total_time, 1),
#                 'transcribe_time': round(transcribe_time, 1),
#                 'estimated_read_time': max(1, word_count // 200),
#                 'source_type': 'tiktok',
#                 'url': tiktok_url
#             }
            
#             print(f"\n{'='*60}")
#             print(f"✓ TIKTOK PROCESSING COMPLETE")
#             print(f"{'='*60}")
#             print(f"Transcript: {len(transcript):,} chars")
#             print(f"Words: {word_count:,}")
#             print(f"Duration: {duration:.1f}s" if duration else "Duration: Unknown")
#             print(f"Total Time: {total_time:.1f}s")
#             print(f"{'='*60}\n")
            
#             return {
#                 "error": None,
#                 "transcript": transcript,
#                 "stats": stats,
#                 "source": tiktok_url
#             }
            
#         except Exception as e:
#             if audio_path and os.path.exists(audio_path):
#                 try:
#                     os.remove(audio_path)
#                 except:
#                     pass
            
#             logger.error(f"TikTok processing error: {str(e)}")
#             return {"error": f"Processing error: {str(e)}", "transcript": None, "stats": None}


class ImageProcessor:
    """Process images with OCR and visual analysis using Gemini Vision - OPTIMIZED FOR URLs"""
    
    def __init__(self):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.max_image_size = 20 * 1024 * 1024  # 20MB
        
    def is_supported_image_format(self, filename):
        """Check if image format is supported"""
        return Path(filename).suffix.lower() in self.supported_image_formats
    
    def process_image_url(self, image_url, filename=None):
        """
        Process image from URL: Download → Upload to Gemini → Get 50-100 word summary
        Returns: {error, text, stats}
        """
        print(f"\n{'='*60}")
        print(f"IMAGE PROCESSING (URL): {filename or image_url}")
        print(f"{'='*60}\n")
        
        temp_file = None
        uploaded_file = None
        
        try:
            # Validate URL
            if not image_url or not image_url.strip():
                return {"error": "Empty URL provided", "text": None, "stats": None}
            
            if not image_url.startswith(('http://', 'https://')):
                return {"error": f"Invalid URL format: {image_url}", "text": None, "stats": None}
            
            # Extract filename from URL if not provided
            if not filename:
                filename = image_url.split('/')[-1].split('?')[0] or 'image.jpg'
            
            print(f"Downloading image from: {image_url}")
            
            # ✅ STEP 1: Download the image
            import requests
            response = requests.get(image_url, timeout=30, allow_redirects=True)
            
            if response.status_code != 200:
                return {"error": f"Download failed: HTTP {response.status_code}", "text": None, "stats": None}
            
            print(f"✓ Downloaded: {len(response.content):,} bytes ({len(response.content)/(1024*1024):.2f} MB)")
            
            # Validate size
            if len(response.content) > self.max_image_size:
                return {"error": f"Image too large ({len(response.content)/(1024*1024):.1f}MB). Max: {self.max_image_size // (1024*1024)}MB", "text": None, "stats": None}
            
            # ✅ STEP 2: Save to temporary file
            import tempfile
            import os
            
            # Determine file extension and MIME type
            file_ext = os.path.splitext(filename)[1].lower()
            if not file_ext or file_ext not in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
                # Try to detect from content-type header
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    file_ext = '.jpg'
                elif 'png' in content_type:
                    file_ext = '.png'
                elif 'webp' in content_type:
                    file_ext = '.webp'
                else:
                    file_ext = '.jpg'  # Default
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext)
            temp_file.write(response.content)
            temp_file.close()
            
            print(f"✓ Saved to temp: {temp_file.name}")
            
            # ✅ STEP 3: Upload to Gemini
            print("Uploading to Gemini...")
            
            # Determine MIME type
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            mime_type = mime_type_map.get(file_ext, 'image/jpeg')
            
            with open(temp_file.name, 'rb') as f:
                uploaded_file = client.files.upload(
                    file=f,
                    config={'mime_type': mime_type}
                )
            
            print(f"✓ Uploaded to Gemini: {uploaded_file.name}")
            
            # Wait for processing
            max_wait = 60
            wait_time = 0
            while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                uploaded_file = client.files.get(name=uploaded_file.name)
                if wait_time % 10 == 0:
                    print(f"  Processing... {wait_time}s")
            
            if uploaded_file.state.name == "FAILED":
                return {"error": "Gemini upload failed", "text": None, "stats": None}
            
            # ✅ STEP 4: Analyze with Gemini Vision
            print("Analyzing with Gemini Vision...")
            
            prompt = """Analyze this image and provide a concise 50-100 word summary covering:

    1. What the image shows (main subject, scene, or content)
    2. Any visible text or data (if present)
    3. Key insights or information that would be useful for YouTube script creation

    Be specific and actionable. Focus ONLY on information relevant for video content."""
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,  # ✅ Now using Gemini URI
                        mime_type=mime_type
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200
                )
            )
            
            # ✅ STEP 5: Cleanup
            try:
                os.remove(temp_file.name)
                print("✓ Cleaned up temp file")
            except:
                pass
            
            try:
                client.files.delete(name=uploaded_file.name)
                print("✓ Cleaned up Gemini file")
            except:
                pass
            
            if not response.text or len(response.text.strip()) < 20:
                return {"error": "Could not extract content from image", "text": None, "stats": None}
            
            extracted_text = response.text.strip()
            word_count = len(extracted_text.split())
            
            # Validate word count
            if word_count < 30:
                print(f"⚠️ Warning: Summary too short ({word_count} words)")
            elif word_count > 150:
                print(f"⚠️ Warning: Summary too long ({word_count} words), truncating...")
                words = extracted_text.split()
                extracted_text = ' '.join(words[:100])
                word_count = 100
            
            stats = {
                'char_count': len(extracted_text),
                'word_count': word_count,
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'image',
                'filename': filename,
                'url': image_url
            }
            
            print(f"\n{'='*60}")
            print(f"✓ IMAGE PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Summary: {word_count} words ({len(extracted_text)} chars)")
            print(f"\nSUMMARY:")
            print(f"{'-'*60}")
            print(extracted_text)
            print(f"{'='*60}\n")
            
            return {
                "error": None,
                "text": extracted_text,
                "stats": stats,
                "source": filename
            }
            
        except Exception as e:
            # Cleanup on error
            if temp_file and os.path.exists(temp_file.name):
                try:
                    os.remove(temp_file.name)
                except:
                    pass
            
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except:
                    pass
            
            logger.error(f"Image URL processing error: {str(e)}")
            import traceback
            print(f"❌ Image processing error: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Processing error: {str(e)}", "text": None, "stats": None}


    # ==========================================================
    # ALSO UPDATE process_image_file() to use same pattern
    # ==========================================================

    def process_image_file(self, image_path, filename):
        """
        Process image from local file: Upload to Gemini → Get 50-100 word summary
        Returns: {error, text, stats}
        """
        print(f"\n{'='*60}")
        print(f"IMAGE PROCESSING (FILE): {filename}")
        print(f"{'='*60}\n")
        
        uploaded_file = None
        
        try:
            # Validate file
            if not os.path.exists(image_path):
                return {"error": f"File not found: {image_path}", "text": None, "stats": None}
            
            file_size = os.path.getsize(image_path)
            if file_size > self.max_image_size:
                return {"error": f"Image too large. Max: {self.max_image_size // (1024*1024)}MB", "text": None, "stats": None}
            
            print(f"Uploading image to Gemini: {filename}")
            
            # Determine MIME type
            file_ext = os.path.splitext(filename)[1].lower()
            mime_type_map = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp',
                '.bmp': 'image/bmp'
            }
            mime_type = mime_type_map.get(file_ext, 'image/jpeg')
            
            # Upload to Gemini
            with open(image_path, 'rb') as f:
                uploaded_file = client.files.upload(
                    file=f,
                    config={'mime_type': mime_type}
                )
            
            print(f"✓ Uploaded: {uploaded_file.name}")
            
            # Wait for processing
            max_wait = 60
            wait_time = 0
            while uploaded_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                uploaded_file = client.files.get(name=uploaded_file.name)
                if wait_time % 10 == 0:
                    print(f"  Processing... {wait_time}s")
            
            if uploaded_file.state.name == "FAILED":
                return {"error": "Image upload failed", "text": None, "stats": None}
            
            # Analyze with Gemini Vision
            print("Analyzing with Gemini Vision...")
            
            prompt = """Analyze this image and provide a concise 50-100 word summary covering:

    1. What the image shows (main subject, scene, or content)
    2. Any visible text or data (if present)
    3. Key insights or information that would be useful for YouTube script creation

    Be specific and actionable. Focus ONLY on information relevant for video content."""
            
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=mime_type
                    ),
                    prompt
                ],
                config=types.GenerateContentConfig(
                    temperature=0.3,
                    max_output_tokens=200
                )
            )
            
            # Cleanup
            try:
                client.files.delete(name=uploaded_file.name)
                print("✓ Cleaned up Gemini file")
            except:
                pass
            
            if not response.text or len(response.text.strip()) < 20:
                return {"error": "Could not extract meaningful content from image", "text": None, "stats": None}
            
            extracted_text = response.text.strip()
            word_count = len(extracted_text.split())
            
            # Validate word count
            if word_count > 150:
                words = extracted_text.split()
                extracted_text = ' '.join(words[:100])
                word_count = 100
            
            stats = {
                'char_count': len(extracted_text),
                'word_count': word_count,
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'image',
                'filename': filename
            }
            
            print(f"\n{'='*60}")
            print(f"✓ IMAGE PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Summary: {word_count} words")
            print(f"{'='*60}\n")
            
            return {
                "error": None,
                "text": extracted_text,
                "stats": stats,
                "source": filename
            }
            
        except Exception as e:
            if uploaded_file:
                try:
                    client.files.delete(name=uploaded_file.name)
                except:
                    pass
            
            logger.error(f"Image file processing error: {str(e)}")
            import traceback
            print(f"❌ Image processing error: {str(e)}")
            print(traceback.format_exc())
            return {"error": f"Processing error: {str(e)}", "text": None, "stats": None}

class TextProcessor:
    """Process direct text input"""
    
    def __init__(self):
        self.max_text_length = 50000  # 50k characters max
    
    def process_text_input(self, text_content, source_name="Direct Text Input"):
        """
        Process direct text input
        Returns: {error, text, stats}
        """
        print(f"\n{'='*60}")
        print(f"TEXT INPUT PROCESSING: {source_name}")
        print(f"{'='*60}\n")
        
        try:
            if not text_content or len(text_content.strip()) < 10:
                return {"error": "Text too short (minimum 10 characters)", "text": None, "stats": None}
            
            text_content = text_content.strip()
            
            if len(text_content) > self.max_text_length:
                print(f"Truncating text from {len(text_content)} to {self.max_text_length} characters")
                text_content = text_content[:self.max_text_length] + "\n\n[Text truncated...]"
            
            word_count = len(text_content.split())
            
            stats = {
                'char_count': len(text_content),
                'word_count': word_count,
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'text_input',
                'source_name': source_name
            }
            
            print(f"✓ Text processed: {len(text_content):,} chars, {word_count:,} words\n")
            
            return {
                "error": None,
                "text": text_content,
                "stats": stats,
                "source": source_name
            }
            
        except Exception as e:
            logger.error(f"Text processing error: {str(e)}")
            return {"error": f"Processing error: {str(e)}", "text": None, "stats": None}
            
class EnhancedScriptGenerator:
    """
    Professional YouTube Script Generator - Topic-Focused Approach
    
    Philosophy:
    - Content over constraints: Quality explanations take priority
    - Knowledge synthesis: Learn from sources, don't copy them
    - Natural flow: Topics should feel organic, not forced
    - Engagement first: Every section must provide value
    """
    
    def __init__(self):
        """Initialize the script generator"""
        # Get API key
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        
        self.anthropic_client = anthropic.Anthropic(api_key=api_key)

        if client is None:
            import google.generativeai as genai_config
            genai_config.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self.genai = genai_config
        else:
            self.client = client
        # Natural speaking rate
        self.WORDS_PER_MINUTE = 135
        
        # Duration limits
        self.MIN_DURATION_SECONDS = 30
        self.MAX_DURATION_SECONDS = 45 * 60  # 45 minutes
        
        # Word count flexibility (10-15%)
        self.WORD_COUNT_TOLERANCE = 0.15
    
    # =========================================================================
    # ANALYSIS PROMPTS - Deep Topic Extraction
    # =========================================================================
    
    def _get_style_analysis_prompt(self):
        """Extract creator's authentic voice and presentation style"""
        return """You are analyzing video transcripts to understand this creator's unique communication style.

**YOUR TASK:**
Identify HOW this creator talks, teaches, and connects with their audience.

**ANALYZE:**

1. **Voice & Tone**
   - Is it conversational, authoritative, excited, calm, humorous, serious?
   - What's the energy level? (Fast-paced, measured, dynamic)
   - How do they address the audience? (Direct "you", inclusive "we", storyteller "I")

2. **Language Patterns**
   - Vocabulary: Simple and accessible vs technical and detailed
   - Sentence structure: Short punchy sentences vs flowing explanations
   - Signature phrases or expressions they use repeatedly
   - How they emphasize important points (repetition, analogies, questions)

3. **Teaching Approach**
   - How do they explain complex ideas? (Analogies, examples, step-by-step)
   - Do they use stories, data, or demonstrations?
   - How do they transition between topics?
   - What's their pacing: quick overview vs deep dive?

4. **Unique Elements**
   - What makes THIS creator different from others?
   - Recurring patterns in how they structure content
   - Their relationship with viewers (expert, peer, guide, entertainer)
   - Any specific hooks or techniques they use

**OUTPUT:**
Write a 400-600 word style guide that captures this creator's essence. This guide will help write scripts that SOUND like them.

Focus on actionable patterns, not generic descriptions.

**TRANSCRIPTS TO ANALYZE:**
"""

    def _get_topic_analysis_prompt(self):
        """Extract deep topic knowledge from inspiration sources"""
        return """You are a content researcher analyzing videos/sources to extract KNOWLEDGE for creating new YouTube content.

**YOUR MISSION:**
Don't just summarize - extract the underlying KNOWLEDGE, INSIGHTS, and TECHNIQUES that can inform new content.

**EXTRACT:**

1. **Core Topics & Themes**
   - What subjects are being discussed?
   - What specific subtopics or angles are explored?
   - What's the depth level? (Beginner, intermediate, expert)
   - How do topics connect to each other?

2. **Facts, Data & Evidence**
   - Specific statistics, numbers, research findings
   - Historical context or timeline information
   - Technical details, processes, or mechanisms
   - Real-world examples and case studies
   - Expert quotes or authoritative sources

3. **Unique Angles & Insights**
   - Fresh perspectives or unconventional viewpoints
   - Common myths or misconceptions addressed
   - Problems identified and solutions proposed
   - Controversies or debates in the field
   - Gaps in coverage - what's missing or underexplored?

4. **Presentation Techniques**
   - How were complex ideas made simple?
   - Effective analogies, metaphors, or comparisons used
   - Storytelling approaches that worked
   - Visual concepts or demonstrations described
   - Hooks that grabbed attention

5. **Actionable Knowledge**
   - Practical tips, strategies, or how-to information
   - Step-by-step processes or frameworks
   - Tools, resources, or recommendations
   - Common mistakes to avoid
   - Key takeaways viewers should remember

**IMPORTANT:**
- Extract KNOWLEDGE, not just content summaries
- Identify PATTERNS across multiple sources
- Note what makes content engaging and valuable
- Look for what's MISSING - topics that need deeper exploration

**OUTPUT:**
Provide a comprehensive knowledge base (600-1000 words) organized by topic area. This should be a rich resource for script creation.

**SOURCES TO ANALYZE:**
"""

    def _get_document_analysis_prompt(self):
        """Extract usable knowledge from documents, images, text inputs"""
        return """You are a knowledge curator extracting information from documents to create engaging YouTube content.

**YOUR TASK:**
Transform document content into USABLE KNOWLEDGE for video scripts.

**EXTRACT:**

1. **Core Concepts & Principles**
   - Main ideas and theories
   - Frameworks or models explained
   - Fundamental concepts viewers need to understand
   - How concepts build from basic to advanced

2. **Data & Evidence**
   - Research findings and studies
   - Statistics, metrics, or measurements
   - Historical data or trends
   - Case studies and real-world applications
   - Expert opinions or authoritative statements

3. **Practical Information**
   - How-to instructions or procedures
   - Best practices and methodologies  
   - Tools, techniques, or technologies
   - Implementation strategies
   - Troubleshooting or problem-solving approaches

4. **Context & Background**
   - Historical development or evolution
   - Current state of affairs
   - Future trends or predictions
   - Related fields or interdisciplinary connections
   - Why this matters - real-world impact

5. **Content Opportunities**
   - Complex topics that need simplification
   - Interesting angles or perspectives
   - Questions this information answers
   - Potential misconceptions to clarify
   - Visual or narrative possibilities

**CRITICAL:**
- Extract KNOWLEDGE, don't just repeat text
- Focus on what's USEFUL for video content
- Identify the "so what?" - why viewers should care
- Note what needs more context or explanation

**OUTPUT:**
Provide an organized knowledge extraction (600-1000 words) structured by topic area. Make it actionable for scriptwriting.

**DOCUMENTS TO ANALYZE:**
"""

    def _get_script_generation_prompt(self):
        """Main script generation prompt - topic-focused and engaging"""
        return """You are an expert YouTube scriptwriter creating an engaging, educational video script.

**CREATOR'S VOICE:**
{style_profile}

**TOPIC KNOWLEDGE BASE:**
{topic_knowledge}

**REFERENCE MATERIALS:**
{document_knowledge}

**USER'S REQUEST:**
{user_prompt}

**DURATION PARAMETERS:**
- Target Duration: {target_minutes:.1f} minutes ({target_seconds} seconds)
- Target Word Count: {target_words} words
- Acceptable Range: {min_words}-{max_words} words (±15% flexibility)
- Speaking Rate: 135 words per minute

**YOUR CREATIVE PROCESS:**

**STEP 1: UNDERSTAND THE TOPIC**
- What is the user asking about?
- What's the core topic viewers need to understand?
- What knowledge from sources is most relevant?
- What angle or approach would be most engaging?

**STEP 2: PLAN THE STRUCTURE**

Allocate time naturally based on topic complexity:
- **Hook** (5-10% of time): Grab attention immediately
- **Introduction** (10-15% of time): Set up what viewers will learn
- **Main Topics** (65-75% of time): Deep, valuable exploration
- **Conclusion** (10-15% of time): Summarize and inspire action

Number of topics based on duration:
- 30 sec - 2 min: 2-3 tight topics
- 2-5 min: 3-5 topics with good depth
- 5-10 min: 4-7 topics, thorough exploration
- 10-20 min: 6-10 topics, detailed coverage
- 20-45 min: 8-15 topics, comprehensive deep-dive

**STEP 3: WRITE ENGAGING CONTENT**

Quality over rigid constraints:
- Use the creator's natural voice throughout
- Explain concepts THOROUGHLY - don't rush
- Include examples, analogies, or stories
- Build on previous topics naturally
- Make every section valuable and interesting
- Synthesize knowledge - don't just repeat sources

**STEP 4: MANAGE DURATION**

Word count discipline:
- Aim for {target_words} words (range: {min_words}-{max_words})
- Count words as you write
- If a topic needs more space for clarity, take it (within ±15%)
- Quality explanations > arbitrary word limits
- Each topic should feel complete, not rushed

**FORMATTING REQUIREMENTS:**

Use this EXACT format:

```
## [Compelling, Curiosity-Driving Title]

### [0:00-{hook_end}] Hook

[Start with IMPACT - no greetings. Use a surprising fact, provocative question, or bold statement that makes viewers want to stay. This should create immediate curiosity about the topic.]

### [{intro_start}-{intro_end}] Introduction

[Build on the hook. Explain what viewers will learn and WHY it matters to them. Set expectations. Create anticipation for what's coming.]

### [{topic1_start}-{topic1_end}] [First Topic Name]

[Thoroughly explain this topic. Provide context, details, examples. Make it engaging and crystal clear. Viewers should understand AND remember this.]

### [{topic2_start}-{topic2_end}] [Second Topic Name]

[Build naturally from the previous topic. Maintain energy. Go deep enough to provide real value. Use the creator's voice and teaching style.]

[Continue with more topics as needed...]

### [{conclusion_start}-{conclusion_end}] Conclusion

[Powerful summary of key takeaways. Final insight or call to action. End with impact that leaves viewers satisfied but wanting more.]
```

**TIMESTAMP CALCULATION RULES:**

CRITICAL - Calculate timestamps based on word count:
- 135 words = 1 minute = 60 seconds
- Examples:
  * 68 words = 30 seconds (0:30)
  * 135 words = 1 minute (1:00)
  * 270 words = 2 minutes (2:00)
  * 405 words = 3 minutes (3:00)

Process:
1. Count words in each section
2. Divide by 135 to get minutes
3. Convert to MM:SS format
4. Make timestamps continuous (no gaps)
5. Start Hook at 0:00
6. End Conclusion at target duration

**CRITICAL RULES:**

1. **NO production notes** - only spoken content
2. **NO visual directions** - focus on what's said
3. **NO tone markers** - words should convey tone naturally
4. **Quality first** - if topics need proper explanation, take the words (within ±15%)
5. **Be authentic** - write in creator's voice
6. **Add value** - every section teaches or engages
7. **Stay natural** - don't force awkward transitions
8. **Format properly** - ### [MM:SS-MM:SS] Topic Name, then newline, then content
9. **MANDATORY: Use \\n\\n before every timestamp** to maintain formatting
10. **Each section structure**: \\n\\n### [MM:SS-MM:SS] Topic Name \\n[content on new line]

**TOPIC EXPLORATION GUIDE:**

For {target_minutes:.1f} minute duration:
- Very Short (0.5-2 min): Keep topics focused, hit key points quickly
- Short (2-5 min): Good depth on each topic, clear explanations
- Medium (5-10 min): Thorough exploration with examples
- Long (10-20 min): Deep dives, multiple angles per topic
- Very Long (20-45 min): Comprehensive, detailed coverage

**GENERATE THE SCRIPT NOW:**

Remember:
- Write naturally in the creator's authentic voice
- SYNTHESIZE knowledge from sources - don't copy
- Make each topic engaging AND complete
- Calculate timestamps accurately from word count
- Stay within {min_words}-{max_words} words (±15% of {target_words})
- Prioritize viewer value over rigid constraints
- Use \\n\\n before EVERY timestamp heading
- Don't use time (ex. 2024, May 2025 etc) in script
"""

    def _get_modification_prompt(self):
        """Prompt for modifying scripts via chat"""
        return """You are modifying a YouTube script based on user feedback.

**CURRENT SCRIPT:**
{current_script}

**CREATOR'S STYLE:**
{style_profile}

**USER'S MODIFICATION REQUEST:**
{user_message}

**YOUR TASK:**

Modify the script according to the request while:
1. Maintaining the creator's authentic voice
2. Keeping the topic-based structure with proper formatting
3. Recalculating timestamps if word count changes (135 words/min)
4. Preserving what works well
5. Making changes feel natural, not forced

**TIMESTAMP RECALCULATION:**
If you change word count significantly:
- Count words in modified section
- Calculate new duration: words ÷ 135 = minutes
- Update timestamps to be continuous
- Ensure no gaps between sections

**FORMATTING RULES:**
- Keep format: \\n\\n### [MM:SS-MM:SS] Topic Name \\n[content]
- Use \\n\\n before every timestamp heading
- Maintain script quality and engagement
- Don't add production notes or visual directions
- Changes should blend seamlessly

**OUTPUT:**
Return the complete modified script with updated timestamps and proper formatting.
"""

    # =========================================================================
    # CLAUDE API INTERACTION
    # =========================================================================
    
    def _call_claude(self, prompt: str, max_tokens: int = 16000, temperature: float = 0.7) -> str:
        """
        Call Claude API with error handling
        
        Args:
            prompt: The prompt to send to Claude
            max_tokens: Maximum tokens in response
            temperature: Creativity level (0.0-1.0)
        
        Returns:
            Claude's response text
        """
        try:
            print(f"\n📡 Calling Claude API...")
            print(f"   Model: claude-sonnet-4-20250514")
            print(f"   Max tokens: {max_tokens:,}")
            print(f"   Temperature: {temperature}")
            print(f"   Prompt length: {len(prompt):,} characters")
            
            message = self.anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            print(f"✓ Response received: {len(response_text):,} characters")
            print(f"✓ Estimated words: {len(response_text.split()):,}")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            print(f"❌ Claude API error: {str(e)}")
            raise

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================
    
    def analyze_creator_style(self, personal_transcripts: list) -> str:
        """
        Analyze creator's unique voice from their personal videos
        
        Args:
            personal_transcripts: List of transcript strings from creator's videos
        
        Returns:
            Detailed style profile as string
        """
        if not personal_transcripts:
            print("ℹ️  No personal videos provided - using professional default style")
            return self._get_default_style()
        
        print("\n" + "="*80)
        print("ANALYZING CREATOR STYLE")
        print("="*80)
        print(f"Videos to analyze: {len(personal_transcripts)}")
        
        # Combine transcripts
        combined = "\n\n--- VIDEO SEPARATOR ---\n\n".join(personal_transcripts)
        
        # Smart truncation (keep beginning and end for style consistency)
        MAX_CHARS = 50000
        if len(combined) > MAX_CHARS:
            print(f"⚠️  Truncating from {len(combined):,} to {MAX_CHARS:,} characters")
            chunk = MAX_CHARS // 2
            combined = f"{combined[:chunk]}\n\n[...MIDDLE CONTENT OMITTED...]\n\n{combined[-chunk:]}"
        
        # Build prompt
        prompt = self._get_style_analysis_prompt() + combined
        
        # Call Claude
        style_profile = self._call_claude(
            prompt=prompt,
            max_tokens=2500,
            temperature=0.1  # Low temp for consistent analysis
        )
        
        print("="*80)
        print("STYLE ANALYSIS COMPLETE")
        print("="*80)
        print(f"Length: {len(style_profile):,} characters\n")
        print("PREVIEW:")
        print("-"*80)
        print(style_profile[:600] + "..." if len(style_profile) > 600 else style_profile)
        print("="*80 + "\n")
        
        return style_profile.strip()
    
    def analyze_inspiration_content(self, inspiration_transcripts: list) -> str:
        """
        Extract deep topic knowledge from inspiration sources
        
        Args:
            inspiration_transcripts: List of transcripts from videos/sources
        
        Returns:
            Comprehensive topic knowledge base as string
        """
        if not inspiration_transcripts:
            print("ℹ️  No inspiration sources provided")
            return "No specific topic guidance provided. Create engaging content based on user request."
        
        print("\n" + "="*80)
        print("ANALYZING TOPIC KNOWLEDGE")
        print("="*80)
        print(f"Sources to analyze: {len(inspiration_transcripts)}")
        
        # Combine sources
        combined = "\n\n--- SOURCE SEPARATOR ---\n\n".join(inspiration_transcripts)
        
        # Smart truncation
        MAX_CHARS = 60000
        if len(combined) > MAX_CHARS:
            print(f"⚠️  Truncating from {len(combined):,} to {MAX_CHARS:,} characters")
            chunk = MAX_CHARS // 2
            combined = f"{combined[:chunk]}\n\n[...MIDDLE CONTENT OMITTED...]\n\n{combined[-chunk:]}"
        
        # Build prompt
        prompt = self._get_topic_analysis_prompt() + combined
        
        # Call Claude
        topic_knowledge = self._call_claude(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.2  # Slightly creative for finding insights
        )
        
        print("="*80)
        print("TOPIC ANALYSIS COMPLETE")
        print("="*80)
        print(f"Length: {len(topic_knowledge):,} characters\n")
        print("PREVIEW:")
        print("-"*80)
        print(topic_knowledge[:600] + "..." if len(topic_knowledge) > 600 else topic_knowledge)
        print("="*80 + "\n")
        
        return topic_knowledge.strip()
    
    def analyze_documents(self, document_texts: list) -> str:
        """
        Extract knowledge from documents, images, text inputs
        
        Args:
            document_texts: List of extracted text from documents
        
        Returns:
            Organized knowledge base as string
        """
        if not document_texts:
            print("ℹ️  No documents provided")
            return "No reference documents provided."
        
        print("\n" + "="*80)
        print("ANALYZING REFERENCE DOCUMENTS")
        print("="*80)
        print(f"Documents to analyze: {len(document_texts)}")
        
        # Combine documents
        combined = "\n\n--- DOCUMENT SEPARATOR ---\n\n".join(document_texts)
        
        # Smart truncation
        MAX_CHARS = 60000
        if len(combined) > MAX_CHARS:
            print(f"⚠️  Truncating from {len(combined):,} to {MAX_CHARS:,} characters")
            chunk = MAX_CHARS // 2
            combined = f"{combined[:chunk]}\n\n[...MIDDLE CONTENT OMITTED...]\n\n{combined[-chunk:]}"
        
        # Build prompt
        prompt = self._get_document_analysis_prompt() + combined
        
        # Call Claude
        document_knowledge = self._call_claude(
            prompt=prompt,
            max_tokens=4000,
            temperature=0.2
        )
        
        print("="*80)
        print("DOCUMENT ANALYSIS COMPLETE")
        print("="*80)
        print(f"Length: {len(document_knowledge):,} characters\n")
        print("PREVIEW:")
        print("-"*80)
        print(document_knowledge[:600] + "..." if len(document_knowledge) > 600 else document_knowledge)
        print("="*80 + "\n")
        
        return document_knowledge.strip()
    
    # =========================================================================
    # SCRIPT GENERATION
    # =========================================================================
    
    def generate_enhanced_script(
        self,
        style_profile: str,
        inspiration_summary: str,  # Keeping original parameter name for compatibility
        document_insights: str,    # Keeping original parameter name for compatibility
        user_prompt: str,
        target_minutes: float = None
    ) -> str:
        """
        Generate a complete YouTube script with topic-based structure
        
        Args:
            style_profile: Creator's voice and style
            inspiration_summary: Extracted knowledge from inspiration sources (topic_knowledge)
            document_insights: Extracted knowledge from documents
            user_prompt: User's request for the script
            target_minutes: Target duration in minutes (default: 10)
        
        Returns:
            Complete formatted script
        """
        print("\n" + "="*80)
        print("GENERATING YOUTUBE SCRIPT")
        print("="*80)
        
        # Set default if not provided
        if target_minutes is None:
            target_minutes = 10.0
            print(f"No duration specified - using default: {target_minutes} minutes")
        
        # Validate duration
        target_seconds = int(target_minutes * 60)
        if target_seconds < self.MIN_DURATION_SECONDS:
            print(f"⚠️  Duration too short ({target_seconds}s), using minimum {self.MIN_DURATION_SECONDS}s")
            target_seconds = self.MIN_DURATION_SECONDS
            target_minutes = target_seconds / 60
        elif target_seconds > self.MAX_DURATION_SECONDS:
            print(f"⚠️  Duration too long ({target_seconds}s), using maximum {self.MAX_DURATION_SECONDS}s")
            target_seconds = self.MAX_DURATION_SECONDS
            target_minutes = target_seconds / 60
        
        # Calculate word counts
        target_words = int(target_minutes * self.WORDS_PER_MINUTE)
        min_words = int(target_words * (1 - self.WORD_COUNT_TOLERANCE))
        max_words = int(target_words * (1 + self.WORD_COUNT_TOLERANCE))
        
        # Calculate example timestamps for the prompt
        hook_words = int(target_words * 0.08)  # ~8% for hook
        intro_words = int(target_words * 0.12)  # ~12% for intro
        conclusion_words = int(target_words * 0.12)  # ~12% for conclusion
        
        hook_seconds = int((hook_words / self.WORDS_PER_MINUTE) * 60)
        intro_seconds = int((intro_words / self.WORDS_PER_MINUTE) * 60)
        conclusion_seconds = int((conclusion_words / self.WORDS_PER_MINUTE) * 60)
        
        intro_start = hook_seconds
        topics_start = intro_start + intro_seconds
        conclusion_start = target_seconds - conclusion_seconds
        
        # Format timestamps for prompt
        def format_timestamp(seconds):
            mins = seconds // 60
            secs = seconds % 60
            return f"{mins}:{secs:02d}"
        
        hook_end = format_timestamp(hook_seconds)
        intro_start_ts = hook_end
        intro_end_ts = format_timestamp(topics_start)
        topic1_start = intro_end_ts
        topic1_end = format_timestamp(topics_start + 120)  # Example 2-min topic
        topic2_start = topic1_end
        topic2_end = format_timestamp(topics_start + 240)  # Example 4-min mark
        conclusion_start_ts = format_timestamp(conclusion_start)
        conclusion_end_ts = format_timestamp(target_seconds)
        
        print(f"\nDURATION PARAMETERS:")
        print(f"  Target: {target_minutes:.1f} minutes ({target_seconds} seconds)")
        print(f"  Target words: {target_words:,}")
        print(f"  Acceptable range: {min_words:,} - {max_words:,} words")
        print(f"  Speaking rate: {self.WORDS_PER_MINUTE} words/minute")
        print(f"  Flexibility: ±{int(self.WORD_COUNT_TOLERANCE * 100)}%")
        
        # Build generation prompt (using parameter names that match the function signature)
        prompt = self._get_script_generation_prompt().format(
            style_profile=style_profile,
            topic_knowledge=inspiration_summary,  # Map to topic_knowledge in prompt
            document_knowledge=document_insights,  # Map to document_knowledge in prompt
            user_prompt=user_prompt,
            target_minutes=target_minutes,
            target_seconds=target_seconds,
            target_words=target_words,
            min_words=min_words,
            max_words=max_words,
            hook_end=hook_end,
            intro_start=intro_start_ts,
            intro_end=intro_end_ts,
            topic1_start=topic1_start,
            topic1_end=topic1_end,
            topic2_start=topic2_start,
            topic2_end=topic2_end,
            conclusion_start=conclusion_start_ts,
            conclusion_end=conclusion_end_ts
        )
        
        # Calculate max_tokens (add 50% buffer for formatting and flexibility)
        max_tokens = int(max_words * 1.8)
        
        # Generate script
        script = self._call_claude(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.75  # More creative for engaging content
        )
        
        # Validate and report
        script = script.strip()
        actual_words = len(script.split())
        actual_minutes = actual_words / self.WORDS_PER_MINUTE
        
        print("\n" + "="*80)
        print("SCRIPT GENERATION COMPLETE")
        print("="*80)
        print(f"Generated words: {actual_words:,}")
        print(f"Target range: {min_words:,} - {max_words:,}")
        print(f"Estimated duration: {actual_minutes:.2f} minutes")
        print(f"Target duration: {target_minutes:.1f} minutes")
        
        # Check if within range
        if min_words <= actual_words <= max_words:
            deviation = ((actual_words - target_words) / target_words) * 100
            print(f"✓ Word count within range ({deviation:+.1f}% from target)")
        elif actual_words < min_words:
            deviation = ((min_words - actual_words) / target_words) * 100
            print(f"⚠️  Script is {deviation:.1f}% shorter than minimum")
            print(f"   This may be acceptable if content is high quality")
        else:
            deviation = ((actual_words - max_words) / target_words) * 100
            print(f"⚠️  Script is {deviation:.1f}% longer than maximum")
            print(f"   This may be acceptable if content provides extra value")
        
        print(f"\nSCRIPT PREVIEW (first 1000 characters):")
        print("-"*80)
        print(script[:1000] + "..." if len(script) > 1000 else script)
        print("="*80 + "\n")
        
        return script
    
    def generate_script_with_tone(self, tone_analyzer, knowledge_base, prompt, target_minutes=None):
        """
        Generate script with optional tone analyzer + knowledge base.
        
        Args:
            tone_analyzer: Optional dict with full tone profile from tone_analyzer API
            knowledge_base: Dict with 'inspiration_knowledge' and 'document_knowledge'
            prompt: User's script requirements
            target_minutes: Optional target duration
        
        Returns:
            Formatted script with 1-2 sentence paragraphs
        """
        try:
            # ============================================
            # PART 1: BUILD TONE/STYLE INSTRUCTIONS
            # ============================================
            
            if tone_analyzer:
                # Extract all relevant fields
                channel_name = tone_analyzer.get('channel_name', 'the creator')
                
                # Core tone profile
                tone_profile = tone_analyzer.get('tone_profile', {})
                primary_tone = tone_profile.get('primary_tone', 'professional')
                secondary_tones = tone_profile.get('secondary_tones', [])
                formality = tone_profile.get('formality_level', 'conversational')
                energy = tone_profile.get('energy_level', 'moderate')
                humor = tone_profile.get('humor_presence', 'subtle')
                
                # Voice characteristics
                voice_chars = tone_analyzer.get('voice_characteristics', {})
                speaking_style = voice_chars.get('speaking_style', 'clear and direct')
                audience_address = voice_chars.get('audience_address', 'you')
                signature_phrases = voice_chars.get('signature_phrases', [])
                
                # Language patterns
                language = tone_analyzer.get('language_analysis', {})
                vocab_level = language.get('vocabulary_level', 'moderate')
                sentence_length = language.get('average_sentence_length', 'medium')
                technical_terms = language.get('technical_terminology', 'moderate')
                
                # Structural patterns
                structural = tone_analyzer.get('structural_patterns', {})
                opening_style = structural.get('typical_opening', 'engaging hook')
                transition_style = structural.get('transition_style', 'clear and smooth')
                explanation_method = structural.get('explanation_method', 'step-by-step')
                closing_pattern = structural.get('closing_pattern', 'strong call-to-action')
                
                # Guidelines
                guidelines = tone_analyzer.get('script_writing_guidelines', {})
                must_include = guidelines.get('must_include', [])
                must_avoid = guidelines.get('avoid', [])
                
                # AI replication instructions (the master guide)
                ai_instructions = tone_analyzer.get('ai_replication_instructions', '')
                
                # Build comprehensive tone instructions
                tone_instructions = f"""**🎯 CRITICAL: REPLICATE {channel_name.upper()}'S EXACT TONE AND STYLE**

    **MASTER REPLICATION GUIDE:**
    {ai_instructions}

    **TONE PROFILE:**
    - Primary Tone: {primary_tone}
    - Secondary Tones: {', '.join(secondary_tones) if secondary_tones else 'N/A'}
    - Formality: {formality}
    - Energy Level: {energy}
    - Humor: {humor}

    **VOICE & DELIVERY:**
    - Speaking Style: {speaking_style}
    - Address Audience As: {audience_address}
    - Signature Phrases to Use: {', '.join(signature_phrases[:3]) if signature_phrases else 'N/A'}
    {f"  → Naturally incorporate: {', '.join(signature_phrases)}" if signature_phrases else ""}

    **LANGUAGE PATTERNS:**
    - Vocabulary Level: {vocab_level}
    - Sentence Length: {sentence_length}
    - Technical Terminology: {technical_terms}

    **STRUCTURE REQUIREMENTS:**
    - Opening: {opening_style}
    - Transitions: {transition_style}
    - Explanations: {explanation_method}
    - Closing: {closing_pattern}

    **✅ MUST INCLUDE:**
    {chr(10).join(['  • ' + item for item in must_include]) if must_include else '  • Engaging, valuable content'}

    **❌ MUST AVOID:**
    {chr(10).join(['  • ' + item for item in must_avoid]) if must_avoid else '  • Generic or boring content'}

    **📝 FORMATTING RULE:**
    Write in 1-2 sentence paragraphs for maximum readability. Each paragraph = one complete thought. Short and punchy."""
            
            else:
                # Default professional tone (no tone_analyzer provided)
                tone_instructions = """**🎯 DEFAULT PROFESSIONAL YOUTUBE TONE**

    Use a professional, engaging, and conversational tone suitable for YouTube:
    - Clear and direct communication
    - Moderate energy and enthusiasm  
    - Conversational but polished
    - Accessible vocabulary (avoid jargon)
    - Well-structured and organized
    - Friendly and approachable

    **📝 FORMATTING RULE:**
    Write in 1-2 sentence paragraphs for maximum readability. Each paragraph = one complete thought. Short and punchy."""
            
            # ============================================
            # PART 2: BUILD KNOWLEDGE CONTEXT
            # ============================================
            
            inspiration_knowledge = knowledge_base.get('inspiration_knowledge', '')
            document_knowledge = knowledge_base.get('document_knowledge', '')
            
            knowledge_context = f"""**📚 KNOWLEDGE BASE (Use as inspiration and reference, NOT as exact content)**

    **From Inspiration Sources (videos, audio, etc.):**
    {inspiration_knowledge if inspiration_knowledge else 'No inspiration sources provided.'}

    **From Documents (PDFs, images, text inputs):**
    {document_knowledge if document_knowledge else 'No document sources provided.'}

    **⚠️ IMPORTANT - How to Use Knowledge:**
    1. Get ideas for topics to cover
    2. Find interesting facts or statistics to reference
    3. Understand the subject matter deeply
    4. Add credibility with expert insights
    5. **Transform everything into the tone/style specified above**
    6. **DO NOT copy content verbatim - make it your own**"""
            
            # ============================================
            # PART 3: CALCULATE TARGET WORD COUNT
            # ============================================
            
            if target_minutes:
                wpm = 145  # Average speaking pace
                target_words = int(target_minutes * wpm)
                duration_instruction = f"""**⏱️ TARGET DURATION:**
    Approximately {target_words} words ({target_minutes} minutes at 145 WPM)
    - This is a guideline, not a hard limit
    - Prioritize quality and completeness over exact word count"""
            else:
                duration_instruction = "**⏱️ DURATION:** Create a comprehensive script (no specific length requirement)."
            
            # ============================================
            # PART 4: BUILD FINAL GENERATION PROMPT
            # ============================================
            
            generation_prompt = f"""You are an expert YouTube script writer creating a high-quality video script.

    ═══════════════════════════════════════════════════════════════════

    {tone_instructions}

    ═══════════════════════════════════════════════════════════════════

    {knowledge_context}

    ═══════════════════════════════════════════════════════════════════

    **📋 USER REQUEST:**
    {prompt}

    {duration_instruction}

    ═══════════════════════════════════════════════════════════════════

    **🎬 SCRIPT STRUCTURE (Follow this flow):**

    1. **HOOK/OPENING** (First 10-15 seconds)
    → Grab attention immediately with a bold statement, question, or promise
    → Match the opening style specified in tone profile

    2. **INTRODUCTION** (30-45 seconds)
    → Set context and preview what's coming
    → Build credibility and rapport

    3. **MAIN CONTENT** (Bulk of video)
    → Organized into clear sections
    → Smooth transitions between topics
    → Include examples, tips, actionable advice
    → Use the explanation method from tone profile

    4. **CONCLUSION** (30-60 seconds)
    → Summarize key takeaways
    → Reinforce main message

    5. **CALL TO ACTION** (15-30 seconds)
    → Encourage engagement (like, subscribe, comment)
    → Match CTA style from tone profile
    → Make it specific and actionable

    ═══════════════════════════════════════════════════════════════════

    **✍️ CRITICAL FORMATTING RULES:**

    ✅ **DO THIS:**
    - Write in SHORT 1-2 sentence paragraphs
    - Each paragraph = ONE complete thought
    - Use line breaks between ALL paragraphs
    - Keep sentences punchy and engaging
    - Make it easy to read and deliver

    ❌ **DON'T DO THIS:**
    - Long dense paragraphs (never more than 2 sentences)
    - Cramming multiple ideas together
    - Wall of text without breaks
    - Overly complex sentence structures

    **📐 FORMATTING EXAMPLE (CORRECT):**

    This is the opening hook. It grabs attention immediately.

    Here's the first key point. Notice the line break between ideas.

    This is another important concept. Each thought stands alone for maximum impact.

    Now we transition smoothly to the next section. See how readable this is?

    **❌ INCORRECT FORMATTING (DON'T DO THIS):**

    This is a long paragraph that goes on and on without breaks and contains multiple ideas crammed together which makes it hard to read and deliver naturally and this is exactly what we want to avoid in our scripts because it reduces readability and viewer engagement.

    ═══════════════════════════════════════════════════════════════════

    **🎯 YOUR TASK:**

    Write the complete YouTube video script following ALL instructions above:
    - Match the tone/style EXACTLY as specified
    - Use knowledge base for inspiration and facts
    - Follow the structure outlined
    - Apply correct formatting (1-2 sentence paragraphs)
    - Make it engaging, valuable, and actionable

    Begin writing the script now:"""

            # ============================================
            # PART 5: GENERATE SCRIPT
            # ============================================
            
            print(f"\n{'='*80}")
            print(f"GENERATING SCRIPT WITH GEMINI 2.0 FLASH EXP")
            print(f"{'='*80}")
            print(f"Tone: {'✓ Custom (' + tone_analyzer.get('channel_name', 'Unknown') + ')' if tone_analyzer else '✗ Default Professional'}")
            print(f"Target: {target_minutes or 'No specific'} minutes")
            print(f"Knowledge: {'✓ Provided' if (inspiration_knowledge or document_knowledge) else '✗ None'}")
            print(f"{'='*80}\n")
            
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=generation_prompt,
                config={
                    'temperature': 0.8,
                    'top_p': 0.95,
                    'top_k': 40,
                    'max_output_tokens': 8000,
                }
            )
            
            script = response.text.strip()
            
            # ============================================
            # PART 6: POST-PROCESS FOR FORMATTING
            # ============================================
            
            # Ensure proper line breaks between sentences
            script = self._format_script_readability(script)
            
            print(f"\n{'='*80}")
            print(f"✅ SCRIPT GENERATED SUCCESSFULLY")
            print(f"{'='*80}")
            print(f"Length: {len(script):,} characters")
            print(f"Word count: ~{len(script.split()):,} words")
            if target_minutes:
                estimated_minutes = len(script.split()) / 145
                print(f"Estimated duration: ~{estimated_minutes:.1f} minutes")
                print(f"Target was: {target_minutes} minutes")
            print(f"{'='*80}\n")
            
            return script
            
        except Exception as e:
            print(f"❌ Error in generate_script_with_tone: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return f"Error generating script: {str(e)}"

    def _format_script_readability(self, script):
        """
        Post-process script to ensure proper formatting with short paragraphs.
        Enforces 1-2 sentence paragraphs with line breaks.
        """
        lines = script.split('\n')
        formatted_lines = []
        current_paragraph = []
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    formatted_lines.append('')  # Add blank line
                    current_paragraph = []
                continue
            
            # Keep special formatting (headers, timestamps, etc.)
            if (line.startswith(('#', '[', '**', '##', '###')) or 
                line.endswith(':') or 
                line.startswith('- ') or
                line.startswith('• ')):
                
                # Flush current paragraph first
                if current_paragraph:
                    formatted_lines.append(' '.join(current_paragraph))
                    formatted_lines.append('')
                    current_paragraph = []
                
                formatted_lines.append(line)
                formatted_lines.append('')
                continue
            
            # Split into sentences (rough split)
            sentences = []
            temp = line
            for delimiter in ['. ', '! ', '? ']:
                parts = temp.split(delimiter)
                for i, part in enumerate(parts[:-1]):
                    sentences.append(part + delimiter.strip())
                temp = parts[-1]
            if temp:
                sentences.append(temp)
            
            # Add sentences to paragraph, breaking after 2 sentences
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                current_paragraph.append(sentence)
                
                # Break paragraph after 2 sentences
                if len(current_paragraph) >= 2:
                    formatted_lines.append(' '.join(current_paragraph))
                    formatted_lines.append('')  # Blank line between paragraphs
                    current_paragraph = []
        
        # Flush remaining paragraph
        if current_paragraph:
            formatted_lines.append(' '.join(current_paragraph))
            formatted_lines.append('')
        
        return '\n'.join(formatted_lines)

    def modify_script_chat(self, current_script, tone_analyzer, knowledge_base, user_message):
        """
        Modify script via chat while maintaining tone and using knowledge base.
    
        Args:
            current_script: The current script content
            tone_analyzer: Optional tone profile dict
            knowledge_base: Dict with inspiration and document knowledge
            user_message: User's modification request
    
        Returns:
            Modified script or response text
        """
        try:
            # Build tone context
            if tone_analyzer:
                tone_context = f"""**MAINTAIN THIS TONE:**
Channel: {tone_analyzer.get('channel_name', 'Unknown')}
Style: {tone_analyzer.get('tone_profile', {}).get('primary_tone', 'professional')}
Instructions: {tone_analyzer.get('ai_replication_instructions', 'Maintain professional tone')}"""
            else:
                tone_context = "**MAINTAIN professional, engaging YouTube tone**"
        
            # Build knowledge context
            inspiration_knowledge = knowledge_base.get('inspiration', '')
            document_knowledge = knowledge_base.get('documents', '')
        
            knowledge_context = f"""**AVAILABLE KNOWLEDGE (reference if needed):**

**Inspiration Sources:**
{inspiration_knowledge if inspiration_knowledge else 'None provided'}

**Document Sources:**
{document_knowledge if document_knowledge else 'None provided'}"""
        
            # Truncate script if too long (max 15000 chars to fit in prompt)
            truncated_script = current_script[:150000]
            if len(current_script) > 150000:
                truncated_script += "\n\n[... script truncated for length ...]"
        
            modification_prompt = f"""{tone_context}

{knowledge_context}

**CURRENT SCRIPT:**
{truncated_script}

**USER REQUEST:**
{user_message}

**TASK:** Respond to the user's request. You can:
1. Modify the script according to their request
2. Answer questions about the script
3. Provide suggestions or improvements
4. Make sure to give whole script in same format not only changes.
5. Do NOT include any introductory phrases, prefaces, explanations, or meta text. Start your response directly with the requested content.

**IMPORTANT:**
- If modifying the script, maintain the exact tone and style specified above
- Keep the 1-2 sentence paragraph format
- Use knowledge base for reference if needed
- Ensure smooth flow and transitions
- If the user asks a question, answer it clearly
- If they want changes, provide the modified version or specific suggestions

Provide your response:"""

            # ✅ FIX: Use Claude API (_call_claude method) - NOT self.genai
            print(f"\n{'='*60}")
            print(f"MODIFYING SCRIPT VIA CHAT")
            print(f"{'='*60}")
            print(f"User request: {user_message[:100]}...")
            print(f"Script length: {len(current_script):,} chars")
            print(f"Tone: {'✓ Custom (' + tone_analyzer.get('channel_name', 'Unknown') + ')' if tone_analyzer else '✗ Default'}")
            print(f"{'='*60}\n")
        
        # Call Claude API using the existing _call_claude helper method
            response = self._call_claude(
                prompt=modification_prompt,
                max_tokens=8000,
                temperature=0.7
            )
        
            result = response.strip()
        
            print(f"\n{'='*60}")
            print(f"✓ MODIFICATION COMPLETE")
            print(f"{'='*60}")
            print(f"Response length: {len(result):,} characters")
            print(f"Words: ~{len(result.split()):,}")
            print(f"{'='*60}\n")
        
            return result
        
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"❌ Error in modify_script_chat: {str(e)}")
            print(error_trace)
            logger.error(f"Script modification error: {str(e)}\n{error_trace}")
            return f"Error modifying script: {str(e)}"
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _get_default_style(self) -> str:
        """Return default professional style when no personal videos provided"""
        return """Professional YouTube Communication Style:

**Voice & Tone:**
- Clear, conversational, and engaging
- Enthusiastic about the topic without being over-the-top
- Confident but approachable - like a knowledgeable friend explaining something
- Natural pacing that allows ideas to breathe

**Language Patterns:**
- Accessible vocabulary - technical terms explained when used
- Mix of short, punchy sentences and flowing explanations
- Uses "you" to connect directly with viewers
- Asks rhetorical questions to maintain engagement
- Emphasizes key points through repetition and examples

**Teaching Approach:**
- Explains concepts clearly before going deep
- Uses analogies and real-world examples
- Breaks complex ideas into digestible pieces
- Builds on previous points naturally
- Provides context for why things matter

**Unique Elements:**
- Balances education with entertainment
- Makes viewers feel smart for understanding
- Maintains energy throughout without rushing
-- Ends with clear takeaways"""

# Initialize processors for export
# Initialize processors for export
document_processor = DocumentProcessor()
video_processor = VideoProcessor()
script_generator = EnhancedScriptGenerator()
instagram_processor = InstagramProcessor(
    session_file=r"/home/harsh/kretoai/session-aashitapatel17"
)
facebook_processor = FacebookProcessor()  # ADD THIS LINE
audio_processor = AudioProcessor()  # ADD THIS LINE
# tiktok_processor = TikTokProcessor()
image_processor = ImageProcessor()
text_processor = TextProcessor()
# Update __all__ to export new processors
__all__ = [
    'DocumentProcessor',
    'VideoProcessor', 
    'InstagramProcessor',
    'FacebookProcessor',
    'TikTokProcessor',  # NEW
    'ImageProcessor',   # NEW
    'TextProcessor',    # NEW
    'AudioProcessor',
    'EnhancedScriptGenerator',
    'user_data',    
    'UPLOAD_FOLDER',
    'ALLOWED_EXTENSIONS',
    'MAX_CONTENT_LENGTH',
    'document_processor',
    'video_processor',
    'instagram_processor',
    'facebook_processor',
    # 'tiktok_processor',  # NEW
    'image_processor',   # NEW
    'text_processor',    # NEW
    'audio_processor',
    'script_generator',
    'load_whisper_model'
]
