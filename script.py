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
    """Script generator with Claude AI - Fully optimized with precise prompts"""

    def __init__(self):
        # ========================================
        # OPTIMIZED PROMPTS - 60% shorter, more precise
        # ========================================
        
        self.style_analysis_prompt = """Analyze these video transcripts to create a concise style profile for script generation.

Extract only:
1. **Speaking Style**: Tone (formal/casual/energetic), pacing, personality traits
2. **Language Patterns**: Vocabulary level, sentence structure, catchphrases, technical vs simple language
3. **Structure**: How they open videos, transition between points, conclude, use examples
4. **Unique Traits**: What makes this creator distinctive, signature phrases

Be specific and actionable. Avoid generic descriptions.

**Transcripts:**
"""

        self.inspiration_analysis_prompt = """Analyze these videos to extract key insights for content creation.

Extract:
1. **Core Topics**: Main subjects with specific subtopics and key data/statistics
2. **Unique Angles**: Fresh perspectives, unexplored viewpoints, controversial points
3. **Presentation Techniques**: How complex topics are explained, types of examples used
4. **Actionable Information**: Specific tips, solutions to problems, tools/resources mentioned

Focus on insights that can directly inform new content.

**Transcripts:**
"""

        self.document_analysis_prompt = """Extract key knowledge from these documents for YouTube script creation.

Extract:
1. **Core Concepts**: Main themes with supporting evidence and data
2. **Actionable Information**: Step-by-step processes, specific strategies, tools
3. **Knowledge Structure**: How concepts build from basic to advanced
4. **Content Opportunities**: Topics for videos, concepts needing simplification

Focus on practical, usable insights.

**Documents:**
"""

        self.chat_modification_prompt = """Modify this YouTube script based on the user's request. Keep the creator's voice and timestamp structure.

**CURRENT SCRIPT:**
{current_script}

**STYLE GUIDE:**
{style_profile}

**USER REQUEST:**
{user_message}

**RULES:**
- Keep timestamp format: **### [MM:SS-MM:SS]** spoken content
- Maintain creator's tone and phrasing
- No production notes or visual directions
- Apply the user's changes precisely
- Keep word count similar to original

Return the updated script only.
"""

        self.enhanced_script_template = """Create a YouTube script with STRICT timing and timestamp format.

**CREATOR VOICE:**
{style_profile}

**TOPIC KNOWLEDGE:**
{inspiration_summary}

**REFERENCE MATERIAL:**
{document_insights}

**USER REQUEST:**
{user_prompt}

**TIMING REQUIREMENTS:**
- Target: {target_minutes} minutes ({target_seconds} seconds)
- Word count: {word_count_target} words (range: {min_words}-{max_words})
- Speaking pace: 150 words/minute
- COUNT YOUR WORDS and stop at {word_count_target}

**STRUCTURE TIMING:**
- Hook: [00:00-00:{hook_duration:02d}] ({hook_words} words)
- Intro: [00:{hook_duration:02d}-00:{intro_end:02d}] ({intro_words} words)  
- Main: [00:{intro_end:02d}-{main_minutes:02d}:{main_seconds:02d}] ({main_words} words)
- Conclusion: [{main_minutes:02d}:{main_seconds:02d}-{target_minutes:02d}:{target_seconds_remainder:02d}] ({conclusion_words} words)

**FORMAT RULES:**
1. Start with: **## [Your Title]**
2. Immediately follow with hook: **### [00:00-00:XX] Hook, topic name or outrow** Content starts here...
3. NO greetings ("Hey everyone") - start with impact
4. Use continuous timestamps: [00:00-00:15], [00:15-00:32], [00:32-00:50]...
5. Each timestamp = 10-20 seconds (25-50 words)
6. End at [{target_minutes:02d}:{target_seconds_remainder:02d}]
7. NO production notes, NO visual directions, NO tone descriptions
8. Use **bold** only for title (## heading) and timestamps (### heading)
9. Make sure to use Hook, Outrow or if topic then topic name beside timestamp its compulsary.

**CONTENT RULES:**
1. Use creator's natural voice from style profile
2. Integrate document knowledge as authority
3. Use inspiration insights for engagement
4. Explain 'why' and 'how', not just 'what'
5. Make every section valuable

**OUTPUT FORMAT EXAMPLE:**
```
## Your Engaging Title Here

### [00:00-xx:xx] Hook
There will be starting hook of video in this section

### [xx:xx-yy:yy] Introduction
Continue building curiosity. Explain what viewers will learn and why it matters to them personally.

### [yy:yy-zz:zz] Topic name
Transition into first main point. Use creator's natural transitions and speaking style.

[Continue with continuous timestamps until end ]

### [{target_minutes:02d}:{target_seconds_remainder:02d}] Outrow
Final thought and call to action in creator's style.
```

Generate the script now. Count words carefully.
"""

    # ========================================
    # CLAUDE API CALLER
    # ========================================
    
    def _call_claude(self, prompt, max_tokens=15000, temperature=0.7):
        """
        Call Claude API with the given prompt
        Returns the response text
        """
        try:
            print(f"📡 Calling Claude API (max_tokens={max_tokens}, temperature={temperature})...")
            
            message = anthropic_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            print(f"✓ Claude response: {len(response_text)} characters")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Claude API error: {str(e)}")
            print(f"❌ Claude API error: {str(e)}")
            raise

    # ========================================
    # ANALYSIS METHODS
    # ========================================

    def analyze_creator_style(self, personal_transcripts):
        """Analyze creator style from personal videos using Claude - OPTIMIZED"""
        print("\n" + "="*60)
        print("ANALYZING CREATOR STYLE")
        print("="*60)
        
        combined_transcripts = "\n\n---VIDEO SEPARATOR---\n\n".join(personal_transcripts)
        
        # Smart truncation - keep beginning and end
        max_chars = 40000
        if len(combined_transcripts) > max_chars:
            print(f"⚠️ Truncating transcripts from {len(combined_transcripts):,} to {max_chars:,} chars")
            chunk_size = 13000
            combined_transcripts = (
                f"{combined_transcripts[:chunk_size]}\n\n"
                f"[...MIDDLE CONTENT OMITTED FOR BREVITY...]\n\n"
                f"{combined_transcripts[-chunk_size:]}"
            )
        
        try:
            full_prompt = self.style_analysis_prompt + combined_transcripts
            response_text = self._call_claude(
                prompt=full_prompt,
                max_tokens=2000,
                temperature=0.1
            )
            
            if response_text:
                print(f"✓ Style analysis complete: {len(response_text)} characters")
                print(f"✓ Preview: {response_text[:200]}...")
                print("="*60 + "\n")
                return response_text
            else:
                print("❌ Empty response from style analysis")
                return "Could not analyze creator style - empty response"
                
        except Exception as e:
            logger.error(f"Error analyzing creator style: {str(e)}")
            print(f"❌ Style analysis error: {str(e)}")
            return f"Error analyzing creator style: {str(e)}"

    def analyze_inspiration_content(self, inspiration_transcripts):
        """Analyze inspiration content for topic insights using Claude - OPTIMIZED"""
        print("\n" + "="*60)
        print("ANALYZING INSPIRATION CONTENT")
        print("="*60)
        
        combined_transcripts = "\n\n---VIDEO SEPARATOR---\n\n".join(inspiration_transcripts)
        
        # Smart truncation
        max_chars = 40000
        if len(combined_transcripts) > max_chars:
            print(f"⚠️ Truncating transcripts from {len(combined_transcripts):,} to {max_chars:,} chars")
            chunk_size = 13000
            combined_transcripts = (
                f"{combined_transcripts[:chunk_size]}\n\n"
                f"[...MIDDLE CONTENT OMITTED FOR BREVITY...]\n\n"
                f"{combined_transcripts[-chunk_size:]}"
            )
        
        try:
            full_prompt = self.inspiration_analysis_prompt + combined_transcripts
            response_text = self._call_claude(
                prompt=full_prompt,
                max_tokens=2000,
                temperature=0.2
            )
            
            if response_text:
                print(f"✓ Inspiration analysis complete: {len(response_text)} characters")
                print(f"✓ Preview: {response_text[:200]}...")
                print("="*60 + "\n")
                return response_text
            else:
                print("❌ Empty response from inspiration analysis")
                return "Could not analyze inspiration content - empty response"
                
        except Exception as e:
            logger.error(f"Error analyzing inspiration content: {str(e)}")
            print(f"❌ Inspiration analysis error: {str(e)}")
            return f"Error analyzing inspiration content: {str(e)}"

    def analyze_documents(self, document_texts):
        """Analyze uploaded documents for insights using Claude - OPTIMIZED"""
        if not document_texts:
            print("ℹ️ No documents provided for analysis")
            return "No documents provided for analysis."
        
        print("\n" + "="*60)
        print("ANALYZING DOCUMENTS")
        print("="*60)
        
        combined_documents = "\n\n---DOCUMENT SEPARATOR---\n\n".join(document_texts)
        
        # Smart truncation
        max_chars = 45000
        if len(combined_documents) > max_chars:
            print(f"⚠️ Truncating documents from {len(combined_documents):,} to {max_chars:,} chars")
            chunk_size = 15000
            combined_documents = (
                f"{combined_documents[:chunk_size]}\n\n"
                f"[...MIDDLE CONTENT OMITTED FOR BREVITY...]\n\n"
                f"{combined_documents[-chunk_size:]}"
            )
        
        try:
            full_prompt = self.document_analysis_prompt + combined_documents
            response_text = self._call_claude(
                prompt=full_prompt,
                max_tokens=2500,
                temperature=0.2
            )
            
            if response_text:
                print(f"✓ Document analysis complete: {len(response_text)} characters")
                print(f"✓ Preview: {response_text[:200]}...")
                print("="*60 + "\n")
                return response_text
            else:
                print("❌ Empty response from document analysis")
                return "Could not analyze documents - empty response"
                
        except Exception as e:
            logger.error(f"Error analyzing documents: {str(e)}")
            print(f"❌ Document analysis error: {str(e)}")
            return f"Error analyzing documents: {str(e)}"

    # ========================================
    # SCRIPT GENERATION
    # ========================================

    def generate_enhanced_script(self, style_profile, inspiration_summary, 
                                document_insights, user_prompt, target_minutes=None):
        """Generate script with STRICT duration control using Claude - OPTIMIZED"""
        print("\n" + "="*60)
        print("GENERATING ENHANCED SCRIPT WITH STRICT TIMING")
        print("="*60)
        
        # Calculate precise timing parameters
        if target_minutes:
            target_seconds = int(target_minutes * 60)
            words_per_minute = 150
            word_count_target = int(target_minutes * words_per_minute)
            min_words = int(word_count_target * 0.95)
            max_words = int(word_count_target * 1.05)
            
            # Section timing (proportional)
            hook_duration = min(15, int(target_seconds * 0.15))
            intro_end = min(45, int(target_seconds * 0.25))
            main_end = int(target_seconds * 0.85)
            
            # Word allocation per section
            hook_words = int(hook_duration / 60 * words_per_minute)
            intro_words = int((intro_end - hook_duration) / 60 * words_per_minute)
            main_words = int((main_end - intro_end) / 60 * words_per_minute)
            conclusion_words = int((target_seconds - main_end) / 60 * words_per_minute)
        else:
            # Default: 10 minutes
            target_seconds = 600
            word_count_target = 1500
            min_words = 1425
            max_words = 1575
            hook_duration = 15
            intro_end = 45
            main_end = 510
            hook_words = 38
            intro_words = 75
            main_words = 1275
            conclusion_words = 112
            target_minutes = 10
        
        target_minutes_int = int(target_minutes)
        target_seconds_remainder = target_seconds % 60
        main_minutes = main_end // 60
        main_seconds = main_end % 60
        
        print(f"📊 Target Duration: {target_minutes} minutes ({target_seconds} seconds)")
        print(f"📊 Target Words: {word_count_target} (range: {min_words}-{max_words})")
        print(f"📊 Speaking Pace: 150 words/minute")
        
        # Format the enhanced prompt
        enhanced_prompt = self.enhanced_script_template.format(
            style_profile=style_profile,
            inspiration_summary=inspiration_summary,
            document_insights=document_insights,
            user_prompt=user_prompt,
            target_minutes=target_minutes_int,
            target_seconds=target_seconds,
            word_count_target=word_count_target,
            min_words=min_words,
            max_words=max_words,
            hook_duration=hook_duration,
            intro_end=intro_end,
            main_minutes=main_minutes,
            main_seconds=main_seconds,
            conclusion_words=conclusion_words,
            hook_words=hook_words,
            intro_words=intro_words,
            main_words=main_words,
            target_seconds_remainder=target_seconds_remainder
        )
        
        try:
            # Generate script with Claude
            script = self._call_claude(
                prompt=enhanced_prompt,
                max_tokens=word_count_target + 1000,  # Buffer for formatting
                temperature=0.7
            )
            
            if script:
                script = script.strip()
                
                # Validate word count
                actual_words = len(script.split())
                actual_duration_seconds = (actual_words / 150) * 60
                actual_duration_minutes = actual_duration_seconds / 60
                
                print(f"\n{'='*60}")
                print(f"SCRIPT GENERATION COMPLETE")
                print(f"{'='*60}")
                print(f"📝 Generated Words: {actual_words}")
                print(f"🎯 Target Words: {word_count_target} (range: {min_words}-{max_words})")
                print(f"⏱️ Estimated Duration: {actual_duration_minutes:.1f} minutes")
                print(f"🎯 Target Duration: {target_minutes} minutes")
                
                # Smart truncation if needed
                if actual_words > max_words:
                    print(f"⚠️ Script too long! Truncating from {actual_words} to {word_count_target} words...")
                    words = script.split()
                    truncated_script = ' '.join(words[:word_count_target])
                    truncated_script += f"\n\n### [{target_minutes_int:02d}:{target_seconds_remainder:02d}]\nThanks for watching!"
                    script = truncated_script
                    actual_words = len(script.split())
                    print(f"✓ Script truncated to {actual_words} words")
                elif actual_words < min_words:
                    print(f"⚠️ Script shorter than minimum ({actual_words} < {min_words})")
                    print(f"   Consider regenerating for better timing")
                else:
                    print(f"✓ Word count within target range")
                
                print(f"{'='*60}\n")
                print(f"Script Preview (first 500 chars):")
                print(f"{'-'*60}")
                print(script[:500] + "..." if len(script) > 500 else script)
                print(f"{'='*60}\n")
                
                return script
            else:
                print("❌ Empty response from script generation")
                return "Error: Could not generate script - empty response"
                
        except Exception as e:
            logger.error(f"Error generating script: {str(e)}")
            print(f"❌ Script generation error: {str(e)}")
            return f"Error generating script: {str(e)}"

    # ========================================
    # SCRIPT MODIFICATION (CHAT)
    # ========================================

    def modify_script_chat(self, current_script, style_profile, topic_insights, 
                          document_insights, user_message):
        """Modify script via chat using Claude - OPTIMIZED"""
        print("\n" + "="*60)
        print("MODIFYING SCRIPT VIA CHAT")
        print("="*60)
        
        # Format chat modification prompt
        chat_prompt = self.chat_modification_prompt.format(
            current_script=current_script,
            style_profile=style_profile,
            user_message=user_message
        )
        
        print(f"📝 Chat prompt: {len(chat_prompt):,} characters")
        print(f"💬 User request: {user_message[:100]}...")
        
        try:
            # Call Claude for modification
            modified_script = self._call_claude(
                prompt=chat_prompt,
                max_tokens=3000,
                temperature=0.6
            )
            
            if modified_script:
                modified_script = modified_script.strip()
                actual_words = len(modified_script.split())
                
                print(f"\n{'='*60}")
                print(f"SCRIPT MODIFICATION COMPLETE")
                print(f"{'='*60}")
                print(f"✓ Modified script: {len(modified_script):,} characters")
                print(f"✓ Word count: {actual_words:,} words")
                print(f"{'='*60}\n")
                
                return modified_script
            else:
                print("❌ Empty response from modification")
                return "Could not modify script - empty response"
                
        except Exception as e:
            logger.error(f"Error modifying script: {str(e)}")
            print(f"❌ Script modification error: {str(e)}")
            return f"Error modifying script: {str(e)}"
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
