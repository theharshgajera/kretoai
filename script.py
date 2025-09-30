from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import time
import re
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

# Document processing imports
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import tempfile
import fitz  # PyMuPDF for better PDF handling
from pathlib import Path

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here")
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# File upload configuration
UPLOAD_FOLDER = r'D:\poppy AI\kretoai\tempfolder'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 50MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simplified in-memory storage - only what's needed for all-in-one approach
user_data = defaultdict(lambda: {
    'chat_sessions': {},
    'current_script': None
})

class DocumentProcessor:
    """Advanced document processing with multiple format support"""
    
    def __init__(self):
        self.max_chars = 100000  # Maximum characters to process per document
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF using multiple methods for reliability"""
        text = ""
        
        try:
            # Try PyMuPDF first (better for complex PDFs)
            doc = fitz.open(file_path)
            print(f"Extracting text from PDF using PyMuPDF: {file_path}")
            for page_num in range(doc.page_count):
                page = doc[page_num]
                text += page.get_text()
            doc.close()
            
            if len(text.strip()) > 50:  # If we got good text
                print(f"Successfully extracted {len(text)} characters from PDF using PyMuPDF.")
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
            print(f"PyMuPDF extraction failed for {file_path}: {e}")
        
        try:
            # Fallback to PyPDF2
            print(f"Falling back to PyPDF2 for PDF: {file_path}")
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            print(f"PyPDF2 extraction failed for {file_path}: {e}")
            return None
        
        if len(text.strip()) > 50:
            print(f"Successfully extracted {len(text)} characters from PDF using PyPDF2.")
        else:
            print(f"Extracted text from PDF is too short: {len(text)} characters.")
        return text if len(text.strip()) > 50 else None
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX files"""
        try:
            print(f"Extracting text from DOCX: {file_path}")
            doc = docx.Document(file_path)
            text = []
            
            # Extract from paragraphs
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            
            # Extract from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text.append(cell.text)
            
            full_text = '\n'.join(text)
            print(f"Successfully extracted {len(full_text)} characters from DOCX.")
            return full_text
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            print(f"DOCX extraction failed for {file_path}: {e}")
            return None
    
    def extract_text_from_doc(self, file_path):
        """Extract text from legacy DOC files (basic support)"""
        try:
            print(f"Extracting text from DOC: {file_path}")
            # Try using python-docx (limited DOC support)
            doc = docx.Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text.append(paragraph.text)
            full_text = '\n'.join(text)
            print(f"Successfully extracted {len(full_text)} characters from DOC.")
            return full_text
        except Exception as e:
            logger.warning(f"DOC extraction failed, file might need conversion: {e}")
            print(f"DOC extraction failed for {file_path}: {e}")
            return None
    
    def extract_text_from_txt(self, file_path):
        """Extract text from TXT files with encoding detection"""
        encodings = ['utf-8', 'utf-16', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                print(f"Trying to extract text from TXT with encoding {encoding}: {file_path}")
                with open(file_path, 'r', encoding=encoding) as file:
                    full_text = file.read()
                print(f"Successfully extracted {len(full_text)} characters from TXT with {encoding}.")
                return full_text
            except UnicodeDecodeError:
                print(f"UnicodeDecodeError with {encoding} for {file_path}, trying next.")
                continue
            except Exception as e:
                logger.error(f"TXT extraction failed: {e}")
                print(f"TXT extraction failed for {file_path}: {e}")
                return None
        
        print(f"All encodings failed for TXT: {file_path}")
        return None
    
    def process_document(self, file_path, filename):
        """Process document and extract text with metadata"""
        print(f"Starting document processing for {filename} at {file_path}")
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        extraction_methods = {
            'pdf': self.extract_text_from_pdf,
            'docx': self.extract_text_from_docx,
            'doc': self.extract_text_from_doc,
            'txt': self.extract_text_from_txt
        }
        
        extract_method = extraction_methods.get(file_ext)
        if not extract_method:
            print(f"Unsupported file format: {file_ext} for {filename}")
            return {"error": "Unsupported file format", "text": None, "stats": None}
        
        try:
            text = extract_method(file_path)
            
            if not text or len(text.strip()) < 50:
                print(f"No meaningful text extracted from {filename}: {len(text) if text else 0} characters.")
                return {"error": "Could not extract meaningful text from document", "text": None, "stats": None}
            
            # Truncate if too long
            if len(text) > self.max_chars:
                print(f"Truncating document {filename} from {len(text)} to {self.max_chars} characters.")
                text = text[:self.max_chars] + "\n\n[Document truncated for processing...]"
            
            stats = self._calculate_document_stats(text, filename)
            print(f"Document {filename} processed successfully. Stats: {stats}")
            
            return {
                "error": None,
                "text": text,
                "stats": stats,
                "filename": filename,
                "file_type": file_ext
            }
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            print(f"Error processing document {filename}: {str(e)}")
            return {"error": f"Error processing document: {str(e)}", "text": None, "stats": None}
    
    def _calculate_document_stats(self, text, filename):
        """Calculate document statistics"""
        if not text:
            stats = {
                'char_count': 0,
                'word_count': 0,
                'page_estimate': 0,
                'read_time': 0
            }
            print(f"Stats for {filename}: {stats}")
            return stats
        
        char_count = len(text)
        word_count = len(text.split())
        page_estimate = max(1, word_count // 250)  # ~250 words per page
        read_time = max(1, word_count // 200)  # ~200 words per minute
        
        stats = {
            'char_count': char_count,
            'word_count': word_count,
            'page_estimate': page_estimate,
            'read_time': read_time,
            'filename': filename
        }
        print(f"Calculated stats for {filename}: {stats}")
        return stats

class VideoProcessor:
    """Enhanced video processor that handles both YouTube URLs and local video files"""
    
    def __init__(self):
        self.rate_limit_delay = 2
        self.last_api_call = 0
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        self.max_video_size = 500 * 1024 * 1024  # 500MB max
        self.max_duration = 3600  # 1 hour max
    
    def is_supported_video_format(self, filename):
        """Check if the video format is supported"""
        is_supported = Path(filename).suffix.lower() in self.supported_video_formats
        print(f"Checking video format for {filename}: {'Supported' if is_supported else 'Unsupported'}")
        return is_supported
    
    def validate_video_file(self, file_path):
        """Validate video file size and basic properties"""
        try:
            file_size = os.path.getsize(file_path)
            print(f"Video file size: {file_size} bytes for {file_path}")
            if file_size > self.max_video_size:
                print(f"Video too large: {file_size} > {self.max_video_size}")
                return False, f"Video file too large. Max size: {self.max_video_size // (1024*1024)}MB"
            
            # Check duration using moviepy
            print(f"Loading video to check duration: {file_path}")
            video = mp.VideoFileClip(file_path)
            duration = video.duration
            video.close()
            print(f"Video duration: {duration} seconds")
            
            if duration > self.max_duration:
                print(f"Video too long: {duration} > {self.max_duration}")
                return False, f"Video too long. Max duration: {self.max_duration // 60} minutes"
            
            return True, None
        except Exception as e:
            print(f"Validation failed for {file_path}: {str(e)}")
            return False, f"Invalid video file: {str(e)}"
    
    def extract_audio_from_video(self, video_path, output_path=None):
        """Extract audio from video file"""
        try:
            if output_path is None:
                output_path = os.path.join(tempfile.gettempdir(), f"temp_audio_{os.getpid()}.wav")
            
            print(f"Extracting audio from {video_path} to {output_path}")
            # Use moviepy to extract audio
            video = mp.VideoFileClip(video_path)
            audio = video.audio
            audio.write_audiofile(output_path, verbose=False, logger=None)
            audio.close()
            video.close()
            print(f"Audio extracted successfully to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            print(f"Audio extraction failed for {video_path}: {str(e)}")
            return None
    
    def transcribe_audio_with_speech_recognition(self, audio_path):
        """Transcribe audio using speech_recognition library"""
        try:
            print(f"Starting transcription for audio: {audio_path}")
            recognizer = sr.Recognizer()
            
            with sr.AudioFile(audio_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source, duration=1)
                audio_data = recognizer.record(source)
            
            # Try Google Speech Recognition (requires internet)
            try:
                print("Trying Google Speech Recognition...")
                text = recognizer.recognize_google(audio_data)
                print(f"Transcription successful: {len(text)} characters")
                return text
            except sr.RequestError:
                print("Google Speech Recognition failed, falling back to Sphinx...")
                # Fallback to offline recognition if available
                try:
                    text = recognizer.recognize_sphinx(audio_data)
                    print(f"Offline transcription successful: {len(text)} characters")
                    return text
                except sr.RequestError:
                    print("All transcription methods failed.")
                    return None
                    
        except Exception as e:
            logger.error(f"Speech recognition failed: {str(e)}")
            print(f"Speech recognition failed for {audio_path}: {str(e)}")
            return None
    
    def transcribe_audio_with_gemini(self, audio_path):
        """Transcribe audio using Gemini API - more reliable for long videos"""
        try:
            print(f"Starting Gemini transcription for audio: {audio_path}")
            
            # Upload audio file to Gemini
            print("Uploading audio file to Gemini...")
            audio_file = genai.upload_file(audio_path)
            print(f"Audio uploaded: {audio_file.name}")
            
            # Wait for processing
            print("Waiting for Gemini to process audio...")
            while audio_file.state.name == "PROCESSING":
                time.sleep(2)
                audio_file = genai.get_file(audio_file.name)
            
            if audio_file.state.name == "FAILED":
                print("Gemini audio processing failed")
                return None
            
            print("Audio processed successfully, starting transcription...")
            
            # Create transcription prompt
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = """Please transcribe this audio file completely and accurately. 
            
            Instructions:
            - Transcribe everything that is spoken
            - Maintain the natural flow and structure
            - Include all content, even if it's lengthy
            - Do not summarize - provide the complete transcription
            - Fix obvious verbal mistakes but keep the authentic speaking style
            
            Provide only the transcription, no additional commentary."""
            
            # Generate transcription
            response = model.generate_content(
                [prompt, audio_file],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=8000
                )
            )
            
            # Cleanup uploaded file
            try:
                genai.delete_file(audio_file.name)
                print("Cleaned up uploaded audio from Gemini")
            except:
                pass
            
            if response.text and len(response.text.strip()) > 50:
                print(f"Gemini transcription successful: {len(response.text)} characters")
                return response.text
            else:
                print(f"Gemini transcription too short: {len(response.text) if response.text else 0} characters")
                return None
                
        except Exception as e:
            logger.error(f"Gemini transcription failed: {str(e)}")
            print(f"Gemini transcription failed: {str(e)}")
            return None
    
    def process_local_video_file(self, video_path, filename):
        """Process local video file to extract transcript - UPDATED VERSION"""
        try:
            print(f"Starting local video processing for {filename} at {video_path}")
            
            # Validate video file
            is_valid, error_msg = self.validate_video_file(video_path)
            if not is_valid:
                print(f"Validation failed: {error_msg}")
                return {"error": error_msg, "transcript": None, "stats": None}
            
            # Extract audio
            temp_audio_path = self.extract_audio_from_video(video_path)
            if not temp_audio_path:
                print("Audio extraction failed.")
                return {"error": "Failed to extract audio from video", "transcript": None, "stats": None}
            
            try:
                # Try Gemini transcription first (recommended)
                print("Attempting Gemini-based transcription...")
                transcript_text = self.transcribe_audio_with_gemini(temp_audio_path)
                
                # Fallback to speech_recognition if Gemini fails
                if not transcript_text or len(transcript_text.strip()) < 50:
                    print("Gemini transcription insufficient, trying speech_recognition...")
                    transcript_text = self.transcribe_audio_with_speech_recognition(temp_audio_path)
                
                if not transcript_text or len(transcript_text.strip()) < 50:
                    print(f"No meaningful transcript: {len(transcript_text) if transcript_text else 0} characters")
                    return {"error": "Could not extract meaningful transcript from video", "transcript": None, "stats": None}
                
                stats = self._calculate_transcript_stats(transcript_text)
                stats['filename'] = filename
                stats['source_type'] = 'local_video'
                print(f"Local video processed successfully. Stats: {stats}")
                print("\n\n" + "="*40)
                print(f"EXTRACTED TRANSCRIPT FOR LOCAL VIDEO: {filename}")
                print("="*40)
                print(transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text)
                print("="*40 + "\n\n")
                
                return {
                    "error": None,
                    "transcript": transcript_text,
                    "stats": stats,
                    "source": filename
                }
                
            finally:
                # Clean up temp audio file
                try:
                    if os.path.exists(temp_audio_path):
                        print(f"Cleaning up temp audio: {temp_audio_path}")
                        os.remove(temp_audio_path)
                except:
                    print(f"Failed to clean up {temp_audio_path}")
                    pass
                    
        except Exception as e:
            logger.error(f"Local video processing error: {str(e)}")
            print(f"Local video processing error for {filename}: {str(e)}")
            return {"error": f"Error processing video: {str(e)}", "transcript": None, "stats": None}
        
    def process_video_content(self, source, source_type='youtube'):
        """
        Universal video processing method
        source: Either YouTube URL or local file path
        source_type: 'youtube' or 'local'
        """
        print(f"Processing video content: {source} (type: {source_type})")
        if source_type == 'youtube':
            if not self.validate_youtube_url(source):
                print(f"Invalid YouTube URL: {source}")
                return {"error": "Invalid YouTube URL format", "transcript": None, "stats": None}
            return self.extract_transcript_details(source)
        
        elif source_type == 'local':
            if not self.is_supported_video_format(source):
                print(f"Unsupported video format: {source}")
                return {"error": "Unsupported video format", "transcript": None, "stats": None}
            filename = os.path.basename(source)
            return self.process_local_video_file(source, filename)
        
        else:
            print(f"Unknown source type: {source_type}")
            return {"error": "Unknown source type", "transcript": None, "stats": None}
    
    # Keep all existing YouTube methods unchanged
    def extract_video_id(self, youtube_url):
        print(f"Extracting video ID from URL: {youtube_url}")
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                video_id = match.group(1)
                print(f"Extracted video ID: {video_id}")
                return video_id
        print("No video ID found.")
        return None
    
    def validate_youtube_url(self, url):
        print(f"Validating YouTube URL: {url}")
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
        try:
            parsed_url = urlparse(url)
            is_valid = any(domain in parsed_url.netloc for domain in youtube_domains)
            print(f"URL validation: {'Valid' if is_valid else 'Invalid'}")
            return is_valid
        except:
            print("URL parsing failed.")
            return False
    
    def rate_limit_wait(self):
        print("Checking rate limit...")
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_call
            print(f"Rate limit wait: sleeping for {sleep_time} seconds")
            time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def extract_transcript_details(self, youtube_video_url, max_retries=3, retry_delay=2):
        """Extract transcript with robust error handling"""
        print(f"Starting transcript extraction for YouTube URL: {youtube_video_url}")
        self.rate_limit_wait()
        
        video_id = self.extract_video_id(youtube_video_url)
        if not video_id:
            print("Invalid URL format.")
            return {"error": "Invalid YouTube URL format", "transcript": None, "stats": None}
        
        for attempt in range(max_retries):
            print(f"Attempt {attempt + 1}/{max_retries} to fetch transcript.")
            try:
                if attempt > 0:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"Retrying transcript extraction in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                
                ytt_api = YouTubeTranscriptApi()
                
                try:
                    print("Trying direct fetch...")
                    fetched_transcript = ytt_api.fetch(video_id, languages=['en', 'en-US', 'en-GB'])
                    transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
                    
                    if len(transcript_text.strip()) >= 50:
                        stats = self._calculate_transcript_stats(transcript_text)
                        logger.info(f"Direct fetch successful - {stats['char_count']} characters")
                        print(f"Direct fetch successful: {stats['char_count']} characters")
                        print("\n\n" + "="*40)
                        print(f"EXTRACTED TRANSCRIPT FOR YOUTUBE VIDEO: {youtube_video_url}")
                        print("="*40)
                        print(transcript_text)
                        print("="*40 + "\n\n")
                        return {"error": None, "transcript": transcript_text, "stats": stats}
                        
                except NoTranscriptFound:
                    logger.info("Direct fetch failed, trying transcript list method")
                    print("Direct fetch failed, trying list method.")
                    pass
                
                try:
                    print("Fetching transcript list...")
                    transcript_list = ytt_api.list(video_id)
                    
                    transcript = None
                    try:
                        transcript = transcript_list.find_transcript(['en', 'en-US', 'en-GB'])
                        logger.info("Found English transcript")
                        print("Found English transcript.")
                    except NoTranscriptFound:
                        try:
                            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
                            logger.info("Found manually created English transcript")
                            print("Found manually created English transcript.")
                        except NoTranscriptFound:
                            try:
                                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                                logger.info("Found auto-generated English transcript")
                                print("Found auto-generated English transcript.")
                            except NoTranscriptFound:
                                available_transcripts = list(transcript_list)
                                if available_transcripts:
                                    transcript = available_transcripts[0]
                                    logger.info(f"Using first available transcript: {transcript.language}")
                                    print(f"Using first available transcript: {transcript.language}")
                                else:
                                    print("No transcripts available.")
                                    return {"error": "No transcripts available for this video", "transcript": None, "stats": None}
                    
                    if not transcript:
                        print("No suitable transcript found.")
                        return {"error": "No suitable transcript found", "transcript": None, "stats": None}
                    
                    fetched_transcript = transcript.fetch()
                    transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
                    
                    if len(transcript_text.strip()) < 50:
                        print("Transcript too short.")
                        return {"error": "Transcript too short or incomplete", "transcript": None, "stats": None}
                    
                    stats = self._calculate_transcript_stats(transcript_text)
                    logger.info(f"Transcript extraction successful - {stats['char_count']} characters, {stats['word_count']} words")
                    print(f"Transcript extraction successful: {stats['char_count']} characters, {stats['word_count']} words")
                    print("\n\n" + "="*40)
                    print(f"EXTRACTED TRANSCRIPT FOR YOUTUBE VIDEO: {youtube_video_url}")
                    print("="*40)
                    print(transcript_text)
                    print("="*40 + "\n\n")
                    return {"error": None, "transcript": transcript_text, "stats": stats}
                    
                except Exception as inner_e:
                    logger.error(f"Inner exception during transcript list processing: {str(inner_e)}")
                    print(f"Inner exception: {str(inner_e)}")
                    if attempt == max_retries - 1:
                        return {"error": f"Could not access transcript list - {str(inner_e)}", "transcript": None, "stats": None}
                    continue

            except VideoUnavailable:
                print("Video unavailable.")
                return {"error": "Video is unavailable, private, or doesn't exist", "transcript": None, "stats": None}
            except TranscriptsDisabled:
                print("Transcripts disabled.")
                return {"error": "Transcripts are disabled for this video", "transcript": None, "stats": None}
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                
                if "quota" in error_msg or "rate" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        logger.warning(f"Rate limit hit. Waiting {retry_delay * (attempt + 2)} seconds before retry...")
                        print(f"Rate limit hit, waiting...")
                        continue
                    else:
                        print("Rate limit exceeded.")
                        return {"error": "API rate limit exceeded. Please try again in a few minutes", "transcript": None, "stats": None}
                elif "403" in error_msg or "forbidden" in error_msg:
                    print("Access forbidden.")
                    return {"error": "Access forbidden. Video might be private or restricted", "transcript": None, "stats": None}
                elif "404" in error_msg:
                    print("Video not found.")
                    return {"error": "Video not found. Please check the URL", "transcript": None, "stats": None}
                elif "blocked" in error_msg or "ipblocked" in error_msg:
                    print("IP blocked.")
                    return {"error": "IP address blocked by YouTube. Try using a VPN or proxy", "transcript": None, "stats": None}
                elif attempt == max_retries - 1:
                    print("Failed after max retries.")
                    return {"error": f"Error fetching transcript after {max_retries} attempts: {str(e)}", "transcript": None, "stats": None}
        
        print("Failed after all attempts.")
        return {"error": "Failed to fetch transcript after multiple attempts", "transcript": None, "stats": None}
    
    def _calculate_transcript_stats(self, transcript_text):
        if not transcript_text:
            stats = {
                'char_count': 0,
                'word_count': 0,
                'estimated_duration': 0,
                'estimated_read_time': 0
            }
            print(f"Transcript stats: {stats}")
            return stats
        
        char_count = len(transcript_text)
        word_count = len(transcript_text.split())
        estimated_duration = max(1, word_count // 150)
        estimated_read_time = max(1, word_count // 200)
        
        stats = {
            'char_count': char_count,
            'word_count': word_count,
            'estimated_duration': estimated_duration,
            'estimated_read_time': estimated_read_time
        }
        print(f"Calculated transcript stats: {stats}")
        return stats

class EnhancedScriptGenerator:
    """Script generator with advanced analysis"""
    
    def __init__(self):
        self.style_analysis_prompt = """
        You are an expert YouTube content analyst. Analyze the following transcripts from the creator's personal videos to create a comprehensive style profile.

        Focus on identifying:

        **VOICE & TONE CHARACTERISTICS:**
        - Speaking style (conversational, formal, energetic, calm, etc.)
        - Emotional tone and energy levels
        - Use of humor, sarcasm, or specific personality traits
        - Level of enthusiasm and passion
        - Pacing and rhythm patterns
        
        **LANGUAGE PATTERNS:**
        - Vocabulary complexity and word choices
        - Sentence structure preferences (short/long, simple/complex)
        - Catchphrases, repeated expressions, or signature sayings
        - Use of technical jargon vs. simple explanations
        - Storytelling approach and narrative style
        - Transition phrases and connection words
        
        **CONTENT STRUCTURE & FLOW:**
        - How they introduce topics and hook viewers
        - Transition techniques between sections
        - How they build up to main points
        - Conclusion and call-to-action styles
        - Use of examples, analogies, and explanations
        - Information presentation patterns
        
        **ENGAGEMENT TECHNIQUES:**
        - How they ask questions to audience
        - Interactive elements and audience engagement
        - Use of personal stories and experiences
        - How they handle complex topics
        - Teaching and explanation methodology
        - Retention strategies used
        
        **UNIQUE CREATOR CHARACTERISTICS:**
        - What makes this creator distinctive
        - Their unique perspective or angle
        - Personal brand elements
        - Values and beliefs that come through
        - Specific expertise areas and how they showcase them
        - Content themes and recurring topics

        **KEY INSIGHTS FOR SCRIPT GENERATION:**
        - Most effective hooks and openings used
        - Common content structures that work well
        - Signature explanations or teaching methods
        - Audience connection techniques
        - Call-to-action patterns

        Provide a detailed, actionable style profile that captures the creator's authentic voice for script generation.

        **Creator's Personal Video Transcripts:**
        """
        
        self.inspiration_analysis_prompt = """
        You are an expert content strategist and topic analyst. Analyze these inspiration video transcripts to extract valuable insights and identify key topics with detailed breakdowns.

        Extract and organize:

        **CORE TOPICS & DETAILED INSIGHTS:**
        - Main subject matters with specific subtopics
        - Key points and arguments presented
        - Data, statistics, and factual claims
        - Expert opinions and industry insights
        - Trending discussions and current debates
        - Evergreen vs. timely content themes
        
        **CONTENT IDEAS & CREATIVE ANGLES:**
        - Unique perspectives and fresh takes on topics
        - Creative approaches to common subjects
        - Unexplored angles or missing viewpoints
        - Potential spin-offs and related topics
        - Cross-topic connection opportunities
        - Controversial or debate-worthy points
        
        **STORYTELLING & PRESENTATION TECHNIQUES:**
        - Narrative structures and story arcs used
        - How complex topics are simplified
        - Types of examples and case studies used
        - Visual or conceptual metaphors
        - Emotional appeals and connection methods
        - Pacing and information delivery patterns
        
        **VALUABLE INSIGHTS & ACTIONABLE INFORMATION:**
        - Specific tips, tricks, and how-to steps
        - Common problems and detailed solution approaches
        - Industry best practices mentioned
        - Tools, resources, and recommendations
        - Success stories and failure case studies
        - Expert advice and professional insights
        
        **TOPIC-SPECIFIC MAIN POINTS BREAKDOWN:**
        For each major topic discussed, provide:
        - Core concept explanation
        - Key supporting arguments
        - Practical applications mentioned
        - Common misconceptions addressed
        - Advanced concepts introduced
        - Related subtopics worth exploring
        
        **CONTENT GAPS & OPPORTUNITIES:**
        - Topics that could be expanded upon
        - Alternative viewpoints not covered
        - Beginner vs. advanced treatment opportunities
        - Updated information or fresh perspectives needed
        - Underexplored subtopics with potential

        Provide a comprehensive analysis that captures both the content insights and the presentation methods for creating informed, original content.

        **Inspiration Video Transcripts:**
        """

        self.document_analysis_prompt = """
        You are an expert content analyst specializing in document comprehension and insight extraction. Analyze the following document content to extract key insights, main points, and actionable information that can inform YouTube script generation.

        Focus on identifying:

        **CORE CONCEPTS & MAIN THEMES:**
        - Primary topics and subject areas covered
        - Key concepts and definitions
        - Central arguments and thesis points
        - Supporting evidence and data
        - Expert opinions and authoritative insights
        
        **ACTIONABLE INFORMATION:**
        - Step-by-step processes and procedures
        - Specific tips, strategies, and recommendations
        - Tools, resources, and methodologies mentioned
        - Best practices and proven approaches
        - Case studies and real-world examples
        
        **KNOWLEDGE STRUCTURE:**
        - Logical flow of information
        - How concepts build upon each other
        - Prerequisites and foundational knowledge needed
        - Advanced concepts and expert-level insights
        - Practical applications and implementations
        
        **CONTENT OPPORTUNITIES FOR VIDEO SCRIPTS:**
        - Main points that could become video topics
        - Complex concepts that need simplification
        - Practical demonstrations or tutorials possible
        - Controversial or debate-worthy points
        - Current vs. outdated information
        - Gaps that could be filled with additional research
        
        **AUDIENCE VALUE PROPOSITIONS:**
        - What viewers would learn or gain
        - Problems this content helps solve
        - Skills or knowledge they would acquire
        - Practical benefits and outcomes
        - Target audience level (beginner/intermediate/advanced)

        Extract the most valuable insights that could inform comprehensive, educational YouTube content creation.

        **Document Content:**
        """

        self.enhanced_script_template = """
        You are an expert YouTube script writer creating a professional, engaging script based on comprehensive content analysis including the creator's style, topic insights, and document knowledge.

        **CREATOR'S AUTHENTIC STYLE PROFILE:**
        {style_profile}

        **TOPIC INSIGHTS FROM INSPIRATION CONTENT:**
        {inspiration_summary}

        **DOCUMENT KNOWLEDGE BASE:**
        {document_insights}

        **USER'S SPECIFIC REQUEST:**
        {user_prompt}

        **TARGET DURATION:** {target_duration}

        **ENHANCED SCRIPT GENERATION INSTRUCTIONS:**

        Create a complete, production-ready YouTube script that:

        1. **MAINTAINS AUTHENTIC VOICE:** Use the creator's natural speaking style, vocabulary, and personality traits identified in the style profile.

        2. **ADHERES TO TARGET DURATION:** {duration_instruction}

        3. **INTEGRATES DOCUMENT KNOWLEDGE STRATEGICALLY:**
           - Use document insights as authoritative foundation
           - Incorporate specific data, facts, and expert knowledge
           - Reference key concepts and methodologies from documents
           - Build upon documented best practices and proven approaches
           
        4. **LEVERAGES INSPIRATION INSIGHTS:**
           - Include trending discussions and current debates
           - Use successful presentation techniques identified
           - Apply proven engagement strategies
           - Reference industry insights and expert opinions

        5. **FOLLOWS PROFESSIONAL STRUCTURE:**
           - **Hook (0-15 seconds):** Attention-grabbing opening with specific value promise
           - **Introduction (15-45 seconds):** Topic setup with authority establishment
           - **Main Content Sections:** Well-structured body with clear progression
           - **Document-Based Authority:** Weave in expert knowledge naturally
           - **Practical Applications:** Include actionable takeaways
           - **Conclusion:** Strong summary with clear next steps

        6. **ENSURES COMPREHENSIVE COVERAGE:**
           - Address the topic from multiple angles identified in documents
           - Include both foundational and advanced concepts appropriately
           - Provide practical examples and real-world applications
           - Reference credible sources and expert insights
           - Balance theory with actionable advice

        7. **MAINTAINS ENGAGEMENT:**
           - Use the creator's proven engagement techniques
           - Include questions, interactions, and retention hooks
           - Apply storytelling methods that resonate
           - Incorporate appropriate humor or personality elements

        **OUTPUT FORMAT:**
        
        # [COMPELLING VIDEO TITLE]
        
        ## CONTENT FOUNDATION
        **Document Authority:** [Key document insights being leveraged]
        **Topic Relevance:** [Why this matters now based on inspiration analysis]
        **Creator Angle:** [Unique perspective based on style profile]
        
        ## HOOK (0-15 seconds)
        [Attention-grabbing opening with authority and promise]
        **[Production Note: Tone, visual, and delivery guidance]**
        
        ## INTRODUCTION (15-45 seconds)  
        [Authority establishment with document-backed credibility]
        **[Expert Insight: Specific fact or data from documents]**
        
        ## MAIN CONTENT
        
        ### Section 1: [Title] (Timing: X:XX - X:XX)
        [Document-informed content with creator's authentic delivery]
        **[Authority Point: Specific expert knowledge from documents]**
        **[Actionable Takeaway: Practical application]**
        **[Production Note: Visual aids, emphasis cues]**
        
        ### Section 2: [Title] (Timing: X:XX - X:XX)
        [Continue with comprehensive, well-researched content]
        **[Expert Validation: Supporting evidence from documents]**
        **[Real-World Application: How viewers implement this]**
        
        [Continue for all main sections...]
        
        ## CONCLUSION (Last 30-60 seconds)
        [Strong summary with document-backed authority and clear next steps]
        **[Final Authority Statement: Key expert insight that reinforces value]**
        
        ---
        
        **PRODUCTION NOTES:**
        - Visual timeline and supporting materials needed
        - Key emphasis points for authority and credibility
        - Document references and source citations
        - Graphics, data visualization opportunities
        - Expert quote overlays or callouts
        
        **AUTHORITY & CREDIBILITY ELEMENTS:**
        - Document insights strategically integrated
        - Expert knowledge naturally woven throughout
        - Factual backing for all major claims
        - Credible source references where appropriate
        - Balance of accessible explanation with authoritative depth

        Generate the complete script now, ensuring it authentically matches the creator's style while delivering authoritative, document-informed content on the requested topic.
        """

        self.chat_modification_prompt = """
        You are an expert YouTube script editor working with a creator to refine their script. You have access to:

        **ORIGINAL SCRIPT:**
        {current_script}

        **CREATOR'S STYLE PROFILE:**
        {style_profile}

        **TOPIC INSIGHTS:**
        {topic_insights}

        **DOCUMENT KNOWLEDGE:**
        {document_insights}

        **MODIFICATION REQUEST:**
        {user_message}

        **INSTRUCTIONS:**
        Based on the creator's request, modify the script while:
        1. Maintaining their authentic voice and style
        2. Keeping document authority and expert knowledge intact
        3. Preserving key insights and valuable information
        4. Making targeted improvements based on the specific request
        5. Ensuring any new content fits naturally with existing flow

        **RESPONSE FORMAT:**
        **Modified Script:**
        [Provide the updated script or specific sections that were changed]

        **Changes Made:**
        - [Bullet point list of specific changes]
        - [Explanation of why these changes improve the script]
        - [Any suggestions for further improvements]

        Respond as if you're collaborating with the creator in a natural conversation.
        """
    
    def analyze_creator_style(self, personal_transcripts):
        """Analyze creator style from personal videos"""
        print("Starting creator style analysis...")
        combined_transcripts = "\n\n---VIDEO SEPARATOR---\n\n".join(personal_transcripts)
        
        max_chars = 50000
        if len(combined_transcripts) > max_chars:
            print(f"Truncating combined transcripts from {len(combined_transcripts)} to {max_chars} characters.")
            chunk_size = max_chars // 3
            start_chunk = combined_transcripts[:chunk_size]
            middle_start = len(combined_transcripts) // 2 - chunk_size // 2
            middle_chunk = combined_transcripts[middle_start:middle_start + chunk_size]
            end_chunk = combined_transcripts[-chunk_size:]
            combined_transcripts = f"{start_chunk}\n\n[...CONTENT CONTINUES...]\n\n{middle_chunk}\n\n[...CONTENT CONTINUES...]\n\n{end_chunk}"
        
        try:
            print("Generating style analysis with Gemini model...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                self.style_analysis_prompt + combined_transcripts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=3000
                )
            )
            
            if response.text:
                print(f"Style analysis completed: {len(response.text)} characters")
                print("\n\n" + "="*40)
                print("CREATOR STYLE PROFILE:")
                print("="*40)
                print(response.text)
                print("="*40 + "\n\n")
                return response.text
            else:
                print("Empty response from style analysis.")
                return "Could not analyze creator style - empty response"
                
        except Exception as e:
            logger.error(f"Error analyzing creator style: {str(e)}")
            print(f"Error in style analysis: {str(e)}")
            return f"Error analyzing creator style: {str(e)}"
    
    def analyze_inspiration_content(self, inspiration_transcripts):
        """Analyze inspiration content for topic insights"""
        print("Starting inspiration content analysis...")
        combined_transcripts = "\n\n---VIDEO SEPARATOR---\n\n".join(inspiration_transcripts)
        
        max_chars = 50000
        if len(combined_transcripts) > max_chars:
            print(f"Truncating inspiration transcripts from {len(combined_transcripts)} to {max_chars} characters.")
            chunk_size = max_chars // len(inspiration_transcripts) if len(inspiration_transcripts) > 1 else max_chars
            sampled_transcripts = []
            for transcript in inspiration_transcripts:
                if len(transcript) > chunk_size:
                    half_chunk = chunk_size // 2
                    sampled = transcript[:half_chunk] + "\n[...CONTENT CONTINUES...]\n" + transcript[-half_chunk:]
                    sampled_transcripts.append(sampled)
                else:
                    sampled_transcripts.append(transcript)
            combined_transcripts = "\n\n---VIDEO SEPARATOR---\n\n".join(sampled_transcripts)
        
        try:
            print("Generating inspiration analysis with Gemini model...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                self.inspiration_analysis_prompt + combined_transcripts,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=3000
                )
            )
            
            if response.text:
                print(f"Inspiration analysis completed: {len(response.text)} characters")
                print("\n\n" + "="*40)
                print("INSPIRATION SUMMARY:")
                print("="*40)
                print(response.text)
                print("="*40 + "\n\n")
                return response.text
            else:
                print("Empty response from inspiration analysis.")
                return "Could not analyze inspiration content - empty response"
                
        except Exception as e:
            logger.error(f"Error analyzing inspiration content: {str(e)}")
            print(f"Error in inspiration analysis: {str(e)}")
            return f"Error analyzing inspiration content: {str(e)}"
    
    def analyze_documents(self, document_texts):
        """Analyze uploaded documents for insights"""
        if not document_texts:
            print("No documents provided for analysis.")
            return "No documents provided for analysis."
        
        print("Starting document analysis...")
        combined_documents = "\n\n---DOCUMENT SEPARATOR---\n\n".join(document_texts)
        
        # Limit document content for processing
        max_chars = 60000
        if len(combined_documents) > max_chars:
            print(f"Truncating documents from {len(combined_documents)} to {max_chars} characters.")
            # Take chunks from each document rather than truncating
            chunk_size = max_chars // len(document_texts) if len(document_texts) > 1 else max_chars
            sampled_docs = []
            for doc_text in document_texts:
                if len(doc_text) > chunk_size:
                    # Take beginning and end of each document
                    half_chunk = chunk_size // 2
                    sampled = doc_text[:half_chunk] + "\n[...DOCUMENT CONTINUES...]\n" + doc_text[-half_chunk:]
                    sampled_docs.append(sampled)
                else:
                    sampled_docs.append(doc_text)
            combined_documents = "\n\n---DOCUMENT SEPARATOR---\n\n".join(sampled_docs)
        
        try:
            print("Generating document analysis with Gemini model...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                self.document_analysis_prompt + combined_documents,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=3500
                )
            )
            
            if response.text:
                print(f"Document analysis completed: {len(response.text)} characters")
                print("\n\n" + "="*40)
                print("DOCUMENT INSIGHTS:")
                print("="*40)
                print(response.text)
                print("="*40 + "\n\n")
                return response.text
            else:
                print("Empty response from document analysis.")
                return "Could not analyze document content - empty response"
                
        except Exception as e:
            logger.error(f"Error analyzing documents: {str(e)}")
            print(f"Error in document analysis: {str(e)}")
            return f"Error analyzing documents: {str(e)}"
    
    def generate_enhanced_script(self, style_profile, inspiration_summary, document_insights, user_prompt, target_minutes=None):
        """Generate script with all available knowledge sources"""
        print("Starting enhanced script generation...")
        
        # Handle duration instructions
        if target_minutes:
            target_duration = f"Approximately {target_minutes} minutes"
            duration_instruction = f"Structure the script to fit approximately {target_minutes} minutes of video content (roughly {target_minutes * 150} words, assuming 150 words per minute speaking pace). Adjust content density and pacing accordingly."
        else:
            target_duration = "No specific duration requirement"
            duration_instruction = "Create a comprehensive script with appropriate length for the topic, typically 8-12 minutes for in-depth content."
        
        enhanced_prompt = self.enhanced_script_template.format(
            style_profile=style_profile,
            inspiration_summary=inspiration_summary,
            document_insights=document_insights,
            user_prompt=user_prompt,
            target_duration=target_duration,
            duration_instruction=duration_instruction
        )
        print(f"Enhanced prompt prepared: {len(enhanced_prompt)} characters")
        
        try:
            print("Generating script with Gemini model...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=4000
                )
            )
            
            if response.text:
                print(f"Script generated: {len(response.text)} characters")
                print("\n\n" + "="*40)
                print("GENERATED SCRIPT:")
                print("="*40)
                print(response.text)
                print("="*40 + "\n\n")
                return response.text
            else:
                print("Empty response from script generation.")
                return "Error: Could not generate script - empty response"
                
        except Exception as e:
            logger.error(f"Error generating enhanced script: {str(e)}")
            print(f"Error in script generation: {str(e)}")
            return f"Error generating script: {str(e)}"

    def modify_script_chat(self, current_script, style_profile, topic_insights, document_insights, user_message):
        """Modify script with full context including documents"""
        print("Starting script modification via chat...")
        
        chat_prompt = self.chat_modification_prompt.format(
            current_script=current_script,
            style_profile=style_profile,
            topic_insights=topic_insights,
            document_insights=document_insights,
            user_message=user_message
        )
        print(f"Chat modification prompt: {len(chat_prompt)} characters")
        
        try:
            print("Generating modification with Gemini model...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                chat_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.6,
                    max_output_tokens=3000
                )
            )
            
            if response.text:
                print(f"Modification response: {len(response.text)} characters")
                print("\n\n" + "="*40)
                print("MODIFIED SCRIPT RESPONSE:")
                print("="*40)
                print(response.text)
                print("="*40 + "\n\n")
                return response.text
            else:
                print("Empty response from modification.")
                return "Could not modify script - empty response"
                
        except Exception as e:
            logger.error(f"Error modifying script: {str(e)}")
            print(f"Error in script modification: {str(e)}")
            return f"Error modifying script: {str(e)}"


# Initialize processors for export
document_processor = DocumentProcessor()
video_processor = VideoProcessor()
script_generator = EnhancedScriptGenerator()

# Export variables that app.py needs
__all__ = [
    'DocumentProcessor',
    'VideoProcessor', 
    'EnhancedScriptGenerator',
    'user_data',
    'UPLOAD_FOLDER',
    'ALLOWED_EXTENSIONS',
    'MAX_CONTENT_LENGTH',
    'document_processor',
    'video_processor',
    'script_generator'
]
