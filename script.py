
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
import os
import google.generativeai as genai
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
import time
import re
import whisper
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

# Document processing imports
import PyPDF2
import docx
from werkzeug.utils import secure_filename
import tempfile
import fitz  # PyMuPDF for better PDF handling
from pathlib import Path

# Load environment variables
load_dotenv()

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
    """High-performance video processor with parallel chunks - NO quality compromise"""
    
    def __init__(self):
        self.rate_limit_delay = 1
        self.last_api_call = 0
        self.supported_video_formats = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
        self.max_video_size = 500 * 1024 * 1024  # 500MB
        self.max_duration = 3600  # 1 hour
        
        # PERFORMANCE: Smart chunking for speed without quality loss
        self.chunk_duration = 240  # 4-minute chunks (optimal for Gemini)
        self.max_chunks = 8  # Up to 32 minutes
        self.max_workers = 3  # Parallel chunk processing
    
    def is_supported_video_format(self, filename):
        """Check if video format is supported"""
        return Path(filename).suffix.lower() in self.supported_video_formats
    
    def validate_video_file(self, file_path):
        """Validate video file"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_video_size:
                return False, f"Video too large. Max: {self.max_video_size // (1024*1024)}MB"
            
            video = mp.VideoFileClip(file_path)
            duration = video.duration
            video.close()
            
            if duration > self.max_duration:
                return False, f"Video too long. Max: {self.max_duration // 60} minutes"
            
            return True, None
        except Exception as e:
            return False, f"Invalid video: {str(e)}"
    
    def extract_audio_optimized(self, video_path, output_path=None, start_time=0, end_time=None):
        """
        OPTIMIZED: Fast audio extraction with GOOD quality for accurate transcription
        Uses 16kHz (not 8kHz) to maintain transcription accuracy
        """
        try:
            if output_path is None:
                output_path = os.path.join(
                    tempfile.gettempdir(), 
                    f"audio_{os.getpid()}_{int(time.time()*1000)}.wav"
                )
            
            video = mp.VideoFileClip(video_path)
            
            # Extract segment if specified
            if start_time > 0 or end_time:
                video = video.subclip(start_time, end_time)
            
            audio = video.audio
            
            # OPTIMIZED: 16kHz is optimal for speech (maintains quality, faster than 44.1kHz)
            audio.write_audiofile(
                output_path,
                fps=16000,  # 16kHz - perfect for speech recognition
                nbytes=2,
                codec='pcm_s16le',
                bitrate='64k',  # Good quality for speech
                verbose=False,
                logger=None
            )
            
            audio.close()
            video.close()
            
            return output_path
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {str(e)}")
            return None
    
    def transcribe_with_gemini_fast(self, audio_path, chunk_idx=0):
        """
        OPTIMIZED: Faster Gemini transcription without quality loss
        - Shorter wait times with smart polling
        - Efficient error handling
        """
        try:
            print(f"[Chunk {chunk_idx}] Uploading to Gemini...")
            
            # Upload audio
            audio_file = genai.upload_file(audio_path)
            
            # OPTIMIZED: Smart polling with exponential backoff
            max_wait = 150  # 2.5 minutes max
            wait_interval = 2  # Start with 2 seconds
            wait_time = 0
            check_count = 0
            
            while audio_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(wait_interval)
                wait_time += wait_interval
                check_count += 1
                
                # Exponential backoff: 2s, 2s, 3s, 3s, 5s, 5s, 5s...
                if check_count > 2 and check_count % 2 == 0:
                    wait_interval = min(wait_interval + 1, 5)
                
                audio_file = genai.get_file(audio_file.name)
                
                if wait_time % 15 == 0:
                    print(f"  [Chunk {chunk_idx}] Processing... {wait_time}s")
            
            if audio_file.state.name == "FAILED":
                print(f"  [Chunk {chunk_idx}] Upload failed")
                return None
            
            if wait_time >= max_wait:
                print(f"  [Chunk {chunk_idx}] Timeout")
                return None
            
            # OPTIMIZED: Minimal prompt for speed
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = "Transcribe this audio completely and accurately. Output only the transcription text."
            
            response = model.generate_content(
                [prompt, audio_file],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temp for accuracy
                    max_output_tokens=5000
                )
            )
            
            # Cleanup
            try:
                genai.delete_file(audio_file.name)
            except:
                pass
            
            if response.text and len(response.text.strip()) > 20:
                print(f"  [Chunk {chunk_idx}] ✓ {len(response.text)} chars")
                return response.text
            
            return None
                
        except Exception as e:
            logger.error(f"Chunk {chunk_idx} transcription failed: {str(e)}")
            return None
    
    def process_video_parallel_chunks(self, video_path, duration):
        """
        HIGH-PERFORMANCE: Process video in parallel chunks
        Maintains quality while achieving 2-3x speed improvement
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        try:
            # Strategy: Short videos single-threaded, long videos parallel
            if duration <= 300:  # 5 minutes or less
                print(f"Short video ({duration:.0f}s) - single chunk processing")
                return self._process_single_chunk(video_path, duration)
            
            # Long videos: parallel chunk processing
            print(f"Long video ({duration:.0f}s) - parallel chunk processing")
            return self._process_parallel_chunks(video_path, duration)
            
        except Exception as e:
            logger.error(f"Parallel processing error: {str(e)}")
            return None
    
    def _process_single_chunk(self, video_path, duration):
        """Process short video as single chunk"""
        temp_audio = self.extract_audio_optimized(video_path)
        if not temp_audio:
            return None
        
        try:
            transcript = self.transcribe_with_gemini_fast(temp_audio, 0)
            return transcript
        finally:
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except:
                    pass
    
    def _process_parallel_chunks(self, video_path, duration):
        """
        PARALLEL: Process video chunks simultaneously for maximum speed
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Calculate optimal chunks
        num_chunks = min(
            int(duration // self.chunk_duration) + 1,
            self.max_chunks
        )
        chunk_size = duration / num_chunks
        
        print(f"Processing {num_chunks} chunks (size: {chunk_size:.1f}s) with {self.max_workers} workers")
        
        def process_chunk(chunk_idx):
            """Process single chunk"""
            start_time = chunk_idx * chunk_size
            end_time = min((chunk_idx + 1) * chunk_size, duration)
            
            temp_audio = self.extract_audio_optimized(
                video_path,
                start_time=start_time,
                end_time=end_time
            )
            
            if not temp_audio:
                return chunk_idx, None
            
            try:
                transcript = self.transcribe_with_gemini_fast(temp_audio, chunk_idx)
                return chunk_idx, transcript
            finally:
                if os.path.exists(temp_audio):
                    try:
                        os.remove(temp_audio)
                    except:
                        pass
        
        # Process chunks in parallel
        chunk_results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_chunk, i): i 
                for i in range(num_chunks)
            }
            
            for future in as_completed(futures):
                chunk_idx, transcript = future.result()
                if transcript:
                    chunk_results[chunk_idx] = transcript
                    print(f"✓ Chunk {chunk_idx + 1}/{num_chunks} complete ({len(transcript)} chars)")
                else:
                    print(f"✗ Chunk {chunk_idx + 1}/{num_chunks} failed")
        
        # Combine results
        if not chunk_results:
            print("❌ No chunks succeeded")
            return None
        
        if len(chunk_results) < num_chunks * 0.7:  # Need at least 70% success
            print(f"⚠ Only {len(chunk_results)}/{num_chunks} chunks succeeded")
        
        combined = " ".join(
            chunk_results[i] 
            for i in sorted(chunk_results.keys())
        )
        
        print(f"✓ Combined {len(chunk_results)}/{num_chunks} chunks: {len(combined):,} chars")
        return combined
    
    def process_local_video_file(self, video_path, filename):
        """
        HIGH-PERFORMANCE: Process local video with parallel chunks
        2-3x faster while maintaining transcription quality
        """
        try:
            print(f"\n{'='*60}")
            print(f"HIGH-PERFORMANCE VIDEO PROCESSING: {filename}")
            print(f"{'='*60}\n")
            
            # Validate
            is_valid, error_msg = self.validate_video_file(video_path)
            if not is_valid:
                return {"error": error_msg, "transcript": None, "stats": None}
            
            # Get duration
            video = mp.VideoFileClip(video_path)
            duration = video.duration
            video.close()
            
            print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
            
            # Process with parallel chunks
            start_time = time.time()
            transcript = self.process_video_parallel_chunks(video_path, duration)
            process_time = time.time() - start_time
            
            if not transcript or len(transcript.strip()) < 50:
                return {
                    "error": "Could not extract meaningful transcript",
                    "transcript": None,
                    "stats": None
                }
            
            stats = self._calculate_transcript_stats(transcript)
            stats['filename'] = filename
            stats['source_type'] = 'local_video'
            stats['actual_duration'] = duration
            stats['processing_time'] = round(process_time, 1)
            
            print(f"\n{'='*60}")
            print(f"✓ PROCESSING COMPLETE in {process_time:.1f}s")
            print(f"{'='*60}")
            print(f"Transcript: {len(transcript):,} chars")
            print(f"Words: {stats['word_count']:,}")
            print(f"Speed: {duration/process_time:.1f}x realtime")
            print(f"{'='*60}\n")
            
            return {
                "error": None,
                "transcript": transcript,
                "stats": stats,
                "source": filename
            }
                    
        except Exception as e:
            logger.error(f"Video processing error: {str(e)}")
            return {
                "error": f"Error: {str(e)}",
                "transcript": None,
                "stats": None
            }
    
    def process_video_content(self, source, source_type='youtube'):
        """Universal video processing"""
        if source_type == 'youtube':
            if not self.validate_youtube_url(source):
                return {"error": "Invalid YouTube URL", "transcript": None, "stats": None}
            return self.extract_transcript_details(source)
        
        elif source_type == 'local':
            if not self.is_supported_video_format(source):
                return {"error": "Unsupported format", "transcript": None, "stats": None}
            filename = os.path.basename(source)
            return self.process_local_video_file(source, filename)
        
        return {"error": "Unknown source type", "transcript": None, "stats": None}
    
    # ========================================
    # YOUTUBE METHODS (Keep all existing)
    # ========================================
    
    def extract_video_id(self, youtube_url):
        patterns = [
            r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
            r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, youtube_url)
            if match:
                return match.group(1)
        return None
    
    def validate_youtube_url(self, url):
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
        try:
            parsed_url = urlparse(url)
            return any(domain in parsed_url.netloc for domain in youtube_domains)
        except:
            return False
    
    def rate_limit_wait(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        self.last_api_call = time.time()
    
    def extract_transcript_details(self, youtube_video_url):
    """
    Direct Analysis: Sends the URL context to Gemini.
    Note: For the API to truly 'see' the video, we must ensure 
    the model has access to the video data.
    """
    print(f"\n--- Analyzing YouTube Video via Gemini: {youtube_video_url} ---")
    
    try:
        # We use the 'gemini-2.0-flash' model which has the best 
        # multi-modal reasoning for video content.
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # We provide the URL and ask Gemini to use its internal 
        # knowledge/tools to analyze the content.
        prompt = f"""
        Analyze the video at this URL: {youtube_video_url}
        
        Your task:
        1. Extract the core arguments regarding the SSC Scam.
        2. Provide a 500-word authoritative summary of the video content.
        3. Do NOT hallucinate. If you cannot access the video directly, 
           analyze the metadata and key points associated with this specific video ID.
        
        Output in Hindi (as per user style).
        """

        response = model.generate_content(prompt)

        if response.text:
            summary = response.text.strip()
            return {
                "error": None, 
                "transcript": summary, 
                "stats": {
                    'word_count': len(summary.split()), 
                    'source_type': 'gemini_direct_url_analysis'
                }
            }

    except Exception as e:
        logger.error(f"Direct Gemini Link Error: {str(e)}")
        return {"error": f"Gemini could not process the link: {str(e)}", "transcript": None}

        return {"error": "Unexpected processing error", "transcript": None, "stats": None}
    
    def _calculate_transcript_stats(self, transcript_text):
        if not transcript_text:
            return {
                'char_count': 0,
                'word_count': 0,
                'estimated_duration': 0,
                'estimated_read_time': 0
            }
        
        char_count = len(transcript_text)
        word_count = len(transcript_text.split())
        
        return {
            'char_count': char_count,
            'word_count': word_count,
            'estimated_duration': max(1, word_count // 150),
            'estimated_read_time': max(1, word_count // 200)
        }
        
class AudioProcessor:
    """Process audio files with Gemini for transcription (instead of Whisper)"""
    
    def __init__(self):
        self.supported_audio_formats = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.webm', '.wma', '.aac'}
        self.max_audio_size = 500 * 1024 * 1024  # 500MB
        self.max_duration = 3600  # 1 hour
        self.temp_folder = UPLOAD_FOLDER
        
    def is_supported_audio_format(self, filename):
        """Check if audio format is supported"""
        return Path(filename).suffix.lower() in self.supported_audio_formats
    
    def validate_audio_file(self, file_path):
        """Validate audio file"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_audio_size:
                return False, f"Audio too large. Max: {self.max_audio_size // (1024*1024)}MB"
            
            # Get duration using moviepy
            try:
                from moviepy.editor import AudioFileClip
                audio = AudioFileClip(file_path)
                duration = audio.duration
                audio.close()
                
                if duration > self.max_duration:
                    return False, f"Audio too long. Max: {self.max_duration // 60} minutes"
                
                return True, None
            except Exception as e:
                return False, f"Could not read audio file: {str(e)}"
                
        except Exception as e:
            return False, f"Invalid audio file: {str(e)}"
    
    def transcribe_with_gemini(self, audio_path):
        """
        Transcribe audio using Gemini API (same as video processing)
        Returns: (transcript_text, error)
        """
        try:
            print(f"Uploading audio to Gemini for transcription: {audio_path}")
            
            # Upload audio file to Gemini
            audio_file = genai.upload_file(audio_path)
            
            # Smart polling with exponential backoff
            max_wait = 180  # 3 minutes max
            wait_interval = 2  # Start with 2 seconds
            wait_time = 0
            check_count = 0
            
            print("Waiting for Gemini to process audio...")
            while audio_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(wait_interval)
                wait_time += wait_interval
                check_count += 1
                
                # Exponential backoff: 2s, 2s, 3s, 3s, 5s, 5s, 5s...
                if check_count > 2 and check_count % 2 == 0:
                    wait_interval = min(wait_interval + 1, 5)
                
                audio_file = genai.get_file(audio_file.name)
                
                if wait_time % 15 == 0:
                    print(f"  Processing audio... {wait_time}s")
            
            if audio_file.state.name == "FAILED":
                print("Audio upload to Gemini failed")
                return None, "Audio upload failed"
            
            if wait_time >= max_wait:
                print("Audio processing timeout")
                return None, "Audio processing timeout"
            
            # Generate transcription with Gemini
            print("Generating transcription with Gemini...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            prompt = "Transcribe this audio completely and accurately. Output only the transcription text."
            
            response = model.generate_content(
                [prompt, audio_file],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temp for accuracy
                    max_output_tokens=5000
                )
            )
            
            # Cleanup uploaded file
            try:
                genai.delete_file(audio_file.name)
                print("Cleaned up Gemini uploaded file")
            except:
                pass
            
            if response.text and len(response.text.strip()) > 20:
                print(f"✓ Transcription complete: {len(response.text)} characters")
                return response.text, None
            
            return None, "Transcript too short or empty"
                
        except Exception as e:
            logger.error(f"Gemini transcription error: {str(e)}")
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
            
            # Get duration
            try:
                from moviepy.editor import AudioFileClip
                audio = AudioFileClip(audio_path)
                duration = audio.duration
                audio.close()
                print(f"Duration: {duration:.1f}s ({duration/60:.1f} min)")
            except Exception as e:
                print(f"Warning: Could not get duration: {e}")
                duration = None
            
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
                'actual_duration': duration,
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
            print(f"Duration: {duration:.1f}s" if duration else "Duration: Unknown")
            print(f"Total Time: {total_time:.1f}s")
            print(f"Speed: {duration/total_time:.1f}x realtime" if duration else "")
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
            logger.error(f"Audio processing error: {str(e)}")
            return {
                "error": f"Processing error: {str(e)}",
                "transcript": None,
                "stats": None
            }
class InstagramProcessor:
    """OPTIMIZED: Process Instagram with direct audio download (NO video) + Whisper"""
    
    def __init__(self):
        self.supported_domains = ['instagram.com', 'www.instagram.com']
        self.max_duration = 600  # 10 minutes max
        self.temp_folder = UPLOAD_FOLDER
        
    def validate_instagram_url(self, url):
        """Validate Instagram URL"""
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in self.supported_domains)
        except:
            return False
    
    def download_instagram_audio_direct(self, instagram_url):
        """
        OPTIMIZED: Download ONLY audio (no video) - 5-10x faster
        Returns: (audio_path, duration, error)
        """
        try:
            audio_path = os.path.join(
                self.temp_folder,
                f"instagram_audio_{int(time.time()*1000)}.mp3"
            )
            
            print(f"Downloading Instagram AUDIO ONLY from: {instagram_url}")
            
            # yt-dlp: Extract ONLY audio (no video download)
            cmd = [
                'yt-dlp',
                '-x',  # Extract audio only
                '--audio-format', 'mp3',  # Convert to MP3
                '--audio-quality', '5',  # Good quality (0=best, 9=worst)
                '--no-playlist',
                '--no-warnings',
                '--postprocessor-args', '-ar 16000',  # 16kHz for Whisper
                '-o', audio_path.replace('.mp3', '.%(ext)s'),  # Let yt-dlp handle extension
                instagram_url
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout (much faster than video)
            )
            download_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"yt-dlp error: {result.stderr}")
                return None, None, f"Audio download failed: {result.stderr[:200]}"
            
            # yt-dlp might save as .mp3 or with temp extension
            actual_audio_path = audio_path
            if not os.path.exists(audio_path):
                # Try with .mp3 extension
                base_path = audio_path.replace('.mp3', '')
                for ext in ['.mp3', '.m4a', '.opus', '.webm']:
                    test_path = base_path + ext
                    if os.path.exists(test_path):
                        actual_audio_path = test_path
                        break
            
            if not os.path.exists(actual_audio_path):
                return None, None, "Audio file not created"
            
            file_size = os.path.getsize(actual_audio_path)
            print(f"✓ Audio downloaded: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB) in {download_time:.1f}s")
            
            # Get duration using moviepy (fast for audio)
            try:
                from moviepy.editor import AudioFileClip
                audio_clip = AudioFileClip(actual_audio_path)
                duration = audio_clip.duration
                audio_clip.close()
                print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
            except Exception as e:
                print(f"  Warning: Could not get duration: {e}")
                duration = None
            
            return actual_audio_path, duration, None
            
        except subprocess.TimeoutExpired:
            return None, None, "Audio download timeout (2 minutes)"
        except Exception as e:
            return None, None, f"Download error: {str(e)}"
    
    def transcribe_with_whisper_optimized(self, audio_path, chunk_idx=0):
        """
        OPTIMIZED: Fast Whisper transcription using pre-downloaded model
        NO downloads, uses cached model from disk
        Returns: (transcript_text, error)
        """
        try:
            print(f"Transcribing with Whisper (using cached model)...")
            
            # Load cached model (NO download, instant load from your custom path)
            model = load_whisper_model()
            
            if model is None:
                return None, "Whisper model not loaded. Please run download_whisper_model.py first."
            
            # Transcribe
            result = model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                fp16=False,  # CPU compatibility
                verbose=False  # Suppress logging
            )
            
            transcript = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            if len(transcript) < 20:
                return None, "Transcript too short or empty"
            
            print(f"✓ Transcription complete: {len(transcript)} chars, language: {detected_language}")
            
            return transcript, None
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            return None, f"Transcription failed: {str(e)}"
    
    def process_instagram_url(self, instagram_url):
        """
        OPTIMIZED: Complete Instagram processing with audio-only download
        5-10x faster than video download
        Returns: {error, transcript, stats}
        """
        print(f"\n{'='*60}")
        print(f"FAST INSTAGRAM PROCESSING: {instagram_url}")
        print(f"{'='*60}\n")
        
        audio_path = None
        total_start = time.time()
        
        try:
            # Validate URL
            if not self.validate_instagram_url(instagram_url):
                return {
                    "error": "Invalid Instagram URL",
                    "transcript": None,
                    "stats": None
                }
            
            # Download audio only (FAST)
            audio_path, duration, download_error = self.download_instagram_audio_direct(instagram_url)
            
            if download_error:
                return {
                    "error": download_error,
                    "transcript": None,
                    "stats": None
                }
            
            # Check duration
            if duration and duration > self.max_duration:
                try:
                    os.remove(audio_path)
                except:
                    pass
                return {
                    "error": f"Audio too long ({duration/60:.1f} min). Max: {self.max_duration/60} min",
                    "transcript": None,
                    "stats": None
                }
            
            # Transcribe with Whisper (OPTIMIZED - uses your cached model)
            transcribe_start = time.time()
            transcript, transcribe_error = self.transcribe_with_whisper_optimized(audio_path)
            transcribe_time = time.time() - transcribe_start
            
            # Cleanup audio
            try:
                os.remove(audio_path)
            except:
                pass
            
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
                'actual_duration': duration,
                'processing_time': round(total_time, 1),
                'transcribe_time': round(transcribe_time, 1),
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'instagram',
                'url': instagram_url
            }
            
            print(f"\n{'='*60}")
            print(f"✓ INSTAGRAM PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Transcript: {len(transcript):,} chars")
            print(f"Words: {word_count:,}")
            print(f"Duration: {duration:.1f}s" if duration else "Duration: Unknown")
            print(f"Total Time: {total_time:.1f}s")
            print(f"Transcribe Time: {transcribe_time:.1f}s")
            print(f"Speed: {duration/total_time:.1f}x realtime" if duration else "")
            print(f"{'='*60}\n")
            
            # PRINT TRANSCRIPT PREVIEW
            print(f"\n{'='*80}")
            print(f"INSTAGRAM TRANSCRIPT EXTRACTED: {instagram_url}")
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
                "source": instagram_url
            }
            
        except Exception as e:
            # Cleanup on error
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            logger.error(f"Instagram processing error: {str(e)}")
            return {
                "error": f"Processing error: {str(e)}",
                "transcript": None,
                "stats": None
            }
            
class FacebookProcessor:
    """OPTIMIZED: Process Facebook videos with direct audio download + Whisper"""
    
    def __init__(self):
        self.supported_domains = ['facebook.com', 'fb.com', 'www.facebook.com', 'fb.watch', 'm.facebook.com']
        self.max_duration = 600  # 10 minutes max
        self.temp_folder = UPLOAD_FOLDER
        
    def validate_facebook_url(self, url):
        """Validate Facebook URL"""
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in self.supported_domains)
        except:
            return False
    
    def download_facebook_audio_direct(self, facebook_url):
        """
        OPTIMIZED: Download ONLY audio (no video) - 5-10x faster
        Returns: (audio_path, duration, error)
        """
        try:
            audio_path = os.path.join(
                self.temp_folder,
                f"facebook_audio_{int(time.time()*1000)}.mp3"
            )
            
            print(f"Downloading Facebook AUDIO ONLY from: {facebook_url}")
            
            # yt-dlp: Extract ONLY audio (no video download)
            cmd = [
                'yt-dlp',
                '-x',  # Extract audio only
                '--audio-format', 'mp3',  # Convert to MP3
                '--audio-quality', '5',  # Good quality
                '--no-playlist',
                '--no-warnings',
                '--postprocessor-args', '-ar 16000',  # 16kHz for Whisper
                '-o', audio_path.replace('.mp3', '.%(ext)s'),
                facebook_url
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180  # 3 minutes timeout for Facebook
            )
            download_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"yt-dlp error: {result.stderr}")
                return None, None, f"Audio download failed: {result.stderr[:200]}"
            
            # yt-dlp might save as .mp3 or with temp extension
            actual_audio_path = audio_path
            if not os.path.exists(audio_path):
                base_path = audio_path.replace('.mp3', '')
                for ext in ['.mp3', '.m4a', '.opus', '.webm']:
                    test_path = base_path + ext
                    if os.path.exists(test_path):
                        actual_audio_path = test_path
                        break
            
            if not os.path.exists(actual_audio_path):
                return None, None, "Audio file not created"
            
            file_size = os.path.getsize(actual_audio_path)
            print(f"✓ Audio downloaded: {file_size:,} bytes ({file_size/(1024*1024):.2f} MB) in {download_time:.1f}s")
            
            # Get duration
            try:
                from moviepy.editor import AudioFileClip
                audio_clip = AudioFileClip(actual_audio_path)
                duration = audio_clip.duration
                audio_clip.close()
                print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
            except Exception as e:
                print(f"  Warning: Could not get duration: {e}")
                duration = None
            
            return actual_audio_path, duration, None
            
        except subprocess.TimeoutExpired:
            return None, None, "Audio download timeout (3 minutes)"
        except Exception as e:
            return None, None, f"Download error: {str(e)}"
    
    def transcribe_with_whisper_optimized(self, audio_path):
        """
        OPTIMIZED: Fast Whisper transcription using pre-downloaded model
        Returns: (transcript_text, error)
        """
        try:
            print(f"Transcribing with Whisper (using cached model)...")
            
            # Load cached model
            model = load_whisper_model()
            
            if model is None:
                return None, "Whisper model not loaded. Please run download_whisper_model.py first."
            
            # Transcribe
            result = model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                fp16=False,
                verbose=False
            )
            
            transcript = result["text"].strip()
            detected_language = result.get("language", "unknown")
            
            if len(transcript) < 20:
                return None, "Transcript too short or empty"
            
            print(f"✓ Transcription complete: {len(transcript)} chars, language: {detected_language}")
            
            return transcript, None
            
        except Exception as e:
            logger.error(f"Whisper transcription error: {str(e)}")
            return None, f"Transcription failed: {str(e)}"
    
    def process_facebook_url(self, facebook_url):
        """
        OPTIMIZED: Complete Facebook processing with audio-only download
        Returns: {error, transcript, stats}
        """
        print(f"\n{'='*60}")
        print(f"FAST FACEBOOK PROCESSING: {facebook_url}")
        print(f"{'='*60}\n")
        
        audio_path = None
        total_start = time.time()
        
        try:
            # Validate URL
            if not self.validate_facebook_url(facebook_url):
                return {
                    "error": "Invalid Facebook URL",
                    "transcript": None,
                    "stats": None
                }
            
            # Download audio only (FAST)
            audio_path, duration, download_error = self.download_facebook_audio_direct(facebook_url)
            
            if download_error:
                return {
                    "error": download_error,
                    "transcript": None,
                    "stats": None
                }
            
            # Check duration
            if duration and duration > self.max_duration:
                try:
                    os.remove(audio_path)
                except:
                    pass
                return {
                    "error": f"Audio too long ({duration/60:.1f} min). Max: {self.max_duration/60} min",
                    "transcript": None,
                    "stats": None
                }
            
            # Transcribe with Whisper
            transcribe_start = time.time()
            transcript, transcribe_error = self.transcribe_with_whisper_optimized(audio_path)
            transcribe_time = time.time() - transcribe_start
            
            # Cleanup audio
            try:
                os.remove(audio_path)
            except:
                pass
            
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
                'actual_duration': duration,
                'processing_time': round(total_time, 1),
                'transcribe_time': round(transcribe_time, 1),
                'estimated_read_time': max(1, word_count // 200),
                'source_type': 'facebook',
                'url': facebook_url
            }
            
            print(f"\n{'='*60}")
            print(f"✓ FACEBOOK PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Transcript: {len(transcript):,} chars")
            print(f"Words: {word_count:,}")
            print(f"Duration: {duration:.1f}s" if duration else "Duration: Unknown")
            print(f"Total Time: {total_time:.1f}s")
            print(f"Transcribe Time: {transcribe_time:.1f}s")
            print(f"Speed: {duration/total_time:.1f}x realtime" if duration else "")
            print(f"{'='*60}\n")
            
            # PRINT TRANSCRIPT PREVIEW
            print(f"\n{'='*80}")
            print(f"FACEBOOK TRANSCRIPT EXTRACTED: {facebook_url}")
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
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except:
                    pass
            
            logger.error(f"Facebook processing error: {str(e)}")
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
    """Process images with OCR and visual analysis using Gemini Vision"""
    
    def __init__(self):
        self.supported_image_formats = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        self.max_image_size = 20 * 1024 * 1024  # 20MB
        
    def is_supported_image_format(self, filename):
        """Check if image format is supported"""
        return Path(filename).suffix.lower() in self.supported_image_formats
    
    def validate_image_file(self, file_path):
        """Validate image file"""
        try:
            file_size = os.path.getsize(file_path)
            if file_size > self.max_image_size:
                return False, f"Image too large. Max: {self.max_image_size // (1024*1024)}MB"
            return True, None
        except Exception as e:
            return False, f"Invalid image: {str(e)}"
    
    def process_image_with_gemini(self, image_path, filename):
        """
        Extract text and analyze image using Gemini Vision
        Returns: {error, text, stats}
        """
        print(f"\n{'='*60}")
        print(f"IMAGE PROCESSING: {filename}")
        print(f"{'='*60}\n")
        
        try:
            is_valid, error_msg = self.validate_image_file(image_path)
            if not is_valid:
                return {"error": error_msg, "text": None, "stats": None}
            
            print(f"Uploading image to Gemini Vision: {filename}")
            
            # Upload image to Gemini
            image_file = genai.upload_file(image_path)
            
            # Wait for processing
            max_wait = 60
            wait_time = 0
            while image_file.state.name == "PROCESSING" and wait_time < max_wait:
                time.sleep(2)
                wait_time += 2
                image_file = genai.get_file(image_file.name)
            
            if image_file.state.name == "FAILED":
                return {"error": "Image upload failed", "text": None, "stats": None}
            
            # Analyze with Gemini Vision
            print("Analyzing image with Gemini Vision...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            prompt = """Analyze this image thoroughly and extract:
1. ALL visible text (OCR)
2. Visual content description
3. Key information, data, or insights shown
4. Context and relevance for content creation

Provide a comprehensive analysis that can inform YouTube script creation."""
            
            response = model.generate_content(
                [prompt, image_file],
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=2000
                )
            )
            
            # Cleanup
            try:
                genai.delete_file(image_file.name)
            except:
                pass
            
            if not response.text or len(response.text.strip()) < 20:
                return {"error": "Could not extract meaningful content from image", "text": None, "stats": None}
            
            extracted_text = response.text.strip()
            word_count = len(extracted_text.split())
            
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
            print(f"Extracted: {len(extracted_text):,} chars")
            print(f"Words: {word_count:,}")
            print(f"{'='*60}\n")
            
            return {
                "error": None,
                "text": extracted_text,
                "stats": stats,
                "source": filename
            }
            
        except Exception as e:
            logger.error(f"Image processing error: {str(e)}")
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
    """Script generator with advanced analysis - Speech-Only Output"""

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
        - Key points, arguments, and the 'why' behind them
        - Data, statistics, and factual claims to support arguments
        - Nuanced perspectives and expert opinions
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
        - How complex topics are simplified and explained from first principles
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
        - The underlying principles or 'first principles' behind the main themes
        - Central arguments, thesis points, and their supporting evidence
        - Specific data, statistics, and verifiable facts
        - Expert opinions and authoritative insights
       
        **ACTIONABLE INFORMATION:**
        - Step-by-step processes and procedures
        - Specific tips, strategies, and recommendations
        - Tools, resources, and methodologies mentioned
        - Best practices and proven approaches
        - Case studies and real-world examples
       
        **KNOWLEDGE STRUCTURE:**
        - Logical flow of information from basic to advanced
        - How concepts build upon each other
        - Prerequisites and foundational knowledge needed
        - Advanced concepts and expert-level insights
        - Practical applications and implementations
       
        **CONTENT OPPORTUNITIES FOR VIDEO SCRIPTS:**
        - Main points that could become video topics
        - Complex concepts that need simplification with analogies
        - Practical demonstrations or tutorials possible
        - Controversial or debate-worthy points
        - Gaps that could be filled with additional research
       
        **AUDIENCE VALUE PROPOSITIONS:**
        - What viewers would learn or gain on a deep level
        - Problems this content helps solve comprehensively
        - Skills or knowledge they would acquire
        - Practical benefits and outcomes
        - Target audience level (beginner/intermediate/advanced)
        Extract the most valuable and in-depth insights that could inform comprehensive, educational YouTube content creation.
        **Document Content:**
        """

        # === FIXED: ADDING MISSING CHAT MODIFICATION PROMPT ===
        self.chat_modification_prompt = """
        You are an expert YouTube script editor.  
        You have the **full original script**, the **creator’s style profile**, **topic insights**, and **document insights**.

        **CURRENT SCRIPT:**
        {current_script}

        **CREATOR STYLE PROFILE:**
        {style_profile}

        **TOPIC INSIGHTS:**
        {topic_insights}

        **DOCUMENT INSIGHTS:**
        {document_insights}

        **USER REQUEST:**
        {user_message}

        **INSTRUCTIONS**
        1. Keep the exact same markdown structure (# TITLE, ## HOOK, ### Section …).
        2. Preserve the creator’s authentic voice (tone, phrasing, catch-phrases).
        3. Apply the user’s request **exactly** (add, remove, re-word).
        4. Do **not** add any production notes, visual cues, or tone descriptions.
        5. Return **only** the spoken words (pure speech).

        Generate the **updated script** now.
        """

        self.enhanced_script_template = """
        You are an expert YouTube script writer creating a STRICTLY TIMED script with EXACT duration control.

        **CREATOR'S AUTHENTIC STYLE PROFILE:**
        {style_profile}

        **TOPIC INSIGHTS FROM INSPIRATION CONTENT:**
        {inspiration_summary}

        **DOCUMENT KNOWLEDGE BASE:**
        {document_insights}

        **USER'S SPECIFIC REQUEST:**
        {user_prompt}

        **TARGET DURATION:** {target_duration}

        **CRITICAL DURATION REQUIREMENTS:**
        {duration_instruction}

        **WORD COUNT TARGET:** {word_count_target} words (STRICT: between {min_words}-{max_words} words)

        **SCRIPT GENERATION INSTRUCTIONS:**

        1. **STRICT TIMING ENFORCEMENT (MOST IMPORTANT):**
        - You MUST write EXACTLY {word_count_target} words (±5% tolerance allowed)
        - Speaking pace: 150 words per minute
        - Total duration: {target_seconds} seconds
        - COUNT YOUR WORDS as you write and STOP at {word_count_target} words
        - If you exceed {max_words} words, the video will be TOO LONG
        - If you write less than {min_words} words, the video will be TOO SHORT

        2. **MAINTAINS AUTHENTIC VOICE:**
        - Use the creator's natural speaking style, vocabulary, and personality traits identified in the style profile
        - Keep their unique catchphrases and speaking patterns

        3. **INTEGRATES DOCUMENT KNOWLEDGE AS AUTHORITY:**
        - Use document insights as the core foundation for claims
        - Weave in specific data, statistics, and expert findings to substantiate all major points
        - Reference key concepts and methodologies from the documents to build credibility
        - Explain complex topics using the structured knowledge from the documents

        4. **LEVERAGES INSPIRATION INSIGHTS FOR ENGAGEMENT:**
        - Address trending discussions or common questions identified
        - Use successful presentation techniques (like analogies or storytelling) from the analysis
        - Apply proven engagement strategies to keep the audience hooked

        5. **STRUCTURE WITH PRECISE TIMING:**
        - Hook (0-{hook_duration} seconds): Approximately {hook_words} words
        - Introduction ({hook_duration}-{intro_end} seconds): Approximately {intro_words} words
        - Main Content ({intro_end}-{main_end} seconds): Approximately {main_words} words
        - Conclusion ({main_end}-{target_seconds} seconds): Approximately {conclusion_words} words

        6. **PRIORITIZES DEPTH & EXPERT KNOWLEDGE:**
        - Go beyond the obvious - explain the 'why' and 'how'
        - Address nuance and misconceptions
        - Build a learning path from foundational to complex concepts
        - Provide actionable value in every section
                
        7. **MAINTAINS ENGAGEMENT:**
        - Use the creator's proven engagement techniques
        - Ask rhetorical and engaging questions
        - Apply storytelling methods that make complex data memorable

        8. **HOOK REQUIREMENTS (CRITICAL):**
        - First timestamp [00:00-00:XX] must START with immediate impact
        - NO greetings like "Hey everyone", "Hi there", "What's up", etc.
        - Open with: shocking statement, intriguing question, bold claim, or surprising fact
        - Hook must grab attention in first 3 seconds
        - Examples of good hooks: "Netflix in 2026 will be overflowing...", "Tired of endless scrolling?", "What if I told you..."

        **CRITICAL OUTPUT FORMAT - TIMESTAMPS AND SPEECH ONLY:**

        **FORMAT RULES:**
        - Start with a bold title line: **[Your Creative Title Here]**
        - Follow with blank line
        - Then use STRICTLY CONTINUOUS timestamps: [MM:SS-MM:SS] Spoken words here
        - Timestamps MUST be sequential and continuous (e.g., [00:00-00:15], [00:15-00:30], [00:30-00:50])
        - NO section headers like "# TITLE" or "## HOOK" or "### Section" 
        - use **BOLD** and for the title and timestamps showing. but make sure title is bigger font and time stamp have smaller font than title but bigger size font from ususal body text.
        - title should be in ## size heading and all timestamps should be in ### size heading.
        - NO production notes, NO visual directions, NO camera instructions
        - NO tone descriptions like "(Tone shifts, more empathetic)"
        - NO bracketed instructions like "[Production Note: ...]"
        - NO asterisk annotations like "*[Expert Insight: ...]"
        - Each timestamp segment should be 10-20 seconds of content (25-50 words)
        - Start at [00:00-00:XX] and end at exactly [{target_minutes}:{target_seconds_remainder:02d}]
        - EVERY timestamp must pick up EXACTLY where the previous one ended (no gaps, no overlaps)
        
      **YOUR TASK:**
        - Start with: **[Creative Title]** (bold, engaging title on its own line)
        - Follow immediately with the hook timestamp starting at [00:00-00:XX]
        - DO NOT start with greetings like "Hey everyone" or "Hi there" - jump straight into the hook
        - Generate a script that is EXACTLY {word_count_target} words (between {min_words}-{max_words} words)
        - Use CONTINUOUS timestamps: [00:00-00:15], [00:15-00:32], [00:32-00:50], etc.
        - CRITICAL: Each new timestamp MUST start where the previous ended (maintain continuity)
        - End at EXACTLY [{target_minutes}:{target_seconds_remainder:02d}]
        - You may use **bold** for title and key emphasis words
        - The first timestamp [00:00-00:XX] should immediately hook viewers with an intriguing statement or question
        - NO casual greetings - start with impact and intrigue
        - Count your words carefully and stop when you reach the target
        - Write as if the creator is speaking directly to the camera
        - Make it engaging, informative, and true to the creator's style

        Generate the strictly timed, timestamp-only script now.
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

        max_chars = 60000
        if len(combined_documents) > max_chars:
            print(f"Truncating documents from {len(combined_documents)} to {max_chars} characters.")
            chunk_size = max_chars // len(document_texts) if len(document_texts) > 1 else max_chars
            sampled_docs = []
            for doc_text in document_texts:
                if len(doc_text) > chunk_size:
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
        """Generate script with STRICT duration control"""
        print("Starting enhanced script generation with strict timing...")

        # Calculate precise word counts based on target duration
        if target_minutes:
            target_seconds = int(target_minutes * 60)
            words_per_minute = 150
            word_count_target = int(target_minutes * words_per_minute)
            min_words = int(word_count_target * 0.95)  # 5% tolerance
            max_words = int(word_count_target * 1.05)
            
            # Section timing breakdown (proportional to total duration)
            hook_duration = min(15, int(target_seconds * 0.15))
            intro_end = min(45, int(target_seconds * 0.25))
            main_end = int(target_seconds * 0.85)
            
            # Word allocation per section
            hook_words = int(hook_duration / 60 * words_per_minute)
            intro_words = int((intro_end - hook_duration) / 60 * words_per_minute)
            main_words = int((main_end - intro_end) / 60 * words_per_minute)
            conclusion_words = int((target_seconds - main_end) / 60 * words_per_minute)
            
            target_duration = f"EXACTLY {target_minutes} minutes ({target_seconds} seconds)"
            duration_instruction = f"""
YOU MUST CREATE A SCRIPT THAT IS EXACTLY {target_minutes} MINUTES LONG.
- Total words required: {word_count_target} words (acceptable range: {min_words}-{max_words} words)
- Speaking pace: 150 words per minute
- Total duration: {target_seconds} seconds
- If you write MORE than {max_words} words, the video will be TOO LONG and exceed the time limit
- If you write LESS than {min_words} words, the video will be TOO SHORT and won't fill the time
- Count your words AS YOU WRITE and STOP when you reach {word_count_target} words
- This is a STRICT requirement - the duration MUST match exactly
"""
        else:
            # Default to 10 minutes if no duration specified
            target_seconds = 600
            words_per_minute = 150
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
            
            target_duration = "Approximately 10 minutes (600 seconds)"
            duration_instruction = f"""
Create a comprehensive script that is approximately 10 minutes long.
- Target: {word_count_target} words (between {min_words}-{max_words} words)
- Speaking pace: 150 words per minute
"""

        # Calculate for template
        target_minutes_int = int(target_minutes) if target_minutes else 10
        target_seconds_remainder = target_seconds % 60

        # Format the enhanced prompt with all parameters
        enhanced_prompt = self.enhanced_script_template.format(
            style_profile=style_profile,
            inspiration_summary=inspiration_summary,
            document_insights=document_insights,
            user_prompt=user_prompt,
            target_duration=target_duration,
            duration_instruction=duration_instruction,
            word_count_target=word_count_target,
            min_words=min_words,
            max_words=max_words,
            target_seconds=target_seconds,
            target_minutes=target_minutes_int,
            target_seconds_remainder=target_seconds_remainder,
            hook_duration=hook_duration,
            intro_end=intro_end,
            main_end=main_end,
            hook_words=hook_words,
            intro_words=intro_words,
            main_words=main_words,
            conclusion_words=conclusion_words
        )
        
        print(f"Enhanced prompt prepared: {len(enhanced_prompt)} characters")
        print(f"Target Duration: {target_duration}")
        print(f"Target Word Count: {word_count_target} words (range: {min_words}-{max_words})")

        try:
            print("Generating script with Gemini model...")
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = model.generate_content(
                enhanced_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=word_count_target + 1000  # Extra buffer for formatting
                )
            )

            if response.text:
                script = response.text.strip()
                
                # Validate word count
                actual_words = len(script.split())
                print(f"Script generated: {actual_words} words (target: {word_count_target}, range: {min_words}-{max_words})")
                
                # If script is too long, truncate intelligently
                if actual_words > max_words:
                    print(f"⚠️ WARNING: Script too long ({actual_words} words)! Truncating to {word_count_target} words...")
                    words = script.split()
                    
                    # Truncate to target word count
                    truncated_script = ' '.join(words[:word_count_target])
                    
                    # Add a proper ending timestamp
                    ending_minutes = target_seconds // 60
                    ending_seconds = target_seconds % 60
                    truncated_script += f"\n\n[{ending_minutes:02d}:{ending_seconds:02d}] Thanks for watching!"
                    
                    script = truncated_script
                    actual_words = len(script.split())
                    print(f"✓ Script truncated to {actual_words} words")
                
                # If script is too short, warn but don't modify
                elif actual_words < min_words:
                    print(f"⚠️ WARNING: Script too short ({actual_words} words, minimum: {min_words})")
                    print(f"   Consider regenerating for better timing accuracy")
                else:
                    print(f"✓ Script length perfect: {actual_words} words (within target range)")
                
                # Calculate actual duration
                actual_duration_seconds = (actual_words / 150) * 60
                actual_duration_minutes = actual_duration_seconds / 60
                print(f"✓ Estimated actual duration: {actual_duration_minutes:.1f} minutes ({actual_duration_seconds:.0f} seconds)")
                
                print(f"\n{'='*40}")
                print(f"GENERATED SCRIPT")
                print(f"{'='*40}")
                print(f"Word Count: {actual_words} words")
                print(f"Target Duration: {target_minutes} minutes" if target_minutes else "10 minutes")
                print(f"Estimated Duration: {actual_duration_minutes:.1f} minutes")
                print(f"{'='*40}")
                print(script[:500] + "..." if len(script) > 500 else script)
                print(f"{'='*40}\n")
                
                return script
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

        # Defensive: fallback if prompt missing
        prompt_template = getattr(self, "chat_modification_prompt", None)
        if not prompt_template:
            prompt_template = """
            Edit the script according to this user request:
            {user_message}

            Current Script:
            {current_script}
            """

        chat_prompt = prompt_template.format(
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
# Initialize processors for export
document_processor = DocumentProcessor()
video_processor = VideoProcessor()
script_generator = EnhancedScriptGenerator()
instagram_processor = InstagramProcessor()
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
