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

# ADD ANTHROPIC IMPORT
import anthropic

# Load environment variables
load_dotenv()

# Configure Anthropic client
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

class DocumentProcessor:
    """Passes document URLs directly to Gemini for analysis"""
    
    def __init__(self):
        self.max_chars = 100000 
    
    def allowed_file(self, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def process_document(self, source, filename=None):
        """
        Direct Link Analysis: Passes the URL to Gemini.
        """
        print(f"\n--- Passing Document Link to Gemini: {filename or source} ---")
        
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            
            # This prompt tells Gemini to fetch and analyze the specific link
            prompt = f"""
            Analyze the document at this URL: {source}
            
            Please perform a DEEP SCAN and extract:
            1. Every major argument and core concept.
            2. All specific data, facts, and statistics.
            3. Technical details and patterns (especially if it's an exam paper).
            
            Ensure the output is a high-density Knowledge Base for a YouTube script.
            """

            response = model.generate_content(prompt)

            if response.text:
                full_text = response.text.strip()
                return {
                    "error": None,
                    "text": full_text,
                    "stats": {
                        "word_count": len(full_text.split()),
                        "source_type": "gemini_direct_doc_link"
                    },
                    "filename": filename or "Document"
                }
            return {"error": "Gemini returned empty text for this document link", "text": None}

        except Exception as e:
            logger.error(f"Document Link Error: {str(e)}")
            return {"error": f"Gemini could not process document link: {str(e)}", "text": None}
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
        Extract COMPLETE YouTube transcript, then create intelligent 500-word summary
        Preserves ALL important points while being concise
        
        SAME METHOD NAME - Just replace the entire method body
        """
        print(f"\n{'='*60}")
        print(f"EXTRACTING YOUTUBE TRANSCRIPT: {youtube_video_url}")
        print(f"{'='*60}\n")
        
        try:
            # Step 1: Extract video ID
            video_id = self.extract_video_id(youtube_video_url)
            if not video_id:
                return {"error": "Could not extract video ID from URL", "transcript": None, "stats": None}
            
            print(f"Video ID: {video_id}")
            
            # Step 2: Get COMPLETE transcript from YouTube
            print("Fetching full transcript from YouTube...")
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
                
                # Combine all transcript segments
                raw_transcript = " ".join([entry['text'] for entry in transcript_list])
                
                if not raw_transcript or len(raw_transcript.strip()) < 100:
                    print("⚠️ Transcript too short")
                    raise Exception("Transcript too short")
                
                print(f"✓ Full transcript extracted: {len(raw_transcript):,} characters")
                print(f"✓ Word count: {len(raw_transcript.split()):,} words")
                
                # Step 3: Create INTELLIGENT 500-word summary with Gemini
                print("Creating intelligent 500-word summary...")
                model = genai.GenerativeModel("gemini-2.0-flash")
                
                # Smart truncation if transcript is MASSIVE (> 100k chars)
                max_chars = 100000
                if len(raw_transcript) > max_chars:
                    print(f"⚠️ Truncating transcript from {len(raw_transcript):,} to {max_chars:,} chars for processing")
                    # Keep beginning, middle sample, and end
                    chunk_size = max_chars // 3
                    raw_transcript = (
                        f"{raw_transcript[:chunk_size]}\n\n"
                        f"[...MIDDLE CONTENT...]\n\n"
                        f"{raw_transcript[len(raw_transcript)//2:len(raw_transcript)//2 + chunk_size]}\n\n"
                        f"[...MORE CONTENT...]\n\n"
                        f"{raw_transcript[-chunk_size:]}"
                    )
                
                summary_prompt = f"""Create a comprehensive 500-word summary of this YouTube transcript that captures EVERY important point.
    
    TRANSCRIPT:
    {raw_transcript}
    
    INSTRUCTIONS:
    1. Extract ALL key arguments, main points, and core concepts
    2. Include specific data, statistics, examples, and facts mentioned
    3. Preserve technical details and terminology
    4. Maintain the speaker's key insights and unique perspectives
    5. Organize logically by topic/theme
    6. Write in dense, information-rich prose (NO fluff)
    7. Target EXACTLY 500 words - make every word count
    
    OUTPUT FORMAT:
    - Write as continuous prose (paragraphs, not bullet points)
    - Focus on WHAT was said, not meta-commentary
    - Preserve context needed for script generation
    - Be comprehensive yet concise
    
    Generate the 500-word summary now:"""
                
                response = model.generate_content(
                    summary_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,  # Low temp for accuracy
                        max_output_tokens=800  # ~500 words + buffer
                    )
                )
                
                if response.text:
                    summary = response.text.strip()
                    word_count = len(summary.split())
                    
                    print(f"\n{'='*60}")
                    print(f"✓ INTELLIGENT SUMMARY COMPLETE")
                    print(f"{'='*60}")
                    print(f"Summary: {len(summary):,} chars")
                    print(f"Words: {word_count:,} (target: 500)")
                    print(f"Compression: {len(raw_transcript):,} → {len(summary):,} chars")
                    print(f"{'='*60}\n")
                    
                    # Preview first 300 chars
                    print(f"Summary Preview:")
                    print(f"{'-'*60}")
                    print(summary[:300] + "..." if len(summary) > 300 else summary)
                    print(f"{'='*60}\n")
                    
                    return {
                        "error": None,
                        "transcript": summary,  # Return summary, not full transcript
                        "stats": {
                            'char_count': len(summary),
                            'word_count': word_count,
                            'original_length': len(raw_transcript),
                            'original_words': len(raw_transcript.split()),
                            'source_type': 'youtube_transcript',
                            'video_id': video_id,
                            'url': youtube_video_url
                        }
                    }
                else:
                    # Fallback: create basic summary from first 500 words
                    print("⚠️ Gemini summary failed, using first 500 words of transcript")
                    words = raw_transcript.split()[:500]
                    fallback_summary = " ".join(words) + "..."
                    
                    return {
                        "error": None,
                        "transcript": fallback_summary,
                        "stats": {
                            'char_count': len(fallback_summary),
                            'word_count': 500,
                            'source_type': 'youtube_transcript_fallback',
                            'video_id': video_id,
                            'url': youtube_video_url
                        }
                    }
                    
            except (TranscriptsDisabled, NoTranscriptFound):
                print("⚠️ No transcript available - video may not have captions")
                return {
                    "error": "This video does not have available transcripts/captions. Please try a video with captions enabled.",
                    "transcript": None,
                    "stats": None
                }
            except VideoUnavailable:
                return {
                    "error": "Video is unavailable or private",
                    "transcript": None,
                    "stats": None
                }
                
        except Exception as e:
            logger.error(f"YouTube transcript extraction error: {str(e)}")
            return {
                "error": f"Could not extract transcript: {str(e)}",
                "transcript": None,
                "stats": None
            }
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
