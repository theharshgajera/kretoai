"""
One-time Whisper Model Downloader
Run this script once to download and store the Whisper model permanently
"""

import whisper
import os
import shutil

# Your custom model storage path
MODEL_DIR = r"C:\Users\harsh\Downloads\base.pt"

def download_whisper_model():
    """Download Whisper model to custom directory"""
    
    print("="*60)
    print("WHISPER MODEL DOWNLOADER")
    print("="*60)
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"\n✓ Model directory: {MODEL_DIR}")
    
    # Model options: tiny, base, small, medium, large
    # tiny = 72MB, fastest, good accuracy for most use cases
    # base = 139MB, better accuracy, slightly slower
    model_name = "base"  # Change to "base" if you want better accuracy
    
    print(f"\n📥 Downloading Whisper '{model_name}' model...")
    print(f"   Size: ~72MB for 'tiny', ~139MB for 'base'")
    print(f"   This will take 1-5 minutes depending on your internet speed...\n")
    
    try:
        # Download model
        model = whisper.load_model(
            model_name,
            download_root=MODEL_DIR
        )
        
        print(f"\n{'='*60}")
        print(f"✅ MODEL DOWNLOADED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Model stored at: {MODEL_DIR}")
        print(f"Model file: {model_name}.pt")
        
        # Verify file exists
        model_file = os.path.join(MODEL_DIR, f"{model_name}.pt")
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file)
            print(f"File size: {file_size / (1024*1024):.2f} MB")
            print(f"\n✓ Ready to use in your application!")
        else:
            print(f"\n⚠️ Warning: Model file not found at expected location")
        
        print(f"\n{'='*60}")
        print(f"NEXT STEPS:")
        print(f"{'='*60}")
        print(f"1. Model is now stored permanently in: {MODEL_DIR}")
        print(f"2. Your application will load it from there (no re-download)")
        print(f"3. You can delete this download script if you want")
        print(f"{'='*60}\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    success = download_whisper_model()
    
    if success:
        print("\n🎉 All done! You can now use the model in your application.")
    else:
        print("\n⚠️ Download failed. Please check your internet connection and try again.")
    
    input("\nPress Enter to exit...")