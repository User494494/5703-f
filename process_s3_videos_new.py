#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import os
import tempfile
import subprocess
from analyse_m_s import analyze_screen
from datetime import datetime
import getpass


BUCKET = 'cs14-2-recordingtool'
s3 = boto3.client('s3')

def convert_webm_to_mp4(input_path, output_path):
    """
    Convert WebM to MP4 using FFmpeg
    
    Args:
        input_path: Path to the input WebM file
        output_path: Path where the output MP4 file will be saved
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    try:
        cmd = [
            'ffmpeg', 
            '-i', input_path,           # Input file
            '-c:v', 'libx264',          # Video codec: H.264
            '-c:a', 'aac',              # Audio codec: AAC
            '-preset', 'fast',          # Encoding speed preset
            '-movflags', '+faststart',  # Optimize for streaming
            '-y',                       # Overwrite output file if exists
            output_path
        ]
        
        # Run FFmpeg with suppressed output
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Conversion successful: {os.path.basename(input_path)} → MP4")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg conversion failed for {input_path}")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("FFmpeg not found. Please install FFmpeg on your system.")
        return False
    except Exception as e:
        print(f"Unexpected error during conversion: {str(e)}")
        return False

def list_screen_files(prefix='recording_results/'):
    """
    List all screen.webm files in the S3 bucket
    
    Args:
        prefix: S3 prefix to search for files
        
    Yields:
        str: S3 key for each screen.webm file found
    """
    try:
        paginator = s3.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('/screen.webm'):
                    yield key
    except Exception as e:
        print(f"Error listing S3 files: {str(e)}")

def process_and_upload(key):
    """
    Download WebM file, convert to MP4, process it, and upload results
    
    Args:
        key: S3 key of the WebM file to process
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            # Step 1: Download the original WebM file
            webm_path = os.path.join(tmpdir, 'screen.webm')
            print(f"Downloading {key}...")
            s3.download_file(BUCKET, key, webm_path)
            
            # Step 2: Convert WebM to MP4
            mp4_path = os.path.join(tmpdir, 'screen.mp4')
            print(f"Converting {os.path.basename(key)} to MP4...")
            
            if not convert_webm_to_mp4(webm_path, mp4_path):
                print(f"Skipping {key} due to conversion failure")
                return
            
            # Step 3: Process the MP4 file (no changes to existing analysis)
            output_dir = os.path.join(tmpdir, 'result')
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"Analyzing video {os.path.basename(key)}...")
            current_date = datetime.utcnow().strftime("%Y-%m-%d")
            current_user = getpass.getuser()
            analyze_screen(mp4_path, output_dir, current_date, current_user)
            
            # Step 4: Upload all analysis results to S3
            print(f"Uploading results for {os.path.basename(key)}...")
            upload_count = 0
            
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, output_dir)
                    rel_parent = os.path.dirname(key).replace('recording_results/', '', 1)
                    output_key = 'Output/' + rel_parent + '/' + rel_path
                    
                    s3.upload_file(abs_path, BUCKET, output_key)
                    upload_count += 1
                    print(f"  Uploaded {output_key}")
            
            print(f"Successfully processed {key} - uploaded {upload_count} files")
            
        except Exception as e:
            print(f"Error processing {key}: {str(e)}")
        
        # Note: tmpdir is automatically cleaned up here, including both WebM and MP4 files

def result_exists(key):
    """
    Check if analysis results already exist for this video
    
    Args:
        key: S3 key of the video file
        
    Returns:
        bool: True if results exist, False otherwise
    """
    try:
        rel_parent = os.path.dirname(key).replace('recording_results/', '', 1)
        # Check for a key result file to determine if processing is complete
        check_key = f'Output/{rel_parent}/scene1/mousecursor.png'
        s3.head_object(Bucket=BUCKET, Key=check_key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception as e:
        print(f"Error checking if results exist for {key}: {str(e)}")
        return False

def main():
    """
    Main processing loop: find all WebM files and process them
    """
    print("Starting video processing pipeline...")
    print(f"S3 Bucket: {BUCKET}")
    print(f"Current time: 2025-05-27 10:25:27 UTC")
    print(f"User: tmpop21")
    print("-" * 50)
    
    processed_count = 0
    skipped_count = 0
    
    try:
        for key in list_screen_files():
            print(f"\nFound video: {key}")
            
            if result_exists(key):
                print(f"  → Skip {key}, already processed.")
                skipped_count += 1
                continue
            
            print(f"  → Processing {key}...")
            process_and_upload(key)
            processed_count += 1
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    except Exception as e:
        print(f"Unexpected error in main loop: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"Processing complete!")
    print(f"Videos processed: {processed_count}")
    print(f"Videos skipped: {skipped_count}")

if __name__ == "__main__":
    main()


