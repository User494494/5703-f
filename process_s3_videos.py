import boto3
import os
import tempfile
from analyze_screen import analyze_screen

BUCKET = 'your-bucket-name'
s3 = boto3.client('s3')

def list_screen_files(prefix='recording__results/'):
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/screen.webm'):
                yield key

def process_and_upload(key):
    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, 'screen.webm')
        s3.download_file(BUCKET, key, local_path)
        output_dir = os.path.join(tmpdir, 'result')
        os.makedirs(output_dir, exist_ok=True)
        analyze_screen(local_path, output_dir, "2025-05-21", "User494494")
        # 上传所有结果文件
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                output_key = 'output/' + os.path.dirname(key) + '/' + rel_path
                s3.upload_file(abs_path, BUCKET, output_key)
                print(f"Uploaded {output_key}")

def main():
    for key in list_screen_files():
        process_and_upload(key)

if __name__ == "__main__":
    main()
