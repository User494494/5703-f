import boto3
import os
import tempfile
from analyse_m_s import analyze_screen

BUCKET = 'cs14-2-recordingtool'
s3 = boto3.client('s3')

def list_screen_files(prefix='recording_results/'):
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
        # Upload all results 
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, output_dir)
                rel_parent = os.path.dirname(key).replace('recording_results/', '', 1)
                output_key = 'Output/' + rel_parent + '/' + rel_path
                s3.upload_file(abs_path, BUCKET, output_key)
                print(f"Uploaded {output_key}")

def result_exists(key):
    rel_parent = os.path.dirname(key).replace('recording_results/', '', 1)

    check_key = f'output/{rel_parent}/scene1/mousecursor.png'
    try:
        s3.head_object(Bucket=BUCKET, Key=check_key)
        return True
    except s3.exceptions.ClientError:
        return False

def main():
    for key in list_screen_files():
        if result_exists(key):
            print(f"Skip {key}, already processed.")
            continue
        process_and_upload(key)

if __name__ == "__main__":
    main()
