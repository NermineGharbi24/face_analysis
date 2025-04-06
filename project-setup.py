#!/usr/bin/env python3
# project-setup.py

import os
import argparse
import subprocess
import sys
import urllib.request
from tqdm import tqdm

def setup_project(download_dataset=True):
    """
    Set up the face analysis project by creating the directory structure, installing dependencies,
    and downloading required files.
    
    Args:
        download_dataset: Whether to download the FairFace dataset
    """
    # Create directory structure
    os.makedirs('models', exist_ok=True)
    
    # Create virtual environment
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"])
    
    # Determine pip command based on operating system
    if os.name == 'nt':  # Windows
        pip_cmd = os.path.join("venv", "Scripts", "pip")
        python_cmd = os.path.join("venv", "Scripts", "python")
    else:  # Unix/Linux/Mac
        pip_cmd = os.path.join("venv", "bin", "pip")
        python_cmd = os.path.join("venv", "bin", "python")
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"])
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"])
    
    # Download face detector model files
    print("Downloading face detector model files...")
    
    # URLs for the model files
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    caffemodel_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    
    # Download files
    prototxt_path = os.path.join("models", "deploy.prototxt")
    caffemodel_path = os.path.join("models", "res10_300x300_ssd_iter_140000.caffemodel")
    
    with tqdm(unit='B', unit_scale=True, desc="Downloading prototxt") as pbar:
        urllib.request.urlretrieve(
            prototxt_url, prototxt_path,
            reporthook=lambda b, bs, ts: pbar.update(bs if pbar.n + bs <= ts else ts - pbar.n)
        )
    
    with tqdm(unit='B', unit_scale=True, desc="Downloading caffemodel") as pbar:
        urllib.request.urlretrieve(
            caffemodel_url, caffemodel_path,
            reporthook=lambda b, bs, ts: pbar.update(bs if pbar.n + bs <= ts else ts - pbar.n)
        )
    
    # Download FairFace dataset
    if download_dataset:
        print("\nDownloading FairFace dataset...")
        subprocess.run([python_cmd, "fairface_downloader.py", "--output_dir", "dataset"])
        
        # Train model with downloaded dataset
        print("\nTraining the model on FairFace dataset...")
        subprocess.run([
            python_cmd, "train.py", 
            "--csv_file", "dataset/fairface_label_train.csv",
            "--val_csv_file", "dataset/fairface_label_val.csv",
            "--img_dir", "dataset",
            "--output_dir", "models",
            "--epochs", "20"  # Reduce epochs for faster training; can increase for better accuracy
        ])
    
    print("\nProject setup completed successfully!")
    print("\nTo activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("    venv\\Scripts\\activate")
    else:  # Unix/Linux/Mac
        print("    source venv/bin/activate")
    
    print("\nTo test the face analysis system with a sample image:")
    print("    python main.py image --input <path_to_image> --show")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up the Face Analysis project')
    parser.add_argument('--skip_dataset', action='store_true', help='Skip downloading the FairFace dataset')
    
    args = parser.parse_args()
    
    setup_project(not args.skip_dataset)