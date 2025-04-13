#!/usr/bin/env python3
# fairface_downloader.py

import os
import argparse
import zipfile
import gdown
import pandas as pd
from tqdm import tqdm

def download_fairface_dataset(output_dir='dataset'):
    """
    Download and extract the FairFace dataset.
    
    Args:
        output_dir: Directory to save the dataset
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Downloading FairFace dataset...")
    
    # URLs for the FairFace dataset files
    # Using Google Drive direct download links
    dataset_urls = {
        'fairface_images': 'https://drive.google.com/uc?id=1Z1RqRo0_JiavaZw2yzZG6WETdZQ8qX86',
        'fairface_label_train': 'https://drive.google.com/uc?id=1i1L3Yqwaio7YSOCj7ftgk8ZZchPG7dmH',
        'fairface_label_val': 'https://drive.google.com/uc?id=1wOdja-ezstMEp81tX1a-EYkFebev4h7D',
    }
    
    # Local file paths
    local_files = {
        'fairface_images': os.path.join(output_dir, 'fairface_images.zip'),
        'fairface_label_train': os.path.join(output_dir, 'fairface_label_train.csv'),
        'fairface_label_val': os.path.join(output_dir, 'fairface_label_val.csv')
    }
    
    # Download files
    for name, url in dataset_urls.items():
        if name.endswith('images'):
            # Check if images are already downloaded
            if os.path.exists(os.path.join(output_dir, 'train')) and os.path.exists(os.path.join(output_dir, 'val')):
                print(f"Image directories already exist, skipping download of {name}")
                continue
                
            # Check if zip file already exists
            if os.path.exists(local_files[name]):
                print(f"Zip file {local_files[name]} already exists, skipping download")
            else:
                print(f"Downloading {name}...")
                gdown.download(url, local_files[name], quiet=False)
            
            # Extract zip file
            print(f"Extracting {name}...")
            with zipfile.ZipFile(local_files[name], 'r') as zip_ref:
                zip_ref.extractall(output_dir)
                
            # Remove zip file to save space
            os.remove(local_files[name])
        else:
            # Download CSV files
            if os.path.exists(local_files[name]):
                print(f"File {local_files[name]} already exists, skipping download")
            else:
                print(f"Downloading {name}...")
                gdown.download(url, local_files[name], quiet=False)
    
    # Process and clean up labels
    print("Processing dataset labels...")
    preprocess_fairface_labels(output_dir)
    
    print("FairFace dataset download and processing complete!")

def preprocess_fairface_labels(dataset_dir):
    """
    Preprocess and standardize the FairFace labels.
    
    Args:
        dataset_dir: Directory containing the dataset
    """
    # Read the CSV files
    train_csv_path = os.path.join(dataset_dir, 'fairface_label_train.csv')
    val_csv_path = os.path.join(dataset_dir, 'fairface_label_val.csv')
    
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    # Standardize column names and required formats
    for df in [train_df, val_df]:
        # Rename columns if needed
        if 'file' in df.columns and 'file_name' not in df.columns:
            df.rename(columns={'file': 'file_name'}, inplace=True)
            
        # Convert race_scores_fair columns to a single race column if needed
        race_cols = [col for col in df.columns if col.startswith('race_score')]
        if race_cols and 'race' not in df.columns:
            # Get the race with the highest score
            race_mapping = {
                0: 'White', 1: 'Black', 2: 'East Asian', 
                3: 'Southeast Asian', 4: 'Indian', 
                5: 'Middle Eastern', 6: 'Latino_Hispanic'
            }
            
            race_scores = df[race_cols].values
            max_race_idx = race_scores.argmax(axis=1)
            df['race'] = [race_mapping[idx] for idx in max_race_idx]
    
    # Save the processed files
    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    
    # Create a combined CSV for convenience
    combined_df = pd.concat([train_df, val_df])
    combined_df.to_csv(os.path.join(dataset_dir, 'fairface_label_combined.csv'), index=False)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Total samples: {len(combined_df)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and extract the FairFace dataset')
    parser.add_argument('--output_dir', type=str, default='dataset', help='Directory to save the dataset')
    
    args = parser.parse_args()
    download_fairface_dataset(args.output_dir)
