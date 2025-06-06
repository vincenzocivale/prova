import os
from huggingface_hub import login, hf_hub_download, snapshot_download
import pandas as pd
from tqdm import tqdm

def setup_huggingface(token):
    """
    Setup connection to Hugging Face using your token
    Args:
        token (str): Your Hugging Face token
    """
    login(token=token)

def get_metadata():
    """
    Download and return the metadata for the dataset
    Returns:
        pd.DataFrame: Metadata information
    """
    repo_id = "nonchev/TCGA_digital_spatial_transcriptomics"
    filename = "metadata_2025-05-21.csv"
    
    file_path = hf_hub_download(repo_id=repo_id, 
                               filename=filename, 
                               repo_type="dataset")
    return pd.read_csv(file_path)

def download_specific_sample(sample_path, output_dir='TCGA_data'):
    """
    Download a specific sample from the dataset
    Args:
        sample_path (str): Path to the specific sample
        output_dir (str): Directory to save the downloaded data
    """
    snapshot_download("nonchev/TCGA_digital_spatial_transcriptomics",
                     local_dir=output_dir,
                     allow_patterns=sample_path,
                     repo_type="dataset")

def download_cancer_type(cancer_type, slide_type=None, output_dir='data/raw', max_files=None):
    """
    Download data for a specific cancer type and optionally slide type
    Args:
        cancer_type (str): Cancer type (e.g., 'TCGA_KIRC', 'TCGA_SKCM')
        slide_type (str): Optional - 'FF' or 'FFPE'
        output_dir (str): Directory to save the downloaded data
        max_files (int): Optional - Maximum number of files to download
    """
    if slide_type:
        pattern = f"{cancer_type}/{slide_type}/*"
    else:
        pattern = f"{cancer_type}/*"
    
    snapshot_download("nonchev/TCGA_digital_spatial_transcriptomics",
                     local_dir=f"{output_dir}/TCGA_data/{cancer_type}",
                     allow_patterns=[pattern],
                     repo_type="dataset")
    
def download_first_n_samples(n, output_dir='data/raw'):
    """
    Download the first n samples from the dataset using metadata
    Args:
        n (int): Number of samples to download
        output_dir (str): Directory to save the downloaded data
    """
    # Get metadata
    metadata = get_metadata()
    
    # Get first n sample paths
    sample_paths = metadata['file_path'].head(n).tolist()
    
    # Download each sample with progress bar
    for path in tqdm(sample_paths, desc="Downloading samples"):
        download_specific_sample(path, output_dir=f"{output_dir}")
