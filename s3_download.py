import boto3
import os
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List

def list_s3_prefixes(bucket: str, prefix: str = '') -> List[str]:
    """
    Lists folders (prefixes) in an S3 bucket at a given prefix path
    
    Args:
        bucket: S3 bucket name
        prefix: The prefix path to list (default is root)
        
    Returns:
        List of prefixes (folders)
    """
    s3_client = boto3.client('s3')
    try:
        # Use delimiter to simulate folder-like behavior
        result = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
        prefixes = []
        
        # Get common prefixes (which represent folders)
        if 'CommonPrefixes' in result:
            prefixes = [p['Prefix'] for p in result['CommonPrefixes']]
            
        return prefixes
    except ClientError as e:
        print(f"Error listing prefixes: {e}")
        return []

def download_s3_folder(bucket: str, prefix: str, local_dir: str) -> None:
    """
    Downloads all files from an S3 folder to a local directory
    
    Args:
        bucket: S3 bucket name
        prefix: S3 folder path (prefix)
        local_dir: Local directory path to download files to
    """
    s3_client = boto3.client('s3')
    
    # Create local directory if it doesn't exist
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    
    # List all objects in the prefix
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Get the relative path of the file
                    key = obj['Key']
                    
                    # Skip if the object is a folder (ends with '/')
                    if key.endswith('/'):
                        continue
                    
                    # Calculate relative path for download
                    relative_path = key[len(prefix):] if prefix else key
                    local_file_path = os.path.join(local_dir, relative_path)
                    
                    # Create directories if they don't exist
                    os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                    
                    print(f"Downloading {key} to {local_file_path}")
                    s3_client.download_file(bucket, key, local_file_path)
        
        print(f"Downloaded folder {prefix} to {local_dir}")
    except ClientError as e:
        print(f"Error downloading folder: {e}")

def select_multiple_options(options: List[str], prompt: str) -> List[str]:
    """
    Custom function to allow multiple selection from a list of options
    
    Args:
        options: List of options to choose from
        prompt: Message to display before options
        
    Returns:
        List of selected options
    """
    if not options:
        return []
    
    print(prompt)
    for i, option in enumerate(options):
        print(f"[{i+1}] {option}")
    print("[0] Select all")
    
    selections = input("Enter your selections (comma-separated numbers, e.g., '1,3,5'): ")
    
    if selections.strip() == "0":
        return options
    
    try:
        selected_indices = [int(x.strip()) - 1 for x in selections.split(',') if x.strip()]
        return [options[i] for i in selected_indices if 0 <= i < len(options)]
    except (ValueError, IndexError):
        print("Invalid selection. Please try again.")
        return select_multiple_options(options, prompt)

def yes_no_prompt(question: str, default: bool = True) -> bool:
    """
    Ask a yes/no question and return the answer
    
    Args:
        question: Question to ask
        default: Default answer if user just presses Enter
        
    Returns:
        True for "yes" or False for "no"
    """
    default_str = "Y/n" if default else "y/N"
    response = input(f"{question} [{default_str}]: ").lower().strip()
    
    if not response:
        return default
    
    return response[0] == 'y'

def main():
    """
    Main function that provides interactive S3 folder selection and download
    """
    try:
        # Ask for S3 bucket name
        bucket_name = input("Enter the S3 bucket name: ")
        
        # Verify bucket exists
        s3 = boto3.client('s3')
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            print(f"Error: Bucket '{bucket_name}' does not exist or you don't have access.")
            return
        
        # First level folder selection
        root_folders = list_s3_prefixes(bucket_name)
        if not root_folders:
            print("No folders found in the root of the bucket.")
            # Ask if user wants to download the entire bucket
            download_all = yes_no_prompt("Do you want to download the entire bucket?")
            if download_all:
                download_s3_folder(bucket_name, '', os.path.join('downloads', bucket_name))
            return
        
        # Ask user to select first level folder(s)
        selected_level1 = select_multiple_options(
            root_folders, 
            "Select folders to download or explore (enter comma-separated numbers):"
        )
        
        if not selected_level1:
            print("No folders were selected. Exiting.")
            return
        
        # Process each selected first-level folder
        for folder in selected_level1:
            # List subfolders for second level selection
            subfolders = list_s3_prefixes(bucket_name, folder)
            
            if not subfolders:
                # No subfolders, ask to download this folder directly
                download_this = yes_no_prompt(f"'{folder}' has no subfolders. Download it?")
                if download_this:
                    local_dir = os.path.join('downloads', bucket_name, folder.rstrip('/'))
                    download_s3_folder(bucket_name, folder, local_dir)
                continue
            
            # Ask if user wants to explore subfolders or download the current folder
            explore_subfolders = yes_no_prompt(
                f"Do you want to explore subfolders in '{folder}'? (No will download the entire folder)"
            )
            
            if not explore_subfolders:
                local_dir = os.path.join('downloads', bucket_name, folder.rstrip('/'))
                download_s3_folder(bucket_name, folder, local_dir)
                continue
            
            # Ask user to select second level folders
            selected_level2 = select_multiple_options(
                subfolders,
                f"Select subfolders from '{folder}' to download:"  # nosec B608
            )
            
            # Download selected second level folders
            if selected_level2:
                for subfolder in selected_level2:
                    # Convert S3 prefix to local folder structure
                    relative_path = subfolder.rstrip('/')
                    local_dir = os.path.join('downloads', bucket_name, relative_path)
                    download_s3_folder(bucket_name, subfolder, local_dir)
            else:
                # None of the subfolders were selected, ask if user wants to download the parent folder
                download_parent = yes_no_prompt(
                    f"No subfolders selected. Do you want to download the entire '{folder}' folder?"
                )
                if download_parent:
                    local_dir = os.path.join('downloads', bucket_name, folder.rstrip('/'))
                    download_s3_folder(bucket_name, folder, local_dir)
        
        print("Download process completed.")
        
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your AWS credentials.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()