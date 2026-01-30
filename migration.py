import boto3
import os
import tempfile
import shutil
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Optional

def list_s3_prefixes(bucket: str, prefix: str = '', s3_client=None) -> List[str]:
    """
    Lists folders (prefixes) in an S3 bucket at a given prefix path
    
    Args:
        bucket: S3 bucket name
        prefix: The prefix path to list (default is root)
        s3_client: Optional S3 client to use
        
    Returns:
        List of prefixes (folders)
    """
    if s3_client is None:
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

def create_destination_s3_client() -> Optional[boto3.client]:
    """
    Creates an S3 client for the destination bucket using user-provided credentials
    
    Returns:
        boto3.client: Configured S3 client or None if configuration failed
    """
    print("\n=== Destination S3 Configuration ===")
    print("Please enter credentials for the destination S3 bucket:")
    
    aws_access_key = input("AWS Access Key ID: ")
    aws_secret_key = input("AWS Secret Access Key: ")
    aws_region = input("AWS Region (default: us-east-1): ") or "us-east-1"
    
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Test the connection
        s3_client.list_buckets()
        print("✓ Destination S3 connection successful")
        return s3_client
    except Exception as e:
        print(f"Error creating destination S3 client: {e}")
        return None

def clean_temp_directory(temp_dir):
    """
    Cleans up temporary directory to free up space
    
    Args:
        temp_dir: Path to temporary directory
    """
    try:
        # Remove all files in temp directory
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)
                
        print(f"✓ Temporary directory cleaned: {temp_dir}")
    except Exception as e:
        print(f"Error cleaning temporary directory: {e}")

def copy_s3_folder(source_bucket: str, source_prefix: str, 
                  dest_bucket: str, dest_prefix: str,
                  source_s3=None, dest_s3=None) -> None:
    """
    Copies files from a source S3 folder to a destination S3 folder
    
    Args:
        source_bucket: Source S3 bucket name
        source_prefix: Source S3 folder path (prefix)
        dest_bucket: Destination S3 bucket name
        dest_prefix: Destination S3 folder path (prefix)
        source_s3: Source S3 client
        dest_s3: Destination S3 client
    """
    if source_s3 is None:
        source_s3 = boto3.client('s3')
    
    if dest_s3 is None:
        dest_s3 = boto3.client('s3')
    
    # Make sure prefixes have trailing slashes if they're not empty
    if source_prefix and not source_prefix.endswith('/'):
        source_prefix = f"{source_prefix}/"
    
    if dest_prefix and not dest_prefix.endswith('/'):
        dest_prefix = f"{dest_prefix}/"
    
    # Create a temporary directory for file transfers
    temp_dir = tempfile.mkdtemp()
    print(f"Created temporary directory: {temp_dir}")
    
    # Track number of processed files for periodic cleanup
    file_count = 0
    cleanup_threshold = 50  # Clean up after every 50 files
    
    # List all objects in the source prefix
    try:
        paginator = source_s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=source_bucket, Prefix=source_prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    # Get the source key
                    source_key = obj['Key']
                    
                    # Skip if the object is a folder (ends with '/')
                    if source_key.endswith('/'):
                        continue
                    
                    # Calculate the destination key
                    relative_path = source_key[len(source_prefix):] if source_prefix else source_key
                    dest_key = f"{dest_prefix}{relative_path}"
                    
                    print(f"Copying {source_bucket}/{source_key} -> {dest_bucket}/{dest_key}")
                    
                    # Use temporary file to transfer between different S3 accounts
                    try:
                        temp_file_path = os.path.join(temp_dir, os.path.basename(source_key))
                        
                        # Download from source
                        source_s3.download_file(source_bucket, source_key, temp_file_path)
                        
                        # Upload to destination
                        dest_s3.upload_file(temp_file_path, dest_bucket, dest_key)
                        
                        # Clean up individual temp file immediately after use
                        os.remove(temp_file_path)
                        
                        # Increment file counter
                        file_count += 1
                        
                        # Periodic cleanup check
                        if file_count % cleanup_threshold == 0:
                            print(f"Processed {file_count} files. Cleaning up temporary directory...")
                            clean_temp_directory(temp_dir)
                            
                    except Exception as e:
                        print(f"Error copying file {source_key}: {e}")
        
        # Final cleanup
        clean_temp_directory(temp_dir)
        
        # Remove temp directory
        try:
            os.rmdir(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except:
            print(f"Could not remove temporary directory: {temp_dir}")
            
        print(f"\n✓ Successfully copied folder {source_prefix} to {dest_bucket}/{dest_prefix}")
    except ClientError as e:
        print(f"Error copying folder: {e}")
    finally:
        # Make sure temp directory is removed even if an error occurs
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except:
                print(f"Warning: Could not remove temporary directory: {temp_dir}")

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

def copy_entire_bucket(source_bucket: str, dest_bucket: str, dest_prefix: str,
                      source_s3=None, dest_s3=None) -> None:
    """
    Copies the entire bucket content to the destination
    
    Args:
        source_bucket: Source bucket name
        dest_bucket: Destination bucket name
        dest_prefix: Base prefix to use in destination (can be empty)
        source_s3: Source S3 client
        dest_s3: Destination S3 client
    """
    print(f"\n=== Copying entire bucket '{source_bucket}' to '{dest_bucket}/{dest_prefix}' ===")
    
    # Ensure dest_prefix ends with a slash if not empty
    if dest_prefix and not dest_prefix.endswith('/'):
        dest_prefix += '/'
    
    # List top-level folders for individual processing
    top_folders = list_s3_prefixes(source_bucket, '', source_s3)
    if top_folders:
        for folder in top_folders:
            print(f"\nProcessing top folder: {folder}")
            # Copy each top-level folder separately to allow memory cleanup between folders
            folder_dest_prefix = f"{dest_prefix}{folder}" if dest_prefix else folder
            copy_s3_folder(source_bucket, folder, dest_bucket, folder_dest_prefix, source_s3, dest_s3)
            
            # Ask to continue to next folder
            if folder != top_folders[-1]:  # If not the last folder
                continue_copy = True #yes_no_prompt(
                    # "Continue to next folder? If no, the process will exit and can be resumed later.",
                    # default=True
                # )
                if not continue_copy:
                    print("Copy process paused. You can run the script again later to resume.")
                    return
    else:
        # No folders, just copy files at root level
        copy_s3_folder(source_bucket, '', dest_bucket, dest_prefix, source_s3, dest_s3)

def main():
    """
    Main function that provides interactive S3 folder selection and copy
    """
    try:
        # Use existing S3 client for source
        source_s3 = boto3.client('s3')
        
        # Ask for source S3 bucket name
        source_bucket_name = input("Enter the source S3 bucket name: ")
        
        # Verify source bucket exists
        try:
            source_s3.head_bucket(Bucket=source_bucket_name)
        except ClientError as e:
            print(f"Error: Source bucket '{source_bucket_name}' does not exist or you don't have access.")
            return
        
        # Configure destination S3 client with new credentials
        dest_s3 = create_destination_s3_client()
        if not dest_s3:
            print("Failed to configure destination S3 client. Exiting.")
            return
        
        # Ask for destination bucket name
        dest_bucket_name = input("Enter the destination S3 bucket name: ")
        
        # Verify destination bucket exists
        try:
            dest_s3.head_bucket(Bucket=dest_bucket_name)
        except ClientError as e:
            print(f"Error: Destination bucket '{dest_bucket_name}' does not exist or you don't have access.")
            
            # Ask if user wants to create the bucket
            create_bucket = yes_no_prompt("Do you want to create this bucket?", default=False)
            if create_bucket:
                region = dest_s3.meta.region_name
                try:
                    if region == 'us-east-1':
                        dest_s3.create_bucket(Bucket=dest_bucket_name)
                    else:
                        dest_s3.create_bucket(
                            Bucket=dest_bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': region}
                        )
                    print(f"✓ Bucket '{dest_bucket_name}' created successfully")
                except ClientError as e:
                    print(f"Error creating bucket: {e}")
                    return
            else:
                return
        
        # Ask if user wants to copy the entire bucket automatically
        auto_copy = yes_no_prompt(
            "Do you want to automatically copy the entire bucket? (No will proceed to interactive selection)",
            default=False
        )
        
        if auto_copy:
            # Prompt for a base destination prefix
            dest_prefix = input("Enter destination prefix for all files [leave empty for bucket root]: ")
            copy_entire_bucket(source_bucket_name, dest_bucket_name, dest_prefix, source_s3, dest_s3)
            print("Automatic copy completed.")
            return
        
        # First level folder selection
        root_folders = list_s3_prefixes(source_bucket_name, s3_client=source_s3)
        if not root_folders:
            print("No folders found in the root of the source bucket.")
            # Ask if user wants to copy the entire bucket
            copy_all = yes_no_prompt("Do you want to copy the entire bucket?")
            if copy_all:
                copy_s3_folder(source_bucket_name, '', dest_bucket_name, '', source_s3, dest_s3)
            return
        
        # Ask user to select first level folder(s)
        selected_level1 = select_multiple_options(
            root_folders, 
            "Select folders to copy or explore (enter comma-separated numbers):"
        )
        
        if not selected_level1:
            print("No folders were selected. Exiting.")
            return
        
        # Process each selected first-level folder
        for folder in selected_level1:
            # List subfolders for second level selection
            subfolders = list_s3_prefixes(source_bucket_name, folder, source_s3)
            
            if not subfolders:
                # No subfolders, ask to copy this folder directly
                copy_this = yes_no_prompt(f"'{folder}' has no subfolders. Copy it?")
                if copy_this:
                    dest_prefix = input(f"Enter destination prefix for '{folder}' [leave empty to use same path]: ")
                    if not dest_prefix:
                        dest_prefix = folder
                    copy_s3_folder(source_bucket_name, folder, dest_bucket_name, dest_prefix, source_s3, dest_s3)
                continue
            
            # Ask if user wants to explore subfolders or copy the current folder
            explore_subfolders = yes_no_prompt(
                f"Do you want to explore subfolders in '{folder}'? (No will copy the entire folder)"
            )
            
            if not explore_subfolders:
                dest_prefix = input(f"Enter destination prefix for '{folder}' [leave empty to use same path]: ")
                if not dest_prefix:
                    dest_prefix = folder
                copy_s3_folder(source_bucket_name, folder, dest_bucket_name, dest_prefix, source_s3, dest_s3)
                continue
            
            # Ask user to select second level folders
            selected_level2 = select_multiple_options(
                subfolders,
                f"Select subfolders from '{folder}' to copy:"  # nosec B608
            )
            
            # Copy selected second level folders
            if selected_level2:
                for subfolder in selected_level2:
                    dest_prefix = input(f"Enter destination prefix for '{subfolder}' [leave empty to use same path]: ")
                    if not dest_prefix:
                        dest_prefix = subfolder
                    copy_s3_folder(source_bucket_name, subfolder, dest_bucket_name, dest_prefix, source_s3, dest_s3)
            else:
                # None of the subfolders were selected, ask if user wants to copy the parent folder
                copy_parent = yes_no_prompt(
                    f"No subfolders selected. Do you want to copy the entire '{folder}' folder?"
                )
                if copy_parent:
                    dest_prefix = input(f"Enter destination prefix for '{folder}' [leave empty to use same path]: ")
                    if not dest_prefix:
                        dest_prefix = folder
                    copy_s3_folder(source_bucket_name, folder, dest_bucket_name, dest_prefix, source_s3, dest_s3)
        
        print("Copy process completed.")
        
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your AWS credentials.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()