from typing import Any, List, Dict, Optional
import os
import boto3
import json
from botocore.exceptions import ClientError, NoCredentialsError
import logging
from datetime import datetime
from config import MODEL_IDS
import tempfile

# Interactive mode helper functions
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

def select_option(options: List[str], prompt: str) -> Optional[str]:
    """
    Prompt user to select a single option from a list of options
    
    Args:
        options: List of options to choose from
        prompt: Message to display before options
        
    Returns:
        Selected option or None if no selection was made
    """
    if not options:
        return None
    
    print(prompt)
    for i, option in enumerate(options):
        print(f"[{i+1}] {option}")
    
    selection = input("Enter your selection (number): ")
    
    try:
        idx = int(selection) - 1
        if 0 <= idx < len(options):
            return options[idx]
        else:
            print("Invalid selection. Please try again.")
            return select_option(options, prompt)
    except ValueError:
        print("Invalid input. Please enter a number.")
        return select_option(options, prompt)

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

def get_input_with_default(prompt: str, default_value: Any) -> str:
    """
    Get user input with a default value
    
    Args:
        prompt: Prompt to display
        default_value: Default value to use if no input provided
        
    Returns:
        User input or default value
    """
    response = input(f"{prompt} [default: {default_value}]: ")
    if not response.strip():
        return str(default_value)
    return response

def interactive_s3_selection() -> Dict[str, str]:
    """
    Interactive selection of checkpoint from S3
    
    Returns:
        Dictionary with selected S3 information
    """
    selection = {}
    
    try:
        # Ask for S3 bucket name
        bucket_name = input("Enter the S3 bucket name: ")
        selection["bucket_name"] = bucket_name
        
        # Verify bucket exists
        s3 = boto3.client('s3')
        try:
            s3.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            print(f"Error: Bucket '{bucket_name}' does not exist or you don't have access.")
            return {}
        
        # First level folder selection
        root_folders = list_s3_prefixes(bucket_name)
        if not root_folders:
            print("No folders found in the root of the bucket.")
            return {}
        
        # Ask user to select first level folder
        selected_level1 = select_option(
            root_folders, 
            "Select experiment folder (experimental runs):"
        )
        
        if not selected_level1:
            print("No folder was selected. Exiting.")
            return {}
        
        selection["experiment_folder"] = selected_level1
        
        # List subfolders for second level selection (checkpoint directories)
        subfolders = list_s3_prefixes(bucket_name, selected_level1)
        
        if not subfolders:
            print(f"'{selected_level1}' has no checkpoint folders.")
            return {}
        
        # Ask user to select second level folder (checkpoint)
        selected_level2 = select_option(
            subfolders,
            f"Select checkpoint folder from '{selected_level1}':"  # nosec B608
        )
        
        if not selected_level2:
            print("No checkpoint folder was selected.")
            return {}
        
        selection["checkpoint_folder"] = selected_level2
        
        # Ask where to download the checkpoint
        local_checkpoint_dir = input("Enter local directory to download checkpoint to [default: ./checkpoints]: ")
        if not local_checkpoint_dir:
            local_checkpoint_dir = "./checkpoints"
        
        selection["local_checkpoint_dir"] = local_checkpoint_dir
        
        # Download the checkpoint
        local_path = os.path.join(local_checkpoint_dir, selected_level1.rstrip('/'), 
                                 os.path.basename(selected_level2.rstrip('/')))
        
        confirm_download = yes_no_prompt(f"Download checkpoint to {local_path}?")
        if confirm_download:
            download_s3_folder(bucket_name, selected_level2, local_path)
            selection["resume_from"] = local_path
        else:
            print("Download canceled.")
            return {}
        
        return selection
        
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your AWS credentials.")
        return {}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {}

def interactive_local_checkpoint_selection() -> Dict[str, str]:
    """
    Interactive selection of local checkpoint
    
    Returns:
        Dictionary with selected checkpoint path
    """
    selection = {}
    
    # Ask for local checkpoint directory
    checkpoint_dir = input("Enter path to checkpoint directory: ")
    
    if not os.path.exists(checkpoint_dir):
        print(f"Error: Directory '{checkpoint_dir}' does not exist.")
        return {}
    
    selection["resume_from"] = checkpoint_dir
    return selection

def interactive_config_setup(args) -> Dict:
    """
    Interactive configuration of training parameters
    
    Args:
        args: Default arguments from command line
    
    Returns:
        Dictionary with updated configuration
    """
    config = vars(args).copy()
    
    print("\n=== Model Configuration ===")
    if not args.resume_from and not args.load_from_checkpoint:
        config["model_name"] = get_input_with_default("Enter model name or path", args.model_name)
    
    # Model family
    model_families = ["llama", "qwen"]
    selected_family = select_option(model_families, "Select model family:")
    if selected_family:
        config["model_family"] = selected_family
    
    # LoRA configuration
    use_lora = yes_no_prompt("Use LoRA for parameter-efficient fine-tuning?", 
                            default=args.use_lora)
    config["use_lora"] = use_lora
    
    if use_lora:
        config["lora_r"] = int(get_input_with_default("LoRA rank (r)", args.lora_r))
        config["lora_alpha"] = int(get_input_with_default("LoRA alpha", args.lora_alpha))
        config["lora_dropout"] = float(get_input_with_default("LoRA dropout", args.lora_dropout))
        config["lora_target_modules"] = get_input_with_default("LoRA target modules (comma-separated)", 
                                                          args.lora_target_modules)
        
        use_quantization = yes_no_prompt("Use quantization with LoRA?", default=False)
        if use_quantization:
            quant_type = select_option(["4-bit", "8-bit"], "Select quantization type:")
            if quant_type == "4-bit":
                config["use_4bit"] = True
                config["use_8bit"] = False
            elif quant_type == "8-bit":
                config["use_4bit"] = False
                config["use_8bit"] = True
    
    print("\n=== Training Configuration ===")
    config["dpo_learning_rate"] = float(get_input_with_default("DPO learning rate", args.dpo_learning_rate))
    
    use_sft = yes_no_prompt("Use Supervised Fine-Tuning (SFT) after successful verification?", 
                           default=args.use_sft)
    config["use_sft"] = use_sft
    
    if use_sft:
        config["sft_learning_rate"] = float(get_input_with_default("SFT learning rate", args.sft_learning_rate))
    
    config["beta"] = float(get_input_with_default("DPO beta (regularization strength)", args.beta))
    config["epochs"] = int(get_input_with_default("Number of epochs", args.epochs))
    config["shuffle_dataset"] = yes_no_prompt("Shuffle dataset for each epoch?", 
                                            default=args.shuffle_dataset)

    print("\n=== SFT-only Mode ===")
    config["sft_only"] = yes_no_prompt("Use SFT-only mode? (skip reasoning improvement cycles)", default=False)
    
    print("\n=== Feedback Configuration ===")
    feedback_options = ["bedrock", "self"]
    selected_feedback = select_option(feedback_options, "Select feedback provider:")
    if selected_feedback:
        config["feedback_provider"] = selected_feedback
        
    if config["feedback_provider"] == "bedrock":
        print("\n=== AWS Bedrock Configuration ===")
        bedrock_models = list(MODEL_IDS.keys())
        print("Available Bedrock models:", ", ".join(bedrock_models))
        
        config["bedrock_model_basic"] = get_input_with_default("Basic Bedrock model for feedback", 
                                                            args.bedrock_model_basic)
        
        use_advanced_model = yes_no_prompt("Use different (advanced) model for bucket list problems?", 
                                         default=args.bedrock_model_advanced is not None)
        
        if use_advanced_model:
            config["bedrock_model_advanced"] = get_input_with_default("Advanced Bedrock model", 
                                                                  args.bedrock_model_advanced)
        else:
            config["bedrock_model_advanced"] = None
    else:
        print("\nUsing target model for self-feedback")
    
    # Add repeat_until_master option
    print("\n=== Repeat Until Master Configuration ===")
    repeat_until_master = select_option(["Yes", "No"], 
                                     "Repeat bucket problems until model masters them (gets correct on first try)?")
    config["repeat_until_master"] = (repeat_until_master == "Yes")
    
    print("\n=== Dataset Configuration ===")
    config["dataset_source"] = get_input_with_default("Dataset source (local path or HuggingFace dataset name)", args.dataset_source)
    
    # If it's not a local file, ask for dataset split
    if not os.path.exists(config["dataset_source"]):
        config["dataset_split"] = get_input_with_default("Dataset split (for HuggingFace datasets)", args.dataset_split)
    else:
        config["dataset_split"] = "train"  # Default value for local files
    
    use_max_samples = yes_no_prompt("Limit number of samples to use from the dataset?", 
                                  default=args.max_samples is not None)
    
    if use_max_samples:
        config["max_samples"] = int(get_input_with_default("Maximum number of samples", 
                                                       args.max_samples or 1000))
    else:
        config["max_samples"] = None
    
    config["batch_size"] = int(get_input_with_default("Batch size", args.batch_size))
    
    print("\n=== Output and Checkpointing ===")
    config["checkpoint_path"] = get_input_with_default("Path to save checkpoints", args.checkpoint_path)
    config["checkpoint_interval"] = int(get_input_with_default("Checkpoint interval", args.checkpoint_interval))
    config["delete_local_checkpoints"] = yes_no_prompt("Delete local checkpoints after upload to S3?", 
                                                     default=args.delete_local_checkpoints)
    
    use_wandb = yes_no_prompt("Use Weights & Biases logging?", 
                            default=args.wandb_project is not None)
    
    if use_wandb:
        config["wandb_project"] = get_input_with_default("W&B project name", 
                                                     args.wandb_project or "gsm8k-dpo")
    else:
        config["wandb_project"] = None
    
    # S3 bucket configuration
    print("\n=== S3 Storage Configuration ===")
    config["s3_bucket_name"] = get_input_with_default("S3 bucket name for saving checkpoints", 
                                                  args.s3_bucket_name)
    
    # Generate a default experiment name
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_name_short = config["model_name"].split('/')[-1] if not args.resume_from and not args.load_from_checkpoint else ""

    # Extract dataset name from the source path
    if os.path.exists(config["dataset_source"]):
        # For local files, use the filename without extension
        dataset_name = os.path.splitext(os.path.basename(config["dataset_source"]))[0]
    else:
        # For HuggingFace datasets, use the last part of the name
        dataset_name = config["dataset_source"].split('/')[-1]

    if args.resume_from:
        default_name = os.path.basename(os.path.dirname(os.path.normpath(args.resume_from)))
        if not default_name or default_name in ["checkpoints", "output"]:
            default_name = f"{timestamp}-{model_name_short}-{dataset_name}-resumed"
    else:
        default_name = f"{timestamp}-{model_name_short}-{dataset_name}"
        if config["use_lora"]:
            default_name += f"-lora-r{config['lora_r']}"
    
    # Let the user customize the experiment name
    experiment_name = input(f"Experiment name for S3 folder [default: {default_name}]: ")
    if not experiment_name.strip():
        experiment_name = default_name
    
    config["experiment_name"] = experiment_name
    
    # Show the full S3 path that will be used
    s3_path = f"s3://{config['s3_bucket_name']}/{experiment_name}"
    print(f"\nResults will be saved to: {s3_path}")
    
    # Final confirmation
    print("\n=== Configuration Summary ===")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    confirm = yes_no_prompt("\nStart training with this configuration?")
    if not confirm:
        print("Training canceled by user.")
        return None
    
    return config

# Set up logging
def setup_logger(log_dir="./logs", experiment_name=None):
    """
    Configure logging to file and console
    
    Args:
        log_dir: Directory to store log files
        experiment_name: Experiment name for log file naming
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Use experiment name for log file if provided, otherwise use timestamp
    if experiment_name:
        log_file = os.path.join(log_dir, f"{experiment_name}.log")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"dpo_training_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('dpo_training')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

#####################################################
#         Save training artifacts to S3             #
#####################################################

def save_checkpoint(model, tokenizer, metrics_tracker, remaining_problems, checkpoint_dir, s3_path=None, delete_local=False, logger=None):
    """
    Save model checkpoint and related metadata to disk and optionally to S3
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        metrics_tracker: Metrics tracker object
        remaining_problems: List of remaining problem IDs and bucket list problems
        checkpoint_dir: Directory to save checkpoint to
        s3_path: Optional S3 path to upload checkpoint to
        delete_local: Whether to delete local checkpoint after uploading to S3
        logger: Logger instance
    
    Returns:
        Path to saved checkpoint
    """
    logger = logger or logging.getLogger('dpo_training')
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Check if model is a PEFT model and save accordingly
    is_peft_model = hasattr(model, "save_pretrained") and hasattr(model, "create_or_update_model_card")
    
    if is_peft_model:
        logger.info(f"Saving PEFT model adapters to {checkpoint_dir}")
        model.save_pretrained(checkpoint_dir)
    else:
        # Save model and tokenizer
        logger.info(f"Saving full model and tokenizer to {checkpoint_dir}")
        model.save_pretrained(checkpoint_dir)
        
    # Always save tokenizer
    tokenizer.save_pretrained(checkpoint_dir)
    
    # Save metrics history
    metrics_path = os.path.join(checkpoint_dir, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_tracker.all_metrics, f)
    
    # Save list of remaining problems
    remaining_path = os.path.join(checkpoint_dir, "remaining_problems.json")
    with open(remaining_path, "w") as f:
        json.dump(remaining_problems, f)
    
    logger.info(f"Saved checkpoint metadata ({len(remaining_problems['dataset'])} remaining dataset problems, "
               f"{len(remaining_problems['bucket'])} bucket problems)")
    
    # Upload to S3 if path is provided
    if s3_path:
        save_to_s3(checkpoint_dir, s3_path, logger=logger, delete_local=delete_local)
    
    return checkpoint_dir

def save_to_s3(local_path, s3_path, logger=None, delete_local=False):
    """
    Save checkpoint files from local path to S3 bucket
    
    Args:
        local_path: Local directory with checkpoint files
        s3_path: S3 path to upload the files to
        logger: Logger instance
        delete_local: Whether to delete local files after upload
    """
    logger = logger or logging.getLogger('dpo_training')
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Extract bucket name and prefix from s3_path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
            
        bucket_name = s3_path.split('/')[0]
        prefix = '/'.join(s3_path.split('/')[1:])
        
        # List files in the local directory
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Calculate relative path for S3 key
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = f"{prefix}/{relative_path}"
                
                # Upload file to S3
                logger.info(f"Uploading {local_file_path} to s3://{bucket_name}/{s3_key}")
                s3_client.upload_file(local_file_path, bucket_name, s3_key)
        
        logger.info(f"Successfully uploaded checkpoint from {local_path} to s3://{bucket_name}/{prefix}")
        
        # Delete local files if requested
        if delete_local:
            import shutil
            logger.info(f"Deleting local checkpoint folder: {local_path}")
            shutil.rmtree(local_path)
            logger.info(f"Local checkpoint folder deleted")
        
    except Exception as e:
        logger.error(f"Error uploading checkpoint to S3: {e}")
        # Don't delete local files if there was an error uploading
        if delete_local:
            logger.warning(f"Not deleting local files due to upload error")

def save_args_to_s3(args, s3_path, logger=None):
    """
    Save argument information to S3 bucket
    
    Args:
        args: Command line arguments
        s3_path: S3 path to upload to
        logger: Logger instance
    """
    logger = logger or logging.getLogger('dpo_training')
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Extract bucket name and prefix from s3_path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
            
        bucket_name = s3_path.split('/')[0]
        prefix = '/'.join(s3_path.split('/')[1:])
        
        # Convert args to dictionary and then to JSON
        args_dict = vars(args)
        args_json = json.dumps(args_dict, indent=2)
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            temp_file.write(args_json)
            temp_file.flush()
            temp_file_path = temp_file.name
        
        # Upload to S3
        s3_key = f"{prefix}/args.json"
        logger.info(f"Uploading arguments to s3://{bucket_name}/{s3_key}")
        s3_client.upload_file(temp_file_path, bucket_name, s3_key)
        
        # Clean up temporary file
        os.unlink(temp_file_path)
        
        logger.info(f"Successfully uploaded arguments to S3")
        
    except Exception as e:
        logger.error(f"Error uploading arguments to S3: {e}")

def save_logs_to_s3(log_dir, s3_path, logger=None, experiment_name=None):
    """
    Save log files to S3 bucket
    
    Args:
        log_dir: Local directory with log files
        s3_path: S3 path to upload to
        logger: Logger instance
        experiment_name: Optional experiment name to filter specific log files
    """
    logger = logger or logging.getLogger('dpo_training')
    
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Extract bucket name and prefix from s3_path
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
            
        bucket_name = s3_path.split('/')[0]
        prefix = '/'.join(s3_path.split('/')[1:])
        
        # List log files
        if experiment_name:
            log_files = [f for f in os.listdir(log_dir) if f.startswith(experiment_name) and f.endswith('.log')]
        else:
            log_files = [f for f in os.listdir(log_dir) if f.endswith('.log')]
        
        for log_file in log_files:
            local_path = os.path.join(log_dir, log_file)
            s3_key = f"{prefix}/logs/{log_file}"
            
            # Upload to S3
            logger.info(f"Uploading log file {log_file} to s3://{bucket_name}/{s3_key}")
            s3_client.upload_file(local_path, bucket_name, s3_key)
        
        logger.info(f"Successfully uploaded log files to S3")
        
    except Exception as e:
        logger.error(f"Error uploading logs to S3: {e}")