import os
import random
import logging
import torch
import wandb
import argparse
import numpy as np
import re
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Any, List, Dict, Optional
import json
from datetime import datetime

from feedback import FeedbackProvider
from model_utils import load_model, add_lora_adapter, load_checkpoint_with_lora, load_checkpoint_metadata
from interactive_reasoning import run_dpo_step, perform_sft_training
from metrics import MetricTracker, restore_metrics_tracker, plot_metrics
from dataset import load_dataset_for_training_or_eval, format_dataset_for_eval
from utils import (select_option, interactive_s3_selection, interactive_config_setup, interactive_local_checkpoint_selection,
                   setup_logger, save_to_s3, save_checkpoint, save_args_to_s3, save_logs_to_s3)
from config import MODEL_IDS

def parse_args():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train models with DPO and SFT on GSM8K")
    
    # Interactive mode argument
    parser.add_argument("--interactive", action="store_true",
                      help="Start in interactive mode to configure training options")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-1B-Instruct",
                        help="Name or path of the model to train")
    parser.add_argument("--load_from_checkpoint", type=str, default=None,
                        help="Load model weights from a local checkpoint directory (without resuming training state)")
    parser.add_argument("--model_family", type=str, choices=["llama", "qwen"], default="llama",
                        help="Model family: 'llama' or 'qwen'")
    parser.add_argument("--trust_remote_code", action="store_true", 
                        help="Whether to trust remote code when loading models")
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device_index", type=int, default=0, help="CUDA device index")
    parser.add_argument("--dpo_learning_rate", type=float, default=5e-7, help="Learning rate for DPO")
    parser.add_argument("--use_sft", type=int, default=1, help="Whether to use SFT after successful verification")
    parser.add_argument("--sft_learning_rate", type=float, default=1e-6, help="Learning rate for SFT")
    parser.add_argument("--beta", type=float, default=0.1, 
                        help="DPO beta parameter (regularization strength)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to train for")
    parser.add_argument("--shuffle_dataset", action="store_true",
                        help="Whether to shuffle the dataset for each epoch")
    parser.add_argument("--process_bucket", type=bool, default=True,
                        help="Whether to process the bucket list after each batch")
    # SFT only argument
    parser.add_argument("--sft_only", action="store_true",
                    help="Only perform SFT training without DPO (skip improvement cycles)")
    
    # LoRA arguments
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA adaptation")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--lora_target_modules", type=str, default="q_proj,v_proj,k_proj,o_proj", 
                        help="Comma-separated list of target modules for LoRA")
    parser.add_argument("--use_4bit", action="store_true", help="Whether to use 4-bit quantization with LoRA")
    parser.add_argument("--use_8bit", action="store_true", help="Whether to use 8-bit quantization with LoRA")
    parser.add_argument("--use_gradient_checkpointing", action="store_true", 
                        help="Use gradient checkpointing for memory efficiency")
    
    # AWS Bedrock arguments
    parser.add_argument("--bedrock_model_basic", type=str, default="c35_haiku", 
                       choices=list(MODEL_IDS.keys()),
                       help="Basic AWS Bedrock model for feedback on standard problems")
    parser.add_argument("--bedrock_model_advanced", type=str, default="c37_sonnet",
                       choices=list(MODEL_IDS.keys()) + [None],
                       help="Advanced AWS Bedrock model for feedback on bucket list problems (uses basic model if None)")
    parser.add_argument("--bedrock_max_retries", type=int, default=3,
                       help="Maximum retries for Bedrock API calls")
    
    # Generation arguments
    parser.add_argument("--improvement_generator", type=str, choices=["bedrock", "self"], default="bedrock",
                        help="Which model to use for generating improvements: 'bedrock' or 'self' model")
    parser.add_argument("--improvement_max_attempts", type=int, default=3, 
                       help="Maximum attempts to improve an answer until format and answer are correct")
    parser.add_argument("--verification_max_attempts", type=int, default=5,
                       help="Maximum attempts for post-DPO verification and improvement")
    parser.add_argument("--repeat_until_master", action="store_true", 
                    help="Repeat bucket problems until model masters them (gets correct on first try)")
    # Feedback provider arguments
    parser.add_argument("--feedback_provider", type=str, choices=["bedrock", "self"], default="bedrock",
                        help="Which model to use for providing feedback: 'bedrock' or 'self' (target model)")
    
    # Dataset arguments
    parser.add_argument("--dataset_source", type=str, default="gsm8k",
                        help="Dataset source (local file path or HuggingFace dataset name)")
    parser.add_argument("--dataset_split", type=str, default="train",
                        help="Dataset split for HuggingFace datasets")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to use from the dataset (None for all)")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of problems to process in each batch")
    
    # Output arguments
    parser.add_argument("--checkpoint_path", type=str, default="./interactive_reasoning_output",
                        help="Path to save model checkpoints")
    parser.add_argument("--checkpoint_interval", type=int, default=50,
                        help="Number of problems between checkpoints (includes bucket list problems)")
    parser.add_argument("--keep_local_checkpoints", action="store_false", dest="delete_local_checkpoints",
                        help="Keep local checkpoint folders after uploading to S3 (default is to delete them)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Weights & Biases project name (if None, WandB is disabled)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint directory to resume training from (including training state)")
    
    # Visualization arguments
    parser.add_argument("--vis_dir", type=str, default="./visualizations_training",
                        help="Directory to save visualizations")
    parser.add_argument("--vis_interval", type=int, default=25,
                        help="Number of problems between visualizations (includes bucket list problems)")

    # s3 bucket arguments
    parser.add_argument("--s3_bucket_name", type=str, default="paper-experiments",
                        help="s3 bucket name")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default="./logs",
                        help="Directory to save log files")
    
    return parser.parse_args()

def main():
    """Main function to run DPO training on GSM8K with the iterative feedback approach"""
    args = parse_args()
    
    # Check for interactive mode
    if args.interactive:
        print("\n=== Interactive DPO Training Setup ===")
        
        # Ask whether to start from S3 or local checkpoint
        start_option = select_option([
            "Start from S3 bucket checkpoint", 
            "Start from local checkpoint",
            "Start with new model (no checkpoint)"
        ], "Select starting point:")
        
        # Initialize options dictionary from args
        selected_options = {}
        
        # Handle different starting options
        if start_option == "Start from S3 bucket checkpoint":
            print("\nSelecting checkpoint from S3...")
            s3_options = interactive_s3_selection()
            if s3_options:
                # Set resume_from to the downloaded checkpoint path
                selected_options["resume_from"] = s3_options.get("resume_from")
                selected_options["s3_bucket_name"] = s3_options.get("bucket_name")
        elif start_option == "Start from local checkpoint":
            print("\nSelecting local checkpoint...")
            local_options = interactive_local_checkpoint_selection()
            if local_options:
                selected_options["resume_from"] = local_options.get("resume_from")
        else:
            print("\nStarting with a new model (no checkpoint)...")
            
        # Configure other parameters interactively
        config = interactive_config_setup(args)
        
        # If user canceled setup, exit
        if config is None:
            print("Training canceled. Exiting.")
            return
        
        # Update args with selected options
        for key, value in selected_options.items():
            setattr(args, key, value)
            
        # Update args with configured options
        for key, value in config.items():
            setattr(args, key, value)
    
    # Generate timestamp for the whole run
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # Determine experiment name - use the one from interactive config if available
    if args.interactive and hasattr(args, "experiment_name"):
        experiment_name = args.experiment_name
    else:
        # Default experiment name generation logic
        if args.load_from_checkpoint:
            model_name = os.path.basename(os.path.normpath(args.load_from_checkpoint))
        else:
            model_name = args.model_name.split('/')[-1]
        
        # Extract dataset name from the source path
        if os.path.exists(args.dataset_source):
            # For local files, use the filename without extension
            dataset_name = os.path.splitext(os.path.basename(args.dataset_source))[0]
        else:
            # For HuggingFace datasets, use the last part of the name
            dataset_name = args.dataset_source.split('/')[-1]
        
        if args.resume_from:
            # Extract experiment name from checkpoint path
            checkpoint_dir = args.resume_from
            experiment_name = os.path.basename(os.path.normpath(checkpoint_dir))
            if "checkpoint" in experiment_name:
                # Use parent directory name instead
                experiment_name = os.path.basename(os.path.dirname(os.path.normpath(checkpoint_dir)))
        else:
            # Generate new experiment name using dataset name instead of hardcoded "gsm8k"
            experiment_name = f"{timestamp}-{model_name}-{dataset_name}-{args.feedback_provider}-{args.improvement_generator}-{args.bedrock_model_basic}-{args.bedrock_model_advanced}"
            if args.sft_only:
                experiment_name = f"{timestamp}-{model_name}-{dataset_name}-sft-only-lr-{args.sft_learning_rate}-epoch-{args.epochs}"
            # Add LoRA indication if using LoRA
            if args.use_lora:
                experiment_name += f"-lora-r{args.lora_r}"
    
    # Create experiment-specific vis_dir
    experiment_vis_dir = os.path.join(args.vis_dir, experiment_name)
    os.makedirs(experiment_vis_dir, exist_ok=True)
    
    # Set up logging with the experiment name
    logger = setup_logger(args.log_dir, experiment_name)
    logger.info(f"Experiment name: {experiment_name}")
    
    if args.resume_from:
        logger.info(f"Resuming training from checkpoint: {args.resume_from}")
    elif args.load_from_checkpoint:
        logger.info(f"Loading model weights from checkpoint: {args.load_from_checkpoint} (without resuming training state)")
    else:
        logger.info("Starting new Interactive Reasoning training")
        
    logger.info(f"Arguments: {args}")
    logger.info(f"Using improvement generator: {args.improvement_generator}")
    logger.info(f"Basic Bedrock model: {args.bedrock_model_basic}")
    logger.info(f"Advanced Bedrock model for bucket list problems: {args.bedrock_model_advanced}")
    logger.info(f"SFT enabled: {args.use_sft}")
    logger.info(f"Number of epochs: {args.epochs}")
    logger.info(f"Random shuffling: {args.shuffle_dataset}")
    logger.info(f"Delete local checkpoints after upload: {args.delete_local_checkpoints}")
    
    # Log LoRA configuration if enabled
    if args.use_lora:
        logger.info(f"Using LoRA with rank: {args.lora_r}, alpha: {args.lora_alpha}")
        logger.info(f"LoRA target modules: {args.lora_target_modules}")
        if args.use_4bit:
            logger.info("Using 4-bit quantization with LoRA")
        elif args.use_8bit:
            logger.info("Using 8-bit quantization with LoRA")
    else:
        logger.info("Not using LoRA (full fine-tuning)")
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directories
    os.makedirs(args.checkpoint_path, exist_ok=True)

    # Initialize AWS Bedrock client
    try:
        bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
        logger.info("Successfully initialized AWS Bedrock client")
    except Exception as e:
        logger.error(f"Failed to initialize AWS Bedrock client: {e}")
        logger.error("Please ensure you have configured AWS credentials properly.")
        return
    
    # Setup device
    device = torch.device(f"cuda:{args.device_index}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    if args.resume_from or args.load_from_checkpoint:
        # Determine which checkpoint path to use
        checkpoint_path = args.resume_from if args.resume_from else args.load_from_checkpoint

        # Load model from checkpoint with full state
        logger.info(f"Loading model from checkpoint with training state: {args.resume_from}")
        model, tokenizer = load_checkpoint_with_lora(
            checkpoint_path,
            args.model_family, 
            device_map=device,
            logger=logger,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
            trust_remote_code=args.trust_remote_code
        )
        
        # If resuming and using LoRA, need to reinitialize the LoRA adapter
        if args.use_lora and hasattr(model, "peft_config"):
            logger.info("Model already has PEFT config, continuing with existing LoRA configuration")
        elif args.use_lora:
            logger.info("Adding LoRA adapter to resumed model")
            model = add_lora_adapter(model, args, logger=logger)
    else:
        # Load model from name/path
        logger.info(f"Loading {args.model_family} model: {args.model_name}")
        model, tokenizer = load_model(
            args.model_name, 
            args.model_family, 
            trust_remote_code=args.trust_remote_code,
            device_map=device,
            logger=logger,
            use_4bit=args.use_4bit,
            use_8bit=args.use_8bit,
        )
        # If using a new model and LoRA, initialize the adapter
        if args.use_lora:
            logger.info("Adding LoRA adapter to new model")
            model = add_lora_adapter(model, args, logger=logger)

    # Create persistent optimizers once
    dpo_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=args.dpo_learning_rate
    )
    sft_optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.sft_learning_rate
    )

    logger.info(f"Created persistent optimizers: DPO (lr={args.dpo_learning_rate}), SFT (lr={args.sft_learning_rate})")

    # Initialize feedback provider with both basic and advanced models
    feedback_provider = FeedbackProvider(
        bedrock, 
        model=model,
        tokenizer=tokenizer,
        feedback_provider=args.feedback_provider,
        improvement_generator=args.improvement_generator,
        basic_model_key=args.bedrock_model_basic,
        advanced_model_key=args.bedrock_model_advanced,
        max_retries=args.bedrock_max_retries,
        dpo_optimizer=dpo_optimizer,  # Pass the DPO optimizer
        sft_optimizer=sft_optimizer,  # Pass the SFT optimizer
        logger=logger
    )

    logger.info("==========FEEDBACK PROMPT=============")
    if args.feedback_provider == "bedrock":
        logger.info(feedback_provider._create_feedback_prompt("","",""))
    else:
        logger.info(feedback_provider._create_feedback_prompt_self("","","",""))
    logger.info("==========IMPROVEMENT PROMPT=============")
    logger.info(feedback_provider._create_improvement_prompt("","","", ""))
    logger.info("==========SYSTEM PROMPT=============")
    from config import system_prompt
    logger.info(system_prompt)

    # Add this after parsing arguments (around line 776)
    if args.repeat_until_master:
        logger.info("Repeat until master mode enabled: Will continue training on bucket problems until mastered on first attempts")
    
    # Initialize W&B logging
    if args.wandb_project:
        logger.info(f"Initializing Weights & Biases with project: {args.wandb_project}")
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"DPO_SFT_GSM8K_{experiment_name}"
        )
    else:
        logger.info("Weights & Biases logging disabled")
        wandb.init(mode="disabled")
    
    # Load dataset from specified source using the new functions
    try:
        # Load dataset for training (is_eval=False will use train split)
        raw_dataset = load_dataset_for_training_or_eval(
            dataset_name=args.dataset_source,
            path=args.dataset_path if hasattr(args, 'dataset_path') else None,
            split=args.dataset_split,
            max_samples=args.max_samples,
            is_eval=False,
            logger=logger
        )
        
        # Format dataset for training
        formatted_data = format_dataset_for_eval(
            dataset_name=args.dataset_source,
            dataset=raw_dataset,
            logger=logger
        )
        
        # Convert to the format expected by the rest of the code
        dataset_data = []
        for i in range(len(formatted_data["questions"])):
            problem = {
                "question": formatted_data["questions"][i],
                "answer": formatted_data["answers"][i],
                "metadata": formatted_data["metadata"][i]
            }
            if args.dataset_source == 'folio':
                problem['question'] += ' Convert the above into FOL (first order logic) utilizing quantifiers, predicates, and logical connectors where needed.'
            dataset_data.append(problem)
        
        logger.info(f"Prepared {len(dataset_data)} problems for training")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # Initialize tracking variables
    total_problems_processed = 0
    metrics_tracker = MetricTracker(window_size=100)
    bucket_list = []
    
    # If resuming, load checkpoint metadata
    if args.resume_from:
        checkpoint_metadata = load_checkpoint_metadata(args.resume_from, logger)
        
        # Restore metrics tracker
        if "metrics_history" in checkpoint_metadata:
            metrics_tracker = restore_metrics_tracker(checkpoint_metadata["metrics_history"])
            logger.info("Restored metrics tracker from checkpoint")
        
        # Get remaining problems
        if "remaining_problems" in checkpoint_metadata:
            remaining_problems = checkpoint_metadata["remaining_problems"]
            
            # Calculate how many problems have been processed
            original_size = len(dataset_data)
            remaining_dataset_size = len(remaining_problems["dataset"])
            total_problems_processed = original_size - remaining_dataset_size
            
            # Filter dataset to only include remaining problems
            if remaining_problems["dataset"]:
                remaining_indices = set(remaining_problems["dataset"])
                dataset_data = [p for i, p in enumerate(dataset_data) if i in remaining_indices]
                
            # Restore bucket list
            if remaining_problems["bucket"]:
                bucket_indices = set(remaining_problems["bucket"])
                bucket_list = [p for i, p in enumerate(dataset_data) if i in bucket_indices]
                
            logger.info(f"Restored training state: {total_problems_processed} problems processed, "
                       f"{len(dataset_data)} problems remaining, {len(bucket_list)} problems in bucket")
        else:
            logger.warning("No remaining problems found in checkpoint, starting with full dataset")
    
    # S3 experiment path
    s3_experiment_path = f"{args.s3_bucket_name}/{experiment_name}"
    
    # Save arguments to S3
    save_args_to_s3(args, s3_experiment_path, logger=logger)

    if args.sft_only:
        logger.info("=== Running in SFT-only mode ===")
        logger.info(f"SFT epochs: {args.epochs}, batch size: {args.batch_size}")
        
        # Track metrics for SFT
        sft_metrics_tracker = MetricTracker(window_size=100)
        
        # Perform SFT training for specified epochs
        for epoch in range(args.epochs):
            logger.info(f"\n\n===== Starting SFT-only Epoch {epoch+1}/{args.epochs} =====")
            
            # Shuffle dataset if specified
            if args.shuffle_dataset:
                logger.info("Shuffling dataset for this epoch")
                random.shuffle(dataset_data)
            
            # Process dataset in batches
            num_batches = (len(dataset_data) + args.batch_size - 1) // args.batch_size
            total_loss = 0
            samples_processed = 0
            
            for batch_idx in range(num_batches):
                # Determine start and end indices for this batch
                start_idx = batch_idx * args.batch_size
                end_idx = min(start_idx + args.batch_size, len(dataset_data))
                batch_problems = dataset_data[start_idx:end_idx]
                batch_loss = 0
                
                # Process each problem in batch
                for problem_idx, problem in enumerate(batch_problems):
                    question = problem["question"]
                    expected_answer = problem["answer"]
                    
                    # Format expected answer with proper tags if not already formatted
                    if "<reasoning>" not in expected_answer:
                        # If we have a full solution in the dataset, use it
                        if "full_solution" in problem and problem["full_solution"]:
                            # Extract answer from full_solution
                            final_answer = re.search(r"####\s*([\d\.\-]+)", problem["full_solution"])
                            if final_answer:
                                answer_value = final_answer.group(1).strip()
                                # Use the full solution as reasoning
                                reasoning = problem["full_solution"].replace(f"#### {answer_value}", "").strip()
                                formatted_answer = f"<reasoning>{reasoning}</reasoning>\n<answer>{answer_value}</answer>"
                            else:
                                # No final answer marker, use as is
                                formatted_answer = f"<reasoning>{problem['full_solution']}</reasoning>\n<answer>{expected_answer}</answer>"
                        else:
                            # No full solution, create simple formatted answer
                            formatted_answer = f"<reasoning>Let me solve this step by step.</reasoning>\n<answer>{expected_answer}</answer>"
                    else:
                        # Already formatted correctly
                        formatted_answer = expected_answer
                    
                    # Use the perform_sft_training function instead of implementing inline
                    sft_result = perform_sft_training(
                        model=model,
                        tokenizer=tokenizer,
                        question=question,
                        answer=formatted_answer,
                        learning_rate=args.sft_learning_rate,
                        optimizer=sft_optimizer,  # Use the persistent optimizer
                        logger=logger
                    )
                    
                    # Track loss
                    if sft_result["success"] and sft_result["loss"] is not None:
                        batch_loss += sft_result["loss"]
                        logger.debug(f"Problem {problem_idx+1}/{len(batch_problems)} - Loss: {sft_result['loss']:.4f}")
                    else:
                        logger.warning(f"SFT training failed for problem {problem_idx+1}: {sft_result.get('error', 'Unknown error')}")
                
                # Update metrics
                avg_batch_loss = batch_loss / max(1, len(batch_problems))
                total_loss += batch_loss
                samples_processed += len(batch_problems)
                
                # Log progress
                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx+1}/{num_batches}, "
                            f"Avg Loss: {avg_batch_loss:.4f}, Overall Loss: {total_loss/max(1, samples_processed):.4f}")
                
                # Update metrics tracker
                sft_metrics_tracker.update({
                    "sft_loss": avg_batch_loss,
                    "epoch": epoch + 1,
                    "samples_processed": samples_processed
                })
                
                # Log to W&B
                if args.wandb_project:
                    wandb.log({
                        "sft_loss": avg_batch_loss,
                        "epoch": epoch + 1,
                        "samples_processed": samples_processed,
                        "total_samples": len(dataset_data)
                    })
                
                # # Save checkpoint periodically
                # if (batch_idx + 1) % (max(1, num_batches // 5)) == 0:
                #     checkpoint_dir = os.path.join(args.checkpoint_path, experiment_name, f"sft_epoch_{epoch+1}_batch_{batch_idx+1}")
                #     s3_checkpoint_dir = f"{s3_experiment_path}/sft_epoch_{epoch+1}_checkpoint_{(epoch+1)*(batch_idx+1)}"
                    
                #     # Save checkpoint
                #     save_checkpoint(
                #         model=model,
                #         tokenizer=tokenizer,
                #         metrics_tracker=sft_metrics_tracker,
                #         remaining_problems={'dataset': [], 'bucket': []},  # No concept of remaining problems in SFT-only mode
                #         checkpoint_dir=checkpoint_dir,
                #         s3_path=s3_checkpoint_dir,
                #         delete_local=args.delete_local_checkpoints,
                #         logger=logger
                #     )
                    
            # Create visualizations (after each epoch)
            vis_file_path = plot_metrics(sft_metrics_tracker, samples_processed,
                                            save_dir=args.vis_dir,
                                            experiment_name=experiment_name)
            logger.info(f"Created metrics visualization at: {vis_file_path}")
                    
            # Upload visualizations to S3
            vis_s3_path = f"{s3_experiment_path}/visualizations"
            save_to_s3(os.path.join(args.vis_dir, experiment_name), vis_s3_path, logger=logger)
            
            # End of epoch stats
            epoch_loss = total_loss / max(1, samples_processed)
            logger.info(f"Completed SFT-only Epoch {epoch+1}/{args.epochs} - "
                    f"Average Loss: {epoch_loss:.4f}")
            
            # Save epoch checkpoint
            checkpoint_dir = os.path.join(args.checkpoint_path, experiment_name, f"sft_epoch_{epoch+1}_final")
            s3_checkpoint_dir = f"{s3_experiment_path}/sft_epoch_checkpoint_{epoch+1}"
            
            # Save checkpoint
            save_checkpoint(
                model=model,
                tokenizer=tokenizer,
                metrics_tracker=sft_metrics_tracker,
                remaining_problems={'dataset': [], 'bucket': []},  # No concept of remaining problems in SFT-only mode
                checkpoint_dir=checkpoint_dir,
                s3_path=s3_checkpoint_dir,
                delete_local=args.delete_local_checkpoints,
                logger=logger
            )
        return
        ## end of sft-only
    
    # Training loop for each epoch
    for epoch in range(args.epochs):
        logger.info(f"\n\n===== Starting Epoch {epoch+1}/{args.epochs} =====")
        
        # Shuffle dataset if specified
        if args.shuffle_dataset:
            logger.info("Shuffling dataset for this epoch")
            random.shuffle(dataset_data)
        
        # Process dataset in batches
        num_batches = (len(dataset_data) + args.batch_size - 1) // args.batch_size
        
        # Track completed and remaining problem indices for this epoch
        completed_indices = set()
        remaining_indices = set(range(len(dataset_data)))
        bucket_indices = set()  # Track indices of problems in bucket list

        for batch_idx in range(num_batches):
            logger.info(f"\n===== Processing Batch {batch_idx+1}/{num_batches} (Epoch {epoch+1}/{args.epochs}) =====")
            
            # Determine start and end indices for this batch
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, len(dataset_data))
            batch_problems = dataset_data[start_idx:end_idx]
            
            # Process each problem in the batch
            for problem_idx, problem in enumerate(batch_problems):
                global_idx = start_idx + problem_idx
                epoch_problem_num = global_idx + 1
                overall_problem_num = total_problems_processed + 1
                logger.info(f"\n----- Epoch {epoch+1}, Problem {epoch_problem_num}/{len(dataset_data)} (Overall: {overall_problem_num}) -----")
                
                # Run DPO step for this problem (using basic model since this is from the main dataset)
                result = run_dpo_step(model, tokenizer, problem, feedback_provider, args, logger=logger, is_bucket_problem=False)
                
                # Mark this problem as completed and remove from remaining list
                completed_indices.add(global_idx)
                if global_idx in remaining_indices:
                    remaining_indices.remove(global_idx)
                
                # Check if problem was solved (both correct answer and format)
                if not (result["is_correct"] and result["format_correct"]) or (args.repeat_until_master and not(result['original_format_correct'] and result['original_answer_correct'])):
                    logger.info(f"Problem {epoch_problem_num} not fully solved. Adding to bucket list.")
                    bucket_list.append(problem)
                    bucket_indices.add(global_idx)  # Track the index in the bucket
                
                # Update metrics
                metrics = {
                    "answer_accuracy": 1.0 if result["is_correct"] else 0.0,
                    "format_accuracy": 1.0 if result["format_correct"] else 0.0,
                    "combined_score": result["combined_score"],
                    "dpo_performed": 1.0 if result["dpo_performed"] else 0.0,
                    "verification_attempts": result["verification_attempts"],
                    "bucket_list_size": len(bucket_list),
                    "sft_performed": 1.0 if result["sft_performed"] else 0.0,
                    "epoch": epoch + 1,
                    "first_time_accuracy": 1.0 if result["first_time_correct"] else 0.0,  # Add first time accuracy
                    "bucket_size_progress": len(bucket_list),  # Bucket size
                }
                
                # Include losses if available
                if result["dpo_performed"] and "dpo_loss" in result and result["dpo_loss"] is not None:
                    metrics["dpo_loss"] = result["dpo_loss"]
                    
                if result["sft_performed"] and "sft_loss" in result and result["sft_loss"] is not None:
                    metrics["sft_loss"] = result["sft_loss"]
                    
                metrics_tracker.update(metrics)
                
                # Log metrics
                logger.info(f"Metrics for problem {epoch_problem_num} (epoch {epoch+1}):")
                for key, value in metrics.items():
                    logger.info(f"  {key}: {value}")
                
                # Log to W&B
                if args.wandb_project:
                    wandb.log(metrics)
                
                # Increment the total problems processed counter
                total_problems_processed += 1
                
                # Save examples periodically (include both dataset and bucket problems)
                if total_problems_processed % args.vis_interval == 0:
                    # Create visualizations
                    vis_file_path = plot_metrics(metrics_tracker, total_problems_processed, 
                                               save_dir=args.vis_dir, 
                                               experiment_name=experiment_name)
                    logger.info(f"Created metrics visualization at: {vis_file_path}")
                    
                    # Upload visualizations to S3
                    vis_s3_path = f"{s3_experiment_path}/visualizations"
                    save_to_s3(os.path.join(args.vis_dir, experiment_name), vis_s3_path, logger=logger)
                
                # Save checkpoint periodically (include both dataset and bucket problems)
                if total_problems_processed % args.checkpoint_interval == 0:
                    # Prepare remaining problems data
                    remaining_data = {
                        "dataset": list(remaining_indices),
                        "bucket": list(bucket_indices)
                    }
                    
                    # Local checkpoint directory
                    checkpoint_dir = os.path.join(args.checkpoint_path, experiment_name, f"checkpoint_{total_problems_processed}")
                    
                    # S3 checkpoint dir
                    s3_checkpoint_dir = f"{s3_experiment_path}/checkpoint_{total_problems_processed}"
                    
                    # Save checkpoint
                    save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        metrics_tracker=metrics_tracker,
                        remaining_problems=remaining_data,
                        checkpoint_dir=checkpoint_dir,
                        s3_path=s3_checkpoint_dir,
                        delete_local=args.delete_local_checkpoints,
                        logger=logger
                    )
            
            # Process bucket list after each batch if enabled
            if bucket_list and args.process_bucket:
                logger.info(f"\n===== Processing Bucket List ({len(bucket_list)} problems) =====")
                logger.info(f"Using advanced Bedrock model: {args.bedrock_model_advanced if args.bedrock_model_advanced else args.bedrock_model_basic}")
                
                # If using repeat_until_master, keep processing the bucket list until all problems are mastered
                master_attempts = 0
                bucket_progress = True  # Track if we're making progress in the bucket
                
                while bucket_list and (not args.repeat_until_master or (master_attempts < args.batch_size)):
                    if args.repeat_until_master and master_attempts > 0:
                        logger.info(f"\n===== Master Attempt {master_attempts+1}/{args.batch_size}: Processing Bucket List ({len(bucket_list)} problems) =====")
                    
                    # Process each problem in the bucket list
                    new_bucket_list = []
                    new_bucket_indices = set()
                    first_attempt_correct_count = 0
                    
                    for i, problem in enumerate(bucket_list):
                        # Find the index of this problem in the original dataset
                        problem_idx = -1
                        for idx, orig_problem in enumerate(dataset_data):
                            if orig_problem["question"] == problem["question"]:
                                problem_idx = idx
                                break
                        
                        logger.info(f"\n----- Bucket List Problem {i+1}/{len(bucket_list)} (Epoch {epoch+1}, Index: {problem_idx}, Overall: {total_problems_processed+1}) -----")
                        
                        # Run DPO step for this problem (using advanced model since this is from the bucket list)
                        result = run_dpo_step(model, tokenizer, problem, feedback_provider, args, logger=logger, is_bucket_problem=True)
                        
                        # Check if problem is solved on the first attempt when using repeat_until_master
                        if args.repeat_until_master and result["first_time_correct"]:
                            logger.info(f"Problem solved correctly on first attempt! Mastered.")
                            first_attempt_correct_count += 1
                        elif not (result["is_correct"] and result["format_correct"]) or (args.repeat_until_master and not result["first_time_correct"]):
                            # Problem not fully solved or not mastered (if using repeat_until_master)
                            logger.info(f"Problem {'not fully solved' if not (result['is_correct'] and result['format_correct']) else 'not mastered on first attempt'}. Keeping in bucket list.")
                            new_bucket_list.append(problem)
                            if problem_idx >= 0:
                                new_bucket_indices.add(problem_idx)
                        else:
                            logger.info(f"Problem solved! Removing from bucket list.")
                        
                        # Update metrics
                        metrics = {
                            "answer_accuracy": 1.0 if result["is_correct"] else 0.0,
                            "format_accuracy": 1.0 if result["format_correct"] else 0.0,
                            "combined_score": result["combined_score"],
                            "dpo_performed": 1.0 if result["dpo_performed"] else 0.0,
                            "verification_attempts": result["verification_attempts"],
                            "bucket_list_size": len(new_bucket_list),
                            "sft_performed": 1.0 if result["sft_performed"] else 0.0,
                            "epoch": epoch + 1,
                            "first_time_accuracy": 1.0 if result["first_time_correct"] else 0.0,
                            "bucket_size_progress": len(bucket_list) / (len(dataset_data) + 0.001),  # Normalized bucket size
                        }
                        
                        # Include losses if available
                        if result["dpo_performed"] and "dpo_loss" in result and result["dpo_loss"] is not None:
                            metrics["dpo_loss"] = result["dpo_loss"]
                            
                        if result["sft_performed"] and "sft_loss" in result and result["sft_loss"] is not None:
                            metrics["sft_loss"] = result["sft_loss"]
                        
                        metrics_tracker.update(metrics)
                        
                        # Log to W&B
                        if args.wandb_project:
                            wandb.log(metrics)
                        
                        # Increment total problems processed counter for bucket problems too
                        total_problems_processed += 1
                        
                        # Save examples periodically (include both dataset and bucket problems)
                        if total_problems_processed % args.vis_interval == 0:
                            # Create visualizations
                            vis_file_path = plot_metrics(metrics_tracker, total_problems_processed, 
                                                    save_dir=args.vis_dir, 
                                                    experiment_name=experiment_name)
                            logger.info(f"Created metrics visualization at: {vis_file_path}")
                            
                            # Upload visualizations to S3
                            vis_s3_path = f"{s3_experiment_path}/visualizations"
                            save_to_s3(os.path.join(args.vis_dir, experiment_name), vis_s3_path, logger=logger)
                        
                        # Save checkpoint periodically (include both dataset and bucket problems)
                        if total_problems_processed % args.checkpoint_interval == 0:
                            # Prepare remaining problems data
                            remaining_data = {
                                "dataset": list(remaining_indices),
                                "bucket": list(new_bucket_indices)
                            }
                            
                            # Local checkpoint directory
                            checkpoint_dir = os.path.join(args.checkpoint_path, experiment_name, f"checkpoint_{total_problems_processed}")
                            
                            # S3 checkpoint dir
                            s3_checkpoint_dir = f"{s3_experiment_path}/checkpoint_{total_problems_processed}"
                            
                            # Save checkpoint
                            save_checkpoint(
                                model=model,
                                tokenizer=tokenizer,
                                metrics_tracker=metrics_tracker,
                                remaining_problems=remaining_data,
                                checkpoint_dir=checkpoint_dir,
                                s3_path=s3_checkpoint_dir,
                                delete_local=args.delete_local_checkpoints,
                                logger=logger
                            )
                    
                    # Check progress for repeat_until_master mode
                    if args.repeat_until_master:
                        mastered_count = len(bucket_list) - len(new_bucket_list)
                        logger.info(f"Master attempt {master_attempts+1}: {mastered_count}/{len(bucket_list)} problems mastered " +
                                    f"({first_attempt_correct_count} first-attempt correct).")
                        
                        # Check if we're making progress (bucket list size decreasing)
                        bucket_progress = len(new_bucket_list) < len(bucket_list)
                        master_attempts += 1
                        
                        if len(new_bucket_list) == 0:
                            logger.info("All bucket problems mastered! Moving to next batch.")
                        elif not bucket_progress:
                            logger.info(f"No progress made in this attempt. Continuing with SFT on last winning answers for attempt {master_attempts+1}/{args.batch_size}.")

                            # Force SFT on all remaining bucket problems using their last winning answers
                            for i, problem in enumerate(new_bucket_list):
                                logger.info(f"Forcing SFT on bucket problem {i+1}/{len(new_bucket_list)} despite no progress")
                                
                                # Extract question from problem
                                question = problem["question"]
                                
                                # Get the last winning answer for this problem (you may need to store this during the DPO process)
                                # This assumes that the last result from run_dpo_step contains the winning answer
                                # Perform SFT using the winning answer
                                sft_result = perform_sft_training(
                                    model=model,
                                    tokenizer=tokenizer,
                                    question=question,
                                    answer=result['winning_answer'],  # Use the winning answer from the last DPO run
                                    learning_rate=args.sft_learning_rate,
                                    optimizer=sft_optimizer,  # Use the persistent optimizer
                                    logger=logger
                                )
                                
                                logger.info(f"Forced SFT result: {'Success' if sft_result['success'] else 'Failed'}, Loss: {sft_result.get('loss', 'N/A')}")
                        elif master_attempts >= args.batch_size:
                            logger.info(f"Reached maximum master attempts ({args.batch_size}). Moving remaining problems to next batch processing.")
                    
                    # Update bucket list and indices
                    bucket_list = new_bucket_list
                    bucket_indices = new_bucket_indices
                    logger.info(f"Updated bucket list size: {len(bucket_list)}")
                    
                    # Break if we're not using repeat_until_master
                    if not args.repeat_until_master:
                        break
        
        # Save epoch checkpoint
        # Prepare remaining problems data for next epoch
        remaining_data = {
            "dataset": list(remaining_indices),
            "bucket": list(bucket_indices)
        }
        
        # Local epoch directory
        local_epoch_dir = os.path.join(args.checkpoint_path, experiment_name, f"epoch_{epoch+1}")
        
        # S3 epoch directory
        s3_epoch_dir = f"{s3_experiment_path}/epoch_{epoch+1}"
        
        # Save epoch checkpoint
        save_checkpoint(
            model=model,
            tokenizer=tokenizer,
            metrics_tracker=metrics_tracker,
            remaining_problems=remaining_data,
            checkpoint_dir=local_epoch_dir,
            s3_path=s3_epoch_dir,
            delete_local=args.delete_local_checkpoints,
            logger=logger
        )
        
        logger.info(f"\n===== Completed Epoch {epoch+1}/{args.epochs} =====")
        logger.info(f"Problems processed in this epoch: {len(dataset_data)}")
        logger.info(f"Total problems processed so far: {total_problems_processed}")
        logger.info(f"Current bucket list size: {len(bucket_list)}")
    
    # Save final model
    # Prepare remaining problems data (should be empty at this point)
    remaining_data = {
        "dataset": list(remaining_indices),
        "bucket": list(bucket_indices)
    }
    
    # Local final directory
    local_final_dir = os.path.join(args.checkpoint_path, experiment_name, "final")
    
    # S3 final directory
    s3_final_dir = f"{s3_experiment_path}/final"
    
    # Save final model
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        metrics_tracker=metrics_tracker,
        remaining_problems=remaining_data,
        checkpoint_dir=local_final_dir,
        s3_path=s3_final_dir,
        delete_local=args.delete_local_checkpoints,
        logger=logger
    )
    
    # Save bucket list for future analysis
    if bucket_list:
        bucket_list_path = os.path.join(args.checkpoint_path, experiment_name, "bucket_list.json")
        with open(bucket_list_path, "w", encoding="utf-8") as f:
            json.dump([{"question": p["question"], "answer": p["answer"]} for p in bucket_list], f, indent=2)
        logger.info(f"Saved {len(bucket_list)} unsolved problems to {bucket_list_path}")
        
        # Upload bucket list to S3
        s3_client = boto3.client('s3')
        bucket_name = args.s3_bucket_name
        s3_key = f"{experiment_name}/bucket_list.json"
        s3_client.upload_file(bucket_list_path, bucket_name, s3_key)
        logger.info(f"Uploaded bucket list to s3://{bucket_name}/{s3_key}")
    
    # Save logs to S3
    save_logs_to_s3(args.log_dir, s3_experiment_path, logger=logger, experiment_name=experiment_name)
    
    # Final metrics summary
    logger.info("\n===== Training Summary =====")
    logger.info(f"Total problems processed: {total_problems_processed}")
    logger.info(f"Epochs completed: {args.epochs}")
    logger.info(f"Final bucket list size: {len(bucket_list)}")
    
    avg_metrics = metrics_tracker.get_all_metrics()
    logger.info("Average metrics over the last window:")
    for key, value in avg_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("Training complete!")
    
if __name__ == "__main__":
    main()
