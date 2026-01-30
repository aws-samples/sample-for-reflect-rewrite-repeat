import os
import re
import json
import glob
import math
import time
import shutil
import gc
import random
import argparse
import subprocess
import aiohttp
import requests
import asyncio
import logging
from datetime import datetime
from tqdm import tqdm
import boto3
import pandas as pd
import matplotlib.pyplot as plt
import torch

from config import system_prompt, MODEL_IDS
from evaluation_utils import evaluate_model_response
from dataset import load_dataset_for_training_or_eval, format_dataset_for_eval
from model_utils import get_local_checkpoints, detect_model_family


def setup_logger(log_dir="./eval_logs"):
    """
    Configure logging to file and console
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger('model_evaluation')
    logger.setLevel(logging.INFO)
    
    if logger.handlers:  # Clear any existing handlers
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


def get_bedrock_response(system_prompt, user_question, max_tokens, temperature, top_p, max_retries, bedrock_client, bedrock_model_name, logger):
    """
    Get response from Bedrock models
    
    Args:
        prompt: The prompt to send to Bedrock
        
    Returns:
        Response from Bedrock
    """
    # Try up to max_retries times if there's an error
    for retry_count in range(max_retries):
        try:
            # Add a small delay to prevent rate limiting
            time.sleep(random.uniform(0.1, 0.5))

            bedrock_model_id = MODEL_IDS[bedrock_model_name]
            
            logger.debug(f"Getting response using: {bedrock_model_name} ({bedrock_model_id})")
            
            response = bedrock_client.converse(
                modelId=bedrock_model_id,
                system = [
                    {"text": system_prompt}
                ],
                messages = [
                    {"role": "user", "content": [{"text": user_question}]}
                ],
                inferenceConfig={
                    "maxTokens": max_tokens,
                    "temperature": temperature,
                    "topP": top_p,
                }
            )
            
            # Extract response text
            result = response['output']['message']['content'][0]['text']
            return result
                    
        except Exception as e:
            if retry_count < max_retries - 1:
                logger.warning(f"Error from Bedrock (attempt {retry_count+1}/{max_retries}): {e}. Retrying...")
                # time.sleep(1.0)  # Exponential backoff
            else:
                logger.error(f"All {max_retries} attempts failed. Error: {e}")
                return "Unable to provide response due to technical issues."
    
    # This should never be reached
    return "Unable to provide response due to technical issues."


def start_model_server(checkpoint_path, model_family, args):
    """
    Start model server
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_family: Model architecture family
        args: Command line arguments
        
    Returns:
        List of dictionaries with process and port information
    """
    processes = []
    
    # Start each model in its own process
    for i in range(args.batch_size):
        hostname = args.api_hostname
        port = args.port_start + i

        # Create command for subprocess
        cmd = [
            "python", "serve.py", 
            "--seed", f"{args.seed}",
            "--bits", f"{args.bits}",
            "--checkpoint_path", f"{checkpoint_path}",
            "--model_family", f"{model_family}",
            "--worker_id", f"{i}",
            "--api_hostname", f"{hostname}",
            "--api_port", f"{port}",
        ]
        
        # Start process
        process = subprocess.Popen(cmd)
        
        processes.append({
            "process": process,
            "port": port,
            "model_id": i,
            "api_url": f"http://{hostname}:{port}"
        })

    return processes

def check_model_servers_ready(processes, timeout=300):
    """
    Wait for all model servers to be ready
    
    Args:
        processes: List of process dictionaries
        timeout: Maximum time to wait in seconds
        
    Returns:
        Boolean indicating if all servers are ready
    """
    start_time = time.time()
    all_ready = False
    
    while time.time() - start_time < timeout:
        ready_count = 0
        
        for proc_info in processes:
            try:
                # Check if server is responding
                response = requests.get(f"{proc_info['api_url']}/health", timeout=10)
                if response.status_code == 200:
                    ready_count += 1
            except:
                pass
        
        if ready_count == len(processes):
            all_ready = True
            break
            
        # # Wait before checking again
        # time.sleep(5)
    
    return all_ready

def cleanup_processes(processes):
    """
    Clean up model server processes
    
    Args:
        processes: List of process dictionaries
    """
    for proc_info in processes:
        try:
            # Try to terminate gracefully
            proc_info["process"].terminate()
            proc_info["process"].wait(timeout=10)
        except:
            # Force kill if necessary
            try:
                proc_info["process"].kill()
            except:
                pass

def log_progress(
    logger,
    question_id,
    response, 
    question, 
    target, 
    result, 
    processed, 
    num_questions, 
    total_correct, 
    last_number_correct_count, 
    format_correct_count, 
    total_valid
):
    logger.info("")
    logger.info("---------------------------------")
    logger.info(f"Summary of question_id {question_id}")
    logger.info("---------------------------------")
    logger.info(f"question: \n{question}")
    logger.info(f"model_response: \n{response}")
    logger.info("---------------------------------")
    logger.info(f"target answer: {target}")
    logger.info(f"extracted_answer: {result['extracted_answer']}")
    logger.info(f"last_number_answer: {result['last_number_answer']}")
    logger.info(f"format_correct: {result['format_correct']}")
    logger.info(f"answer_correct: {result['answer_correct']}")
    logger.info(f"last_number_correct: {result['last_number_correct']}")
    logger.info(f"is_valid: {result['is_valid']}")
    logger.info("---------------------------------\n")
    
    if processed % 20 == 0 or processed == num_questions:
        accuracy = total_correct / processed if processed > 0 else 0
        last_number_accuracy = last_number_correct_count / processed if processed > 0 else 0
        format_accuracy = format_correct_count / processed if processed > 0 else 0
        valid_rate = total_valid / processed if processed > 0 else 0
        
        logger.info(f"Progress: {processed}/{num_questions} questions")
        logger.info(f"  Current Accuracy: {accuracy:.4f} ({total_correct}/{processed})")
        logger.info(f"  Last Number Accuracy: {last_number_accuracy:.4f} ({last_number_correct_count}/{processed})")
        logger.info(f"  Format Correctness: {format_accuracy:.4f} ({format_correct_count}/{processed})")
        logger.info(f"  Valid Answer Rate: {valid_rate:.4f} ({total_valid}/{processed})")

async def prepare_task_queue(formatted_data, logger, args):
    # Apply max_samples limit if specified
    questions = formatted_data["questions"]
    float_answers = formatted_data["answers"]
    metadata = formatted_data["metadata"]
    
    if args.max_samples > 0 and args.max_samples < len(questions):
        logger.info(f"Limiting evaluation to {args.max_samples} samples (out of {len(questions)})")
        questions = questions[:args.max_samples]
        float_answers = float_answers[:args.max_samples]
        metadata = metadata[:args.max_samples]
        
    # Save the sampled dataset for reproducibility
    if args.max_samples > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sampled_dataset_path = os.path.join(
            args.output_dir, 
            f"sampled_{args.dataset_type}_{args.max_samples}_seed{args.seed}_{timestamp}.jsonl"
        )
        logger.info(f"Saving sampled dataset with {len(questions)} examples to {sampled_dataset_path}")
        
        with open(sampled_dataset_path, 'w', encoding='utf-8') as f:
            for q, a, m in zip(questions, float_answers, metadata):
                item = {"question": q, "answer": a, **m}
                f.write(json.dumps(item) + '\n')

    # Create a task queue for better load balancing
    task_queue = asyncio.Queue()
        
    # Add all questions to the queue
    for i, (question, target, meta) in enumerate(zip(questions, float_answers, metadata)):
        await task_queue.put((i, question, target, meta))

    return task_queue, len(questions)

async def evaluate_with_bedrock_async(formatted_data, logger, bedrock_client, args):
    """
    Asynchronously evaluate model on a dataset using API
    
    Args:
        formatted_data: Dictionary with questions, answers, and metadata
        logger: Logger instance
        bedrock_client: Bedrock Client
        args: Command line arguments
    
    Returns:
        Evaluation results dictionary
    """

    task_queue, num_tasks = await prepare_task_queue(formatted_data, logger, args)
    
    # Results tracking
    results = [None] * num_tasks  # Placeholder for results
    total_correct = 0
    total_valid = 0
    format_correct_count = 0
    last_number_correct_count = 0
    
    # Create a progress bar
    pbar = tqdm(total=num_tasks, desc="Processing questions")
    
    # Process a single question
    async def process_question(worker_id):
        """
        Worker task that processes questions from the queue
        """
        # assert url or client, "api url or boto3 client has to be provided"

        while True:
            try:
                # Get next question from the queue with timeout
                try:
                    i, question, target, question_meta = await asyncio.wait_for(
                        task_queue.get(), timeout=0.5
                    )
                except asyncio.TimeoutError:
                    # Check if queue is empty - if so, we're done
                    if task_queue.empty():
                        return
                    continue
                
                # Send request to bedrock
                max_retries = 3
                response = get_bedrock_response(
                    system_prompt,
                    question,
                    args.max_length,
                    args.temperature,
                    args.top_p,
                    max_retries,
                    bedrock_client,
                    args.bedrock_model_name,
                    logger
                )
                result = evaluate_model_response(response, target)
                result.update({
                    "question": question,
                    "target_answer": target,
                    "question_id": i,
                    "question_metadata": question_meta
                })

                # Store the result
                results[i] = result
                
                # Update counters
                nonlocal total_correct, total_valid, format_correct_count, last_number_correct_count
                if result["format_correct"]:
                    format_correct_count += 1
                    
                if result["is_valid"]:
                    total_valid += 1
                    
                if result["answer_correct"]:
                    total_correct += 1
                    
                if result.get("last_number_correct", False):
                    last_number_correct_count += 1
                
                # Mark task as done in queue
                task_queue.task_done()
                
                # Update progress
                pbar.update(1)
                processed = pbar.n

                log_progress(
                    logger, i, response, question, target, result, 
                    processed, num_tasks, total_correct, 
                    last_number_correct_count, format_correct_count, total_valid
                )
                
            except Exception as e:
                logger.error(f"Unexpected error in worker {worker_id}: {e}")
                # Just continue processing the next task

    # Create a list of worker tasks based on the number of model servers
    workers = []    
    for worker_i in range(args.max_concurrent):
        worker = process_question(worker_i)
        workers.append(asyncio.create_task(worker))

    # Wait for all worker tasks to complete
    await asyncio.gather(*workers)
    
    # Final check to ensure all tasks were processed
    if not task_queue.empty():
        logger.warning(f"Task queue not empty! {task_queue.qsize()} tasks remaining.")
        # Process any remaining tasks
        remaining_tasks = []
        while not task_queue.empty():
            try:
                remaining_tasks.append(await task_queue.get())
                task_queue.task_done()
            except:
                break
        
        logger.warning(f"Unprocessed tasks: {len(remaining_tasks)}")

    # Close progress bar
    pbar.close()
    
    # Calculate final metrics
    total = num_tasks
    accuracy = total_correct / total if total > 0 else 0
    last_number_accuracy = last_number_correct_count / total if total > 0 else 0
    format_accuracy = format_correct_count / total if total > 0 else 0
    valid_percentage = total_valid / total if total > 0 else 0
    
    # Compile final results
    final_results = {
        "accuracy": accuracy,
        "last_number_accuracy": last_number_accuracy,
        "format_accuracy": format_accuracy,
        "valid_percentage": valid_percentage,
        "correct": total_correct,
        "last_number_correct": last_number_correct_count,
        "format_correct": format_correct_count,
        "total": total,
        "valid_predictions": total_valid,
        "predictions": results,
    }
    
    return final_results
        
async def evaluate_model_via_api_async(formatted_data, logger, args, processes=None):
    """
    Asynchronously evaluate model on a dataset using API
    
    Args:
        formatted_data: Dictionary with questions, answers, and metadata
        logger: Logger instance
        args: Command line arguments
        processes: Model processes (for separate process mode)

    Returns:
        Evaluation results dictionary
    """

    task_queue, num_tasks = await prepare_task_queue(formatted_data, logger, args)

    # Results tracking
    results = [None] * num_tasks  # Placeholder for results
    total_correct = 0
    total_valid = 0
    format_correct_count = 0
    last_number_correct_count = 0
    
    # Create counters to monitor request distribution
    num_servers = len(processes) if processes else args.batch_size
    server_request_counts = [0] * num_servers
    server_completion_counts = [0] * num_servers
    
    # Create a progress bar
    pbar = tqdm(total=num_tasks, desc="Processing questions")
    
    # Process a single question
    # async def process_question(session, worker_id, url):
    #     """
    #     Worker task that processes questions from the queue
    #     """
    #     while True:
    #         try:
    #             # Get next question from the queue with timeout
    #             try:
    #                 i, question, target, question_meta = await asyncio.wait_for(
    #                     task_queue.get(), timeout=0.5
    #                 )
    #             except asyncio.TimeoutError:
    #                 # Check if queue is empty - if so, we're done
    #                 if task_queue.empty():
    #                     return
    #                 continue
                
    #             # Update request counter
    #             server_request_counts[worker_id] += 1
                
    #             # Prepare request
    #             request_data = {
    #                 "system_prompt": system_prompt,
    #                 "question": question,
    #                 "target_answer": target,
    #                 "model_id": 0 if processes else worker_id,
    #                 "temperature": args.temperature,
    #                 "top_p": args.top_p,
    #                 "max_new_tokens": args.max_length,
    #                 "question_id": i,
    #                 "question_metadata": question_meta
    #             }

    #             # Send request to API server with retries
    #             retry_count = 0
    #             max_retries = 300
    #             result = None
                
    #             while retry_count < max_retries:
    #                 try:
    #                     async with session.post(url, json=request_data, timeout=300) as response:
    #                         if response.status == 200:
    #                             result = await response.json()
    #                             break
    #                         else:
    #                             logger.warning(f"Request failed with status {response.status}: {await response.text()}")
    #                             retry_count += 1
    #                             await asyncio.sleep(0.5)  # Wait before retrying
    #                 except Exception as e:
    #                     logger.error(f"Error processing question {i} on worker {worker_id}: {e}")
    #                     retry_count += 1
    #                     await asyncio.sleep(0.5)  # Wait before retrying
                
    #             # If all retries fail, create error placeholder
    #             if result is None:
    #                 result = {
    #                     "question": question,
    #                     "model_response": "ERROR: Failed after retries",
    #                     "extracted_answer": None,
    #                     "last_number_answer": None,
    #                     "target_answer": target,
    #                     "format_correct": False,
    #                     "answer_correct": False,
    #                     "last_number_correct": False,
    #                     "is_valid": False,
    #                     "error": "Request failed after multiple retries",
    #                     "question_id": i,
    #                     "question_metadata": question_meta
    #                 }
                
    #             # Store the result
    #             results[i] = result
                
    #             # Update counters
    #             nonlocal total_correct, total_valid, format_correct_count, last_number_correct_count
    #             if result["format_correct"]:
    #                 format_correct_count += 1
                    
    #             if result["is_valid"]:
    #                 total_valid += 1
                    
    #             if result["answer_correct"]:
    #                 total_correct += 1
                    
    #             if result.get("last_number_correct", False):
    #                 last_number_correct_count += 1
                
    #             # Mark task as done in queue
    #             task_queue.task_done()
                
    #             # Update progress
    #             pbar.update(1)
    #             processed = pbar.n

    #             log_progress(
    #                 logger, i, result["model_response"], question, target, result, 
    #                 processed, num_tasks, total_correct, 
    #                 last_number_correct_count, format_correct_count, total_valid
    #             )

    #             # Update completion counter
    #             server_completion_counts[worker_id] += 1

    #             if processed % 10 == 0:
    #                 # Report server load distribution
    #                 logger.info(f"Server request distribution: {server_request_counts}")
    #                 logger.info(f"Server completion distribution: {server_completion_counts}")
                
    #         except Exception as e:
    #             logger.error(f"Unexpected error in worker {worker_id}: {e}")
    #             # Just continue processing the next task
    
    # Create a session for HTTP requests
    async with aiohttp.ClientSession() as session:
        # Create a list of worker tasks based on the number of model servers
        workers = []

        # Separate process mode
        for i, proc_info in enumerate(processes):
            worker = process_question(session, i, f"{proc_info['api_url']}/generate")
            workers.append(asyncio.create_task(worker))
        
        # Wait for all worker tasks to complete
        await asyncio.gather(*workers)
        
        # Final check to ensure all tasks were processed
        if not task_queue.empty():
            logger.warning(f"Task queue not empty! {task_queue.qsize()} tasks remaining.")
            # Process any remaining tasks
            remaining_tasks = []
            while not task_queue.empty():
                try:
                    remaining_tasks.append(await task_queue.get())
                    task_queue.task_done()
                except:
                    break
            
            logger.warning(f"Unprocessed tasks: {len(remaining_tasks)}")
    
    # Close progress bar
    pbar.close()
    
    # Log final server distribution
    logger.info("Final server request distribution:")
    for i, count in enumerate(server_request_counts):
        logger.info(f"  Server {i}: {count} requests, {server_completion_counts[i]} completed")
    
    # Calculate final metrics
    total = num_tasks
    accuracy = total_correct / total if total > 0 else 0
    last_number_accuracy = last_number_correct_count / total if total > 0 else 0
    format_accuracy = format_correct_count / total if total > 0 else 0
    valid_percentage = total_valid / total if total > 0 else 0
    
    # Compile final results
    final_results = {
        "accuracy": accuracy,
        "last_number_accuracy": last_number_accuracy,
        "format_accuracy": format_accuracy,
        "valid_percentage": valid_percentage,
        "correct": total_correct,
        "last_number_correct": last_number_correct_count,
        "format_correct": format_correct_count,
        "total": total,
        "valid_predictions": total_valid,
        "predictions": results,
        "server_request_counts": server_request_counts,
        "server_completion_counts": server_completion_counts
    }
    
    return final_results

def interactive_s3_checkpoint_selection(bucket, logger):
    """
    Interactively select checkpoints from an S3 bucket up to two levels deep
    
    Args:
        bucket: S3 bucket name
        logger: Logger instance
        
    Returns:
        List of selected checkpoint folders (full S3 paths)
    """
    s3_client = boto3.client('s3')
    logger.info(f"Interactively selecting checkpoints from bucket: {bucket}")
    
    # List top-level folders in the bucket
    try:
        # Use delimiter to simulate folder-like behavior
        result = s3_client.list_objects_v2(Bucket=bucket, Delimiter='/')
        top_level_folders = []
        
        # Get common prefixes (which represent folders)
        if 'CommonPrefixes' in result:
            top_level_folders = [p['Prefix'] for p in result['CommonPrefixes']]
        
        if not top_level_folders:
            logger.warning("No folders found in the root of the bucket.")
            return []
        
        # Display top-level folders with numbers for selection
        logger.info("Available top-level folders:")
        for i, folder in enumerate(top_level_folders):
            logger.info(f"[{i+1}] {folder}")
        
        # Get user selection for top-level folders
        selections = input("Enter folder number(s) to explore (comma-separated, or 'all'): ")
        
        selected_top_folders = []
        if selections.strip().lower() == 'all':
            selected_top_folders = top_level_folders
        else:
            try:
                indices = [int(x.strip()) - 1 for x in selections.split(',') if x.strip()]
                selected_top_folders = [top_level_folders[i] for i in indices if 0 <= i < len(top_level_folders)]
            except (ValueError, IndexError):
                logger.error("Invalid selection. Please try again.")
                return interactive_s3_checkpoint_selection(bucket, logger)
        
        if not selected_top_folders:
            logger.warning("No folders were selected. Exiting.")
            return []
        
        # Process each selected top-level folder to find checkpoints
        selected_checkpoints = []
        
        for folder in selected_top_folders:
            # Check if this folder directly contains checkpoints
            direct_checkpoints = []
            result = s3_client.list_objects_v2(Bucket=bucket, Prefix=folder, Delimiter='/')
            
            if 'CommonPrefixes' in result:
                prefixes = [p['Prefix'] for p in result['CommonPrefixes']]
                # Look for checkpoint folders
                direct_checkpoints = [p for p in prefixes if 'checkpoint_' in p or 'final' in p]
            
            if direct_checkpoints:
                logger.info(f"Found checkpoints in '{folder}'")
                
                # Sort checkpoints by step number
                step_pattern = re.compile(r'checkpoint_(\d+)')
                
                def get_step_number(path):
                    match = step_pattern.search(path)
                    if match:
                        return int(match.group(1))
                    elif 'final' in path:
                        return float('inf')  # Final checkpoint comes last
                    return 0
                
                # Sort the checkpoints by step number
                direct_checkpoints.sort(key=get_step_number)
                
                # Display checkpoint count and offer evenly-spaced selection
                logger.info(f"Found {len(direct_checkpoints)} checkpoints in '{folder}'")
                
                # Extract all the step numbers for evenly spaced selection
                checkpoint_steps = []
                for cp in direct_checkpoints:
                    match = step_pattern.search(cp)
                    if match:
                        checkpoint_steps.append(int(match.group(1)))
                
                if checkpoint_steps:
                    min_step = min(checkpoint_steps)
                    max_step = max(checkpoint_steps) if checkpoint_steps else 0
                    logger.info(f"Step range: {min_step} to {max_step}")
                    
                    # First, ask if user wants to select specific checkpoints or use evenly spaced selection
                    selection_mode = input("Choose selection mode ([1] Specific checkpoints, [2] Evenly spaced, [3] All checkpoints): ")
                    
                    if selection_mode.strip() == "1":
                        # Show all checkpoints for manual selection
                        logger.info(f"Checkpoints in '{folder}':")
                        for i, cp in enumerate(direct_checkpoints):
                            logger.info(f"[{i+1}] {cp}")
                        
                        cp_selections = input("Enter checkpoint number(s) to evaluate (comma-separated, 'all', or 'none'): ")
                        
                        if cp_selections.strip().lower() == 'all':
                            selected_checkpoints.extend(direct_checkpoints)
                        elif cp_selections.strip().lower() != 'none':
                            try:
                                indices = [int(x.strip()) - 1 for x in cp_selections.split(',') if x.strip()]
                                selected_checkpoints.extend([direct_checkpoints[i] for i in indices if 0 <= i < len(direct_checkpoints)])
                            except (ValueError, IndexError):
                                logger.warning(f"Invalid checkpoint selection for '{folder}'. Skipping.")
                    
                    elif selection_mode.strip() == "2":
                        # Evenly spaced selection
                        num_checkpoints = input(f"How many evenly spaced checkpoints do you want to select (max {len(direct_checkpoints)}): ")
                        
                        try:
                            num_checkpoints = int(num_checkpoints.strip())
                            if num_checkpoints <= 0 or num_checkpoints > len(direct_checkpoints):
                                logger.warning(f"Invalid number. Using all {len(direct_checkpoints)} checkpoints.")
                                num_checkpoints = len(direct_checkpoints)
                            
                            if num_checkpoints == 1:
                                # Just select the final checkpoint
                                final_checkpoint = direct_checkpoints[-1] if 'final' in direct_checkpoints[-1] else direct_checkpoints[-1]
                                selected_checkpoints.append(final_checkpoint)
                                logger.info(f"Selected 1 checkpoint: {final_checkpoint}")
                            else:
                                # Calculate evenly spaced indices
                                if num_checkpoints >= len(direct_checkpoints):
                                    # If requesting all or more than available, just use all
                                    evenly_spaced = direct_checkpoints
                                else:
                                    # Create evenly spaced indices
                                    step_size = max(1, len(direct_checkpoints) // num_checkpoints)
                                    
                                    # Find the closest checkpoints to the desired step intervals
                                    target_steps = []
                                    for i in range(num_checkpoints):
                                        # Calculate ideal step number at this position
                                        if i == num_checkpoints - 1:  # Always include last checkpoint
                                            target_steps.append(max_step)
                                        else:
                                            target_step = min_step + (i * (max_step - min_step)) // (num_checkpoints - 1)
                                            target_steps.append(target_step)
                                    
                                    # Find the closest available checkpoint to each target
                                    evenly_spaced = []
                                    for target_step in target_steps:
                                        # Find closest checkpoint to target_step
                                        closest_cp = min(direct_checkpoints, 
                                                        key=lambda cp: abs(get_step_number(cp) - target_step) 
                                                        if get_step_number(cp) != float('inf') else 0)
                                        
                                        if closest_cp not in evenly_spaced:  # Avoid duplicates
                                            evenly_spaced.append(closest_cp)
                                
                                # Add the selected checkpoints
                                selected_checkpoints.extend(evenly_spaced)
                                
                                # Display the selected steps
                                selected_steps = [get_step_number(cp) for cp in evenly_spaced 
                                                if get_step_number(cp) != float('inf')]
                                selected_steps_str = ", ".join(map(str, selected_steps))
                                logger.info(f"Selected {len(evenly_spaced)} checkpoints with steps: {selected_steps_str}")
                                
                        except ValueError:
                            logger.warning(f"Invalid input. Skipping checkpoint selection for '{folder}'.")
                    
                    elif selection_mode.strip() == "3":
                        # Select all checkpoints
                        selected_checkpoints.extend(direct_checkpoints)
                        logger.info(f"Selected all {len(direct_checkpoints)} checkpoints")
                    
                    else:
                        logger.warning(f"Invalid selection mode. Skipping checkpoint selection for '{folder}'.")
                
                else:
                    # No valid step numbers found
                    logger.warning(f"No checkpoints with valid step numbers found in '{folder}'. Skipping selection.")
            
            else:
                # No direct checkpoints, list second-level folders
                result = s3_client.list_objects_v2(Bucket=bucket, Prefix=folder, Delimiter='/')
                subfolders = []
                
                if 'CommonPrefixes' in result:
                    subfolders = [p['Prefix'] for p in result['CommonPrefixes']]
                
                if subfolders:
                    logger.info(f"Subfolders in '{folder}':")
                    for i, subfolder in enumerate(subfolders):
                        logger.info(f"[{i+1}] {subfolder}")
                    
                    # Get user selection for subfolders
                    sub_selections = input("Enter subfolder number(s) to check for checkpoints (comma-separated, 'all', or 'none'): ")
                    
                    selected_subfolders = []
                    if sub_selections.strip().lower() == 'all':
                        selected_subfolders = subfolders
                    elif sub_selections.strip().lower() != 'none':
                        try:
                            indices = [int(x.strip()) - 1 for x in sub_selections.split(',') if x.strip()]
                            selected_subfolders = [subfolders[i] for i in indices if 0 <= i < len(subfolders)]
                        except (ValueError, IndexError):
                            logger.warning(f"Invalid subfolder selection for '{folder}'. Skipping.")
                    
                    # Check each selected subfolder for checkpoints using the same evenly-spaced approach
                    for subfolder in selected_subfolders:
                        result = s3_client.list_objects_v2(Bucket=bucket, Prefix=subfolder, Delimiter='/')
                        if 'CommonPrefixes' in result:
                            checkpoints = [p['Prefix'] for p in result['CommonPrefixes'] 
                                        if 'checkpoint_' in p['Prefix'] or 'final' in p['Prefix']]
                            
                            # Sort checkpoints by step number
                            step_pattern = re.compile(r'checkpoint_(\d+)')
                            checkpoints.sort(key=lambda path: int(step_pattern.search(path).group(1)) 
                                            if step_pattern.search(path) else float('inf'))
                            
                            if checkpoints:
                                # Extract all the step numbers
                                checkpoint_steps = []
                                for cp in checkpoints:
                                    match = step_pattern.search(cp)
                                    if match:
                                        checkpoint_steps.append(int(match.group(1)))
                                
                                if checkpoint_steps:
                                    min_step = min(checkpoint_steps)
                                    max_step = max(checkpoint_steps)
                                    logger.info(f"Found {len(checkpoints)} checkpoints in '{subfolder}' (steps {min_step} to {max_step})")
                                    
                                    # Ask selection mode
                                    selection_mode = input("Choose selection mode ([1] Specific checkpoints, [2] Evenly spaced, [3] All): ")
                                    
                                    if selection_mode.strip() == "1":
                                        # Manual selection
                                        logger.info(f"Checkpoints in '{subfolder}':")
                                        for i, cp in enumerate(checkpoints):
                                            logger.info(f"[{i+1}] {cp}")
                                        
                                        cp_selections = input("Enter checkpoint number(s) to evaluate (comma-separated, 'all', or 'none'): ")
                                        
                                        if cp_selections.strip().lower() == 'all':
                                            selected_checkpoints.extend(checkpoints)
                                        elif cp_selections.strip().lower() != 'none':
                                            try:
                                                indices = [int(x.strip()) - 1 for x in cp_selections.split(',') if x.strip()]
                                                selected_checkpoints.extend([checkpoints[i] for i in indices if 0 <= i < len(checkpoints)])
                                            except (ValueError, IndexError):
                                                logger.warning(f"Invalid checkpoint selection for '{subfolder}'. Skipping.")
                                    
                                    elif selection_mode.strip() == "2":
                                        # Evenly spaced selection
                                        num_checkpoints = input(f"How many evenly spaced checkpoints do you want to select (max {len(checkpoints)}): ")
                                        
                                        try:
                                            num_checkpoints = int(num_checkpoints.strip())
                                            if num_checkpoints <= 0 or num_checkpoints > len(checkpoints):
                                                logger.warning(f"Invalid number. Using all {len(checkpoints)} checkpoints.")
                                                num_checkpoints = len(checkpoints)
                                            
                                            if num_checkpoints == 1:
                                                # Just select the final checkpoint
                                                final_checkpoint = checkpoints[-1]
                                                selected_checkpoints.append(final_checkpoint)
                                                logger.info(f"Selected 1 checkpoint: {final_checkpoint}")
                                            else:
                                                # Calculate evenly spaced indices
                                                if num_checkpoints >= len(checkpoints):
                                                    # If requesting all or more than available, just use all
                                                    evenly_spaced = checkpoints
                                                else:
                                                    # Find the closest checkpoints to the desired step intervals
                                                    target_steps = []
                                                    for i in range(num_checkpoints):
                                                        # Calculate ideal step number at this position
                                                        if i == num_checkpoints - 1:  # Always include last checkpoint
                                                            target_steps.append(max_step)
                                                        else:
                                                            target_step = min_step + (i * (max_step - min_step)) // (num_checkpoints - 1)
                                                            target_steps.append(target_step)
                                                    
                                                    # Find the closest available checkpoint to each target
                                                    def get_step(cp):
                                                        match = step_pattern.search(cp)
                                                        return int(match.group(1)) if match else float('inf')
                                                    
                                                    evenly_spaced = []
                                                    for target_step in target_steps:
                                                        # Find closest checkpoint to target_step
                                                        closest_cp = min(checkpoints, key=lambda cp: 
                                                                        abs(get_step(cp) - target_step) if get_step(cp) != float('inf') else 0)
                                                        
                                                        if closest_cp not in evenly_spaced:  # Avoid duplicates
                                                            evenly_spaced.append(closest_cp)
                                                
                                                # Add the selected checkpoints
                                                selected_checkpoints.extend(evenly_spaced)
                                                
                                                # Display the selected steps
                                                selected_steps = [get_step(cp) for cp in evenly_spaced 
                                                                if get_step(cp) != float('inf')]
                                                selected_steps_str = ", ".join(map(str, selected_steps))
                                                logger.info(f"Selected {len(evenly_spaced)} checkpoints with steps: {selected_steps_str}")
                                                
                                        except ValueError:
                                            logger.warning(f"Invalid input. Skipping checkpoint selection for '{subfolder}'.")
                                    
                                    elif selection_mode.strip() == "3":
                                        # Select all checkpoints
                                        selected_checkpoints.extend(checkpoints)
                                        logger.info(f"Selected all {len(checkpoints)} checkpoints")
                                    
                                    else:
                                        logger.warning(f"Invalid selection mode. Skipping checkpoint selection for '{subfolder}'.")
        
        # Sort selected checkpoints by step number
        step_pattern = re.compile(r'checkpoint_(\d+)')
        
        def get_step_number(path):
            match = step_pattern.search(path)
            if match:
                return int(match.group(1))
            elif 'final' in path:
                return float('inf')  # Final checkpoint comes last
            return 0
        
        selected_checkpoints.sort(key=get_step_number)
        logger.info(f"Selected a total of {len(selected_checkpoints)} checkpoint folders for evaluation")
        
        return selected_checkpoints
        
    except Exception as e:
        logger.error(f"Error accessing S3 bucket: {e}")
        return []

def download_single_checkpoint(s3_folder, bucket, local_dir, checkpoint_folder, logger):
    """
    Download a single checkpoint from S3 bucket
    
    Args:
        s3_folder: Path to folder in S3 containing checkpoints (can be empty for interactive mode)
        bucket: S3 bucket name
        local_dir: Local directory to download checkpoints
        checkpoint_folder: Specific checkpoint folder to download (e.g. '20250402-225935-Qwen2.5-3B-math_train_4digit/checkpoint_700/')
        logger: Logger instance
        
    Returns:
        Tuple of (path to downloaded checkpoint, checkpoint name, step number)
    """
    s3 = boto3.client('s3')
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    
    # Log which checkpoint folder we're processing from S3
    logger.info(f"Processing S3 checkpoint folder: {checkpoint_folder}")
    
    # Extract parts from the checkpoint path
    folder_parts = checkpoint_folder.strip('/').split('/')
    
    # Extract the model directory and checkpoint folder
    if len(folder_parts) >= 2:
        model_dir = folder_parts[-2]  # e.g. '20250402-225935-Qwen2.5-3B-math_train_4digit'
        checkpoint_dir = folder_parts[-1]  # e.g. 'checkpoint_700'
        
        # Create a combined name that includes both model info and checkpoint
        folder_name = f"{model_dir}_{checkpoint_dir}"
    else:
        # Fallback if path structure is different
        folder_name = folder_parts[-1]
    
    logger.info(f"Model directory: {model_dir if len(folder_parts) >= 2 else 'unknown'}")
    logger.info(f"Checkpoint directory: {checkpoint_dir if len(folder_parts) >= 2 else folder_parts[-1]}")
    
    # For the local path, we'll use just the checkpoint dir to avoid long paths
    local_checkpoint_path = os.path.join(local_dir, checkpoint_dir if len(folder_parts) >= 2 else folder_name)
    
    # Extract step number from checkpoint name
    step = 0
    step_match = re.search(r'.*checkpoint_(\d+)', checkpoint_folder)
    if step_match:
        step = int(step_match.group(1))
    elif "final" in checkpoint_folder.lower():
        step = 99999  # Use a large number for final checkpoint
    
    # Log the checkpoint info for tracking
    logger.info(f"Extracted step number: {step}")
    
    # For CSV output, use the combined name
    full_checkpoint_name = checkpoint_folder.strip('/')
    
    if os.path.exists(local_checkpoint_path):
        # Clear the directory if it exists
        logger.info(f"Clearing existing checkpoint directory: {local_checkpoint_path}")
        for file_path in glob.glob(os.path.join(local_checkpoint_path, "**"), recursive=True):
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(local_checkpoint_path, exist_ok=True)
            
    logger.info(f"Downloading checkpoint: {full_checkpoint_name}")
    
    try:
        # Download checkpoint files
        checkpoint_prefix = checkpoint_folder  # Already full path in interactive mode
        
        # List all objects in this checkpoint
        paginator = s3.get_paginator('list_objects_v2')
        download_count = 0
        
        for page in paginator.paginate(Bucket=bucket, Prefix=f"{checkpoint_prefix}"):
            for obj in page.get('Contents', []):
                file_key = obj['Key']
                if file_key.endswith('/'):
                    continue
                    
                # Get relative path within checkpoint
                rel_path = file_key.replace(f"{checkpoint_prefix}", "")
                if rel_path.startswith('/'):
                    rel_path = rel_path[1:]
                local_file_path = os.path.join(local_checkpoint_path, rel_path)
                
                # Create directory if needed
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                
                # Download file
                s3.download_file(bucket, file_key, local_file_path)
                download_count += 1
                
                # Log progress periodically
                if download_count % 100 == 0:
                    logger.info(f"Downloaded {download_count} files for checkpoint {checkpoint_dir if len(folder_parts) >= 2 else folder_name}")
        
        logger.info(f"Completed download of checkpoint {full_checkpoint_name}: {download_count} files")
        return local_checkpoint_path, full_checkpoint_name, step
    
    except Exception as e:
        logger.error(f"Error downloading checkpoint {full_checkpoint_name}: {e}")
        return None, None, None
    
def list_checkpoint_folders(s3_folder, bucket, logger):
    """
    List all checkpoint folders in S3 bucket
    
    Args:
        s3_folder: Path to folder in S3 containing checkpoints
        bucket: S3 bucket name
        logger: Logger instance
        
    Returns:
        List of S3 checkpoint folders sorted by step number
    """
    s3 = boto3.client('s3')
    
    # List objects in the S3 folder
    logger.info(f"Listing checkpoints in s3://{bucket}/{s3_folder}")
    
    # Find all checkpoint folders
    checkpoint_folders = set()
    paginator = s3.get_paginator('list_objects_v2')
    
    try:
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_folder):
            for obj in page.get('Contents', []):
                # Extract the checkpoint folder name
                key_parts = obj['Key'].split('/')
                if len(key_parts) > 1:
                    if 'checkpoint_' in key_parts[-2]:
                        checkpoint_folders.add('/'.join(key_parts[:-1]))
                    elif 'final' in key_parts[-2]:
                        checkpoint_folders.add('/'.join(key_parts[:-1]))
    except Exception as e:
        logger.error(f"Error listing checkpoint folders: {e}")
        return []
    
    checkpoint_folders = list(checkpoint_folders)
    
    # Sort checkpoints by step number
    step_pattern = re.compile(r'checkpoint_(\d+)')
    
    def get_step_number(path):
        match = step_pattern.search(path)
        if match:
            return int(match.group(1))
        elif 'final' in path:
            return float('inf')  # Final checkpoint comes last
        return 0
    
    checkpoint_folders.sort(key=get_step_number)
    logger.info(f"Found {len(checkpoint_folders)} checkpoint folders in S3 bucket")
    
    return checkpoint_folders

def delete_checkpoint(local_checkpoint_path, logger):
    """
    Delete a local checkpoint to free up disk space
    
    Args:
        local_checkpoint_path: Path to local checkpoint directory
        logger: Logger instance
        
    Returns:
        True if successful, False otherwise
    """
    if not os.path.exists(local_checkpoint_path):
        logger.warning(f"Checkpoint path does not exist: {local_checkpoint_path}")
        return True  # Already gone
    
    try:
        logger.info(f"Deleting local checkpoint: {local_checkpoint_path}")
        # First remove all files
        for file_path in glob.glob(os.path.join(local_checkpoint_path, "**"), recursive=True):
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Then remove directories
        shutil.rmtree(local_checkpoint_path, ignore_errors=True)
        
        if os.path.exists(local_checkpoint_path):
            logger.warning(f"Failed to completely remove directory: {local_checkpoint_path}")
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error deleting checkpoint {local_checkpoint_path}: {e}")
        return False

def main():
    """
    Main function to run evaluation on math datasets across multiple checkpoints
    """
    args = parse_args()
    
    # Setup logger
    logger = setup_logger(args.log_dir)
    logger.info("Starting math problem evaluation")
    logger.info(f"Arguments: {args}")
    
    ## Create output directory
    # os.makedirs(args.output_dir, exist_ok=True)
    
    # Use args.dataset_type and args.s3_folder and args.seed to make a new directory within args.output_dir
    detailed_output_dir = os.path.join(args.output_dir, args.dataset_type, args.s3_folder, f'seed-{args.seed}')
    if os.path.exists(detailed_output_dir):
        raise FileExistsError(f"Directory already exists: {detailed_output_dir}")
    # Create the directory for detailed output
    os.makedirs(detailed_output_dir)
    
    # Track results for each checkpoint
    results = []
    
    # Load dataset from the updated dataset.py module
    logger.info(f"Loading dataset of type: {args.dataset_type}...")
    try:        
        # Load dataset with is_eval=True to use test/validation splits
        raw_dataset = load_dataset_for_training_or_eval(
            dataset_name=args.dataset_type, 
            path=args.dataset_path,
            max_samples=args.max_samples,
            is_eval=True,
            logger=logger
        )
        
        # Format dataset for evaluation
        formatted_data = format_dataset_for_eval(
            dataset_name=args.dataset_type,
            dataset=raw_dataset,
            logger=logger
        )
        
        logger.info(f"Prepared {len(formatted_data['questions'])} questions for evaluation")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return

    if args.evaluate_bedrock_model:
        bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

        # Run evaluation by sending requests to API server
        logger.info("Using asynchronous evaluation")
        eval_results = asyncio.run(evaluate_with_bedrock_async(
            formatted_data=formatted_data,
            logger=logger,
            bedrock_client=bedrock_client,
            args=args,
        ))

        result = {
            "bedrock_model_name": args.bedrock_model_name,
            "accuracy": eval_results["accuracy"],
            "last_number_accuracy": eval_results["last_number_accuracy"],
            "format_accuracy": eval_results["format_accuracy"],
            "valid_percentage": eval_results["valid_percentage"],
            "correct": eval_results["correct"],
            "last_number_correct": eval_results["last_number_correct"],
            "format_correct": eval_results["format_correct"],
            "total": eval_results["total"],
            "valid_predictions": eval_results["valid_predictions"],
            "dataset_type": args.dataset_type
        }
        results.append(result)

        # Save detailed results for this checkpoint
        # Use a clean filename that avoids path issues
        detailed_output_path = os.path.join(args.output_dir, f"{args.bedrock_model_name}_predictions.json")
        with open(detailed_output_path, 'w', encoding="utf-8") as f:
            # Filter out full responses to save space (they can be large)
            simplified_predictions = []
            for pred in eval_results["predictions"]:
                simplified_pred = {
                    "question": pred["question"],
                    "extracted_answer": pred["extracted_answer"],
                    "last_number_answer": pred.get("last_number_answer"),
                    "target_answer": pred["target_answer"],
                    "format_correct": pred["format_correct"],
                    "answer_correct": pred["answer_correct"],
                    "last_number_correct": pred.get("last_number_correct", False),
                    "question_id": pred["question_id"],
                    "question_metadata": pred.get("question_metadata", {})
                }
                simplified_predictions.append(simplified_pred)
            
            json.dump(simplified_predictions, f, indent=2)
        
        logger.info(f"Bedrock Model Name {args.bedrock_model_name}:")
        logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
        logger.info(f"  Last Number Accuracy: {eval_results['last_number_accuracy']:.4f}")
        logger.info(f"  Format Accuracy: {eval_results['format_accuracy']:.4f}")
        logger.info(f"  Valid Answer %: {eval_results['valid_percentage']:.4f}")
        logger.info(f"  Correct: {eval_results['correct']}/{eval_results['total']}")

    else:
        
        # Handle direct loading of HF model if specified
        if args.model_name:
            logger.info(f"Directly evaluating Hugging Face model: {args.model_name}")
            checkpoint_paths = [args.model_name]
            checkpoint_names = [args.model_name]  # Store both paths and display names
            model_family = args.model_family
            if model_family == "auto":
                # Try to detect model family from model name
                if "llama" in args.model_name.lower():
                    model_family = "llama"
                    logger.info(f"Auto-detected model family from name: {model_family}")
                elif "qwen" in args.model_name.lower():
                    model_family = "qwen"
                    logger.info(f"Auto-detected model family from name: {model_family}")
                else:
                    logger.warning(f"Could not auto-detect model family from name. Defaulting to 'llama'.")
                    model_family = "llama"
        else:
            # Get checkpoint paths - either from S3 or locally
            checkpoint_paths = []
            checkpoint_names = []  # Track full names for CSV output
            
            if args.bucket:
                # Interactive selection mode if bucket is provided but no S3 folder
                if not args.s3_folder:
                    logger.info(f"Interactive checkpoint selection mode from bucket: {args.bucket}")
                    s3_checkpoint_folders = interactive_s3_checkpoint_selection(args.bucket, logger)
                    
                    if not s3_checkpoint_folders:
                        logger.error("No checkpoints selected. Exiting.")
                        return
                    
                    logger.info(f"Selected {len(s3_checkpoint_folders)} checkpoints for evaluation")
                    
                    # For S3, we'll download, evaluate, and delete one at a time
                    # Just store a placeholder and process them in the evaluation loop
                    checkpoint_paths = ["S3_CHECKPOINT_PLACEHOLDER"] * len(s3_checkpoint_folders)
                    # Store the actual paths for reference
                    checkpoint_names = s3_checkpoint_folders.copy()
                else:
                    logger.info(f"Using S3 bucket {args.bucket} with sequential download and evaluation")
                    # Get list of checkpoint folders in S3, but don't download them all at once
                    s3_checkpoint_folders = list_checkpoint_folders(args.s3_folder, args.bucket, logger)
                    
                    if not s3_checkpoint_folders:
                        logger.error("No checkpoints found in S3. Exiting.")
                        return
                    
                    logger.info(f"Found {len(s3_checkpoint_folders)} checkpoints in S3")
                    
                    # For S3, we'll download, evaluate, and delete one at a time
                    # Just store a placeholder and process them in the evaluation loop
                    checkpoint_paths = ["S3_CHECKPOINT_PLACEHOLDER"] * len(s3_checkpoint_folders)
                    # Store the actual paths for reference
                    checkpoint_names = s3_checkpoint_folders.copy()
                
                # Determine model family if set to auto (use the last checkpoint for detection)
                if args.model_family == "auto":
                    # Download a small file from the last checkpoint to detect model family
                    s3 = boto3.client('s3')
                    last_checkpoint = s3_checkpoint_folders[-1]
                    config_key = f"{last_checkpoint}/config.json"
                    
                    try:
                        # Create a temporary directory for config
                        temp_dir = os.path.join(args.local_dir, "temp_config")
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_config = os.path.join(temp_dir, "config.json")
                        
                        # Download config file
                        s3.download_file(args.bucket, config_key, temp_config)
                        
                        # Detect model family
                        model_family = detect_model_family(temp_dir) or "llama"
                        logger.info(f"Auto-detected model family: {model_family}")
                        
                        # Clean up
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception as e:
                        logger.error(f"Error detecting model family from S3: {e}")
                        model_family = "llama"
                        logger.warning("Defaulting to 'llama' model family")
                else:
                    model_family = args.model_family
            else:
                logger.info(f"Reading checkpoints from local directory {args.local_dir}")
                checkpoint_paths = get_local_checkpoints(args.local_dir, logger)
                
                if not checkpoint_paths:
                    logger.error("No checkpoints found. Exiting.")
                    return
                
                # For local checkpoints, track full paths as checkpoint names as well
                checkpoint_names = [os.path.basename(path) for path in checkpoint_paths]
                
                # Determine model family if set to auto
                model_family = args.model_family
                if model_family == "auto":
                    detected_family = detect_model_family(checkpoint_paths[-1])
                    if detected_family:
                        model_family = detected_family
                        logger.info(f"Auto-detected model family: {model_family}")
                    else:
                        logger.warning("Could not auto-detect model family. Defaulting to 'llama'.")
                        model_family = "llama"
            
        logger.info(f"Using model family: {model_family}")
        
        # Evaluate each checkpoint
        for i, checkpoint_path in enumerate(checkpoint_paths):
            try:
                # Initialize variables to track checkpoint info for all code paths
                step = 0
                csv_checkpoint_name = ""
                checkpoint_display_name = ""
                
                # For S3 sequential processing, download the checkpoint now
                if args.bucket and checkpoint_path == "S3_CHECKPOINT_PLACEHOLDER":
                    s3_checkpoint_folder = s3_checkpoint_folders[i]
                    
                    # Log which checkpoint we're about to download (with index)
                    logger.info(f"\n===== Processing checkpoint {i+1}/{len(s3_checkpoint_folders)} =====")
                    logger.info(f"S3 path: {s3_checkpoint_folder}")
                    
                    # Download this checkpoint and get checkpoint name and step
                    checkpoint_path, full_checkpoint_name, step = download_single_checkpoint(
                        args.s3_folder if args.s3_folder else "", 
                        args.bucket, 
                        args.local_dir, 
                        s3_checkpoint_folder, 
                        logger
                    )
                    
                    if not checkpoint_path:
                        logger.error(f"Failed to download checkpoint from {s3_checkpoint_folder}. Skipping.")
                        continue
                        
                    # Use the full checkpoint name for CSV results
                    csv_checkpoint_name = full_checkpoint_name
                    
                    # Extract a friendlier checkpoint name for display - just the "checkpoint_NNN" part
                    checkpoint_match = re.search(r'(checkpoint_\d+|final)', full_checkpoint_name)
                    if checkpoint_match:
                        checkpoint_display_name = checkpoint_match.group(1)
                    else:
                        checkpoint_display_name = full_checkpoint_name.split('/')[-1]
                    
                    # Log checkpoint info for tracking
                    logger.info(f"Successfully downloaded checkpoint: {checkpoint_display_name} with step {step}")
                else:
                    # Local checkpoint or HF model
                    checkpoint_display_name = os.path.basename(checkpoint_path) if os.path.exists(checkpoint_path) else checkpoint_path
                    
                    # For CSV results, use the full name (with path for local checkpoints)
                    csv_checkpoint_name = checkpoint_names[i] if i < len(checkpoint_names) else checkpoint_display_name
                    
                    # Extract step number for plotting
                    step_match = re.search(r'checkpoint_(\d+)', checkpoint_display_name)
                    step = int(step_match.group(1)) if step_match else 0
                    if "final" in checkpoint_display_name:
                        # For final checkpoint, use a large step number for plotting at the end
                        step = 99999
                    
                    # If using direct model name, use 0 as step
                    if checkpoint_path == args.model_name:
                        step = 0
                
                logger.info(f"\n===== Evaluating checkpoint/model: {checkpoint_display_name} (step {step}) =====")
                
                processes = None
            
                # Start separate processes for each model instance
                logger.info(f"Starting {args.batch_size} separate processes for model instances")
                processes = start_model_server(checkpoint_path, model_family, args)
                
                # Wait for all servers to be ready
                logger.info("Waiting for all model servers to start...")
                if not check_model_servers_ready(processes, timeout=300):
                    logger.error("Not all model servers started successfully. Aborting.")
                    cleanup_processes(processes)
                    continue
                    
                logger.info(f"All {len(processes)} model servers are ready")
                    
                # Run evaluation by sending requests to API server
                # Use asynchronous version
                logger.info("Using asynchronous evaluation")
                eval_results = asyncio.run(evaluate_model_via_api_async(
                    formatted_data=formatted_data,
                    logger=logger,
                    args=args,
                    processes=processes
                ))
                
                # Cleanup 
                cleanup_processes(processes)

                # Save results for this checkpoint - now with correct name and step info
                checkpoint_result = {
                    "checkpoint": csv_checkpoint_name,  # Now properly tracked for all code paths
                    "step": step,  # Now properly tracked
                    "accuracy": eval_results["accuracy"],
                    "last_number_accuracy": eval_results["last_number_accuracy"],
                    "format_accuracy": eval_results["format_accuracy"],
                    "valid_percentage": eval_results["valid_percentage"],
                    "correct": eval_results["correct"],
                    "last_number_correct": eval_results["last_number_correct"],
                    "format_correct": eval_results["format_correct"],
                    "total": eval_results["total"],
                    "valid_predictions": eval_results["valid_predictions"],
                    "dataset_type": args.dataset_type
                }
                
                results.append(checkpoint_result)
                
                # Save detailed results for this checkpoint
                # Use a clean filename that avoids path issues
                safe_filename = checkpoint_display_name.replace('/', '_').replace('\\', '_')
                detailed_output_path = os.path.join(args.output_dir, f"{safe_filename}_predictions.json")
                with open(detailed_output_path, 'w', encoding="utf-8") as f:
                    # Filter out full responses to save space (they can be large)
                    simplified_predictions = []
                    for pred in eval_results["predictions"]:
                        simplified_pred = {
                            "question": pred["question"],
                            "extracted_answer": pred["extracted_answer"],
                            "last_number_answer": pred.get("last_number_answer"),
                            "target_answer": pred["target_answer"],
                            "format_correct": pred["format_correct"],
                            "answer_correct": pred["answer_correct"],
                            "last_number_correct": pred.get("last_number_correct", False),
                            "question_id": pred["question_id"],
                            "question_metadata": pred.get("question_metadata", {})
                        }
                        simplified_predictions.append(simplified_pred)
                    
                    json.dump(simplified_predictions, f, indent=2)
                
                logger.info(f"Checkpoint {checkpoint_display_name} (step {step}):")
                logger.info(f"  Accuracy: {eval_results['accuracy']:.4f}")
                logger.info(f"  Last Number Accuracy: {eval_results['last_number_accuracy']:.4f}")
                logger.info(f"  Format Accuracy: {eval_results['format_accuracy']:.4f}")
                logger.info(f"  Valid Answer %: {eval_results['valid_percentage']:.4f}")
                logger.info(f"  Correct: {eval_results['correct']}/{eval_results['total']}")
                
                # Save results to CSV after each checkpoint (for S3 sequential processing)
                # This allows tracking progress even if the script is interrupted
                if i > 0:
                    temp_results_df = pd.DataFrame(results)
                    temp_csv_path = os.path.join(args.output_dir, f"{args.dataset_type}_evaluation_results_progress.csv")
                    temp_results_df.to_csv(temp_csv_path, index=False)
                    logger.info(f"Saved intermediate results to {temp_csv_path}")
                
                # Delete the local checkpoint to free up space if it was downloaded from S3
                if args.bucket and checkpoint_path != "S3_CHECKPOINT_PLACEHOLDER" and checkpoint_path != args.model_name:
                    if delete_checkpoint(checkpoint_path, logger):
                        logger.info(f"Successfully deleted local checkpoint: {checkpoint_path}")
                    else:
                        logger.warning(f"Could not completely delete checkpoint: {checkpoint_path}")
                        
                    # Force garbage collection and GPU memory clearing
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            except Exception as e:
                logger.error(f"Error evaluating checkpoint: {e}", exc_info=True)
        
        logger.info("\n===== Evaluation Summary =====")
        if results:
            best_checkpoint = max(results, key=lambda x: x["accuracy"])
            best_last_number = max(results, key=lambda x: x["last_number_accuracy"])
            
            logger.info(f"Best checkpoint by formal answer: {best_checkpoint['checkpoint']}")
            logger.info(f"  Accuracy: {best_checkpoint['accuracy']:.4f}")
            logger.info(f"  Last Number Accuracy: {best_checkpoint['last_number_accuracy']:.4f}")
            logger.info(f"  Format accuracy: {best_checkpoint['format_accuracy']:.4f}")
            
            logger.info(f"Best checkpoint by last number: {best_last_number['checkpoint']}")
            logger.info(f"  Accuracy: {best_last_number['accuracy']:.4f}")
            logger.info(f"  Last Number Accuracy: {best_last_number['last_number_accuracy']:.4f}")
            logger.info(f"  Format accuracy: {best_last_number['format_accuracy']:.4f}")        
    
    # Skip visualization if no results
    if not results:
        logger.error("No evaluation results to visualize. Exiting.")
        return
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(detailed_output_dir, f"{args.dataset_type}_evaluation_results.csv")

    # Check if CSV already exists and append results instead of overwriting
    if os.path.exists(csv_path):
        # Read existing CSV and append new results
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        combined_df.to_csv(csv_path, index=False)
        logger.info(f"Appended evaluation results to existing file {csv_path}")
    else:
        # Create new CSV if it doesn't exist
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved evaluation results to new file {csv_path}")
    
    # Generate plot
    try:
        # Sort by step for visualization
        results_df = results_df.sort_values(by="step")
        
        plt.figure(figsize=(12, 10))
        
        # Plot accuracy metrics
        plt.subplot(3, 1, 1)
        plt.plot(results_df["step"], results_df["accuracy"], marker='o', linestyle='-', 
                label="Answer Accuracy", color='blue')
        plt.plot(results_df["step"], results_df["last_number_accuracy"], marker='d', linestyle='-.', 
                label="Last Number Accuracy", color='red')
        plt.xlabel("Training Step")
        plt.ylabel("Accuracy")
        plt.title(f"{args.dataset_type.upper()} Answer Accuracy ({model_family.upper()} model)")
        plt.legend()
        plt.grid(True)
        
        # Plot format accuracy
        plt.subplot(3, 1, 2)
        plt.plot(results_df["step"], results_df["format_accuracy"], marker='x', linestyle='--', 
                label="Format Accuracy", color='green')
        plt.xlabel("Training Step")
        plt.ylabel("Format Accuracy")
        plt.title("Percentage of Responses with Correct Format")
        plt.grid(True)
        
        # Plot valid predictions percentage
        plt.subplot(3, 1, 3)
        plt.plot(results_df["step"], results_df["valid_percentage"], marker='s', linestyle='-', color='purple')
        plt.xlabel("Training Step")
        plt.ylabel("Valid Answer %")
        plt.title("Percentage of Valid Numerical Answers Extracted")
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(detailed_output_dir, f"{args.dataset_type}_performance.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved performance plot to {plot_path}")
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
    
    logger.info(f"Evaluation completed. All results saved to {args.output_dir}")

def parse_args():
    """
    Parse command line arguments for evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate math problem performance across model checkpoints')
    parser.add_argument('--s3_folder', type=str, default="", help='S3 folder path containing checkpoints (optional with bucket)')
    parser.add_argument('--bucket', type=str, default="", help='S3 bucket name (if provided without s3_folder, enables interactive selection)')
    parser.add_argument('--local_dir', type=str, default='./checkpoints', 
                        help='Local directory to download checkpoints (or to read from if no bucket specified)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Directory to save results')
    parser.add_argument('--max_length', type=int, default=1024, help='Maximum sequence length for generation')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of model instances to run in parallel')
    parser.add_argument('--model_family', type=str, choices=['llama', 'qwen', 'auto'], default='auto', 
                        help='Model family (llama, qwen, or auto for automatic detection)')
    parser.add_argument('--bits', type=int, choices=[0, 4, 8], default=0, 
                        help='Quantization bits (0 for no quantization, 4 or 8 for quantized)')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperature for generation')
    parser.add_argument('--top_p', type=float, default=0.99, help='Top-p for generation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--log_dir', type=str, default='./eval_logs', help='Directory for logs')
    parser.add_argument('--max_samples', type=int, default=0, 
                        help='Maximum number of samples to evaluate (0 for all)')
    parser.add_argument('--api_hostname', type=str, default="127.0.0.1", help='Hostname for API server')
    parser.add_argument('--api_port', type=int, default=8000, help='Port for API server')
    parser.add_argument('--max_concurrent', type=int, default=4, 
                        help='Maximum number of concurrent API requests (0 for batch_size*2)')
    # New option to directly load model from Hugging Face
    parser.add_argument('--model_name', type=str, default='', 
                        help='Hugging Face model name to load directly (bypasses checkpoint loading)')
    # Port range for separate model instances
    parser.add_argument('--port_start', type=int, default=8100, 
                        help='Starting port number for separate model instances')
    # Dataset options
    parser.add_argument('--dataset_type', type=str, 
                    default='gsm8k',
                    help='Type of dataset to evaluate')
    parser.add_argument('--dataset_path', type=str, default='./datasets/math_test_5digit.jsonl',
                        help='Path to dataset file (for custom JSONL datasets)')
    # For evaluating bedrock models
    parser.add_argument('--evaluate_bedrock_model', action='store_true',
                        help='Set boolean to true to evaluate the bedrock model instead of checkpoints / hf models')
    parser.add_argument('--bedrock_model_name', type=str, default='c35_haiku', 
                        help='Bedrock model name abbreviated. Refer to config.py for exact model names')

    return parser.parse_args()

if __name__ == "__main__":
    main()