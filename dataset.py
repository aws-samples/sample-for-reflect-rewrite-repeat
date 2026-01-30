import logging
import os
import json
import re
from typing import Dict, List, Optional, Union, Any, Tuple
from datasets import load_dataset, Dataset
import math

# Dataset registry - Define all datasets and their configurations here
DATASET_REGISTRY = {
    "gsm8k": {
        "source": "gsm8k",
        "config": "main",
        "train_split": "train",
        "eval_split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_extractor": "gsm8k",  # Special handling for GSM8K answer format
        "description": "Grade School Math 8K dataset"
    },
    "svamp": {
        "source": "ChilleD/SVAMP",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "question_field": "question_concat",
        "answer_field": "Answer",
        "answer_extractor": "direct",  # Direct numerical answer
        "description": "SVAMP word math problems"
    },
    "aime2024": {
        "source": "HuggingFaceH4/aime_2024",
        "config": None,
        "train_split": "train",
        "eval_split": "train",
        "question_field": "problem",
        "answer_field": "answer",
        "answer_extractor": "direct",  # Direct numerical answer
        "description": "AIME 2024 only math problems"
    },
    "aime2025": {
        "source": "TIGER-Lab/AIME25", #"yentinglin/aime_2025",
        "config": 'default',
        "train_split": "train",
        "eval_split": "train",
        "question_field": "question", #"problem",
        "answer_field": "answer",
        "answer_extractor": "direct",  # Direct numerical answer
        "description": "AIME 2025 only math problems"
    },
    "aime": {
        "source": "di-zhang-fdu/AIME_1983_2024",
        "config": None,
        "train_split": "train",
        "eval_split": "train",
        "question_field": "Question",
        "answer_field": "Answer",
        "answer_extractor": "direct",  # Direct numerical answer
        "description": "AIME 1983-2024 math problems"
    },
    "sat_math": {
        "source": "hails/agieval-sat-math",
        "config": None,
        "train_split": "test",
        "eval_split": "test",
        "question_field": "query",
        "answer_field": "gold",
        "answer_extractor": "sat_math",  
        "description": "SAT math problems"
    },
    "numina_math": {
        "source": "AI-MO/NuminaMath-TIR",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "question_field": "problem",
        "answer_field": "solution",
        "answer_extractor": "numina_math",  
        "description": "Numina math problems"
    },
    "amc23": {
        "source": "zwhe99/amc23",
        "config": None,
        "train_split": "train",
        "eval_split": "test",
        "question_field": "question",
        "answer_field": "answer",
        "answer_extractor": "direct",  
        "description": "amc2023 math problems"
    },
    "symbolic_data": {
        "source": "SAGI-1/SYMBOLIC_DATA_FACTS_2_FOL",
        "config": None,
        "train_split": "train",
        "eval_split": "train",
        "question_field": "instruction",
        "answer_field": "answer",
        "answer_extractor": "direct",  
        "description": "symblic data problems"
    },
    "symbolic_folio": {
        "source": "yale-nlp/FOLIO",
        "config": None,
        "train_split": "train",
        "eval_split": "validation",
        "question_field": "premises",
        "answer_field": "premises-FOL",
        "answer_extractor": "direct",  
        "description": "symblic data problems"
    },
    "math_jsonl": {
        "source": "local_jsonl",  # Special handling for local JSONL files
        "question_field": "question",
        "answer_field": "answer",
        "description": "Custom math problems in JSONL format"
    }
    # Add new datasets here
}

def load_jsonl_dataset(file_path):
    """
    Load dataset from a JSONL file
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        Dataset object with questions and answers
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    return Dataset.from_list(data)

def get_dataset_config(dataset_name: str) -> Dict:
    """
    Get dataset configuration from the registry
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dataset configuration dictionary
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not found in registry")
    return DATASET_REGISTRY[dataset_name]

def load_dataset_for_training_or_eval(dataset_name: str, split: Optional[str] = None, 
                                     path: Optional[str] = None, max_samples: Optional[int] = None, 
                                     is_eval: bool = False, logger=None) -> Dataset:
    """
    Unified function to load a dataset for either training or evaluation
    
    Args:
        dataset_name: Name of the dataset in the registry
        split: Dataset split to use (overrides default from registry)
        path: Path to dataset file for local datasets
        max_samples: Maximum number of samples to load
        is_eval: Whether this is for evaluation (affects default split choice)
        logger: Logger instance
        
    Returns:
        HuggingFace Dataset object
    """
    logger = logger or logging.getLogger('dataset_handler')
    
    # Get dataset configuration
    try:
        config = get_dataset_config(dataset_name)
    except ValueError as e:
        # For custom datasets not in the registry, try loading directly
        if os.path.exists(dataset_name):
            logger.info(f"Dataset name {dataset_name} is a file path, trying to load directly")
            return load_local_dataset(dataset_name, max_samples, logger)
        else:
            # Try loading as a HuggingFace dataset
            logger.info(f"Trying to load {dataset_name} as a HuggingFace dataset")
            try:
                dataset = load_dataset(dataset_name, revision="main")
                available_splits = dataset.keys()
                
                # Determine which split to use
                effective_split = None
                if split and split in available_splits:
                    effective_split = split
                elif is_eval:
                    # For evaluation, prioritize test > validation > first available
                    if "test" in available_splits:
                        effective_split = "test"
                    elif "validation" in available_splits:
                        effective_split = "validation"
                    else:
                        effective_split = list(available_splits)[0]
                else:
                    # For training, prioritize train > first available
                    if "train" in available_splits:
                        effective_split = "train"
                    else:
                        effective_split = list(available_splits)[0]
                
                logger.info(f"Using split '{effective_split}' for dataset {dataset_name}")
                dataset = dataset[effective_split]
                logger.info(f"Loaded dataset {dataset_name} with {len(dataset)} examples")
                return dataset
            except Exception as e2:
                logger.warning(f"Failed to load dataset: {e}")
                logger.warning(f"And also failed to load as direct HuggingFace dataset: {e2}")
                raise ValueError(f"Unknown dataset {dataset_name} and not a valid file path")
    
    # Determine which split to use more explicitly
    effective_split = None
    if split:
        # User-specified split takes highest priority
        effective_split = split
        logger.info(f"Using user-specified split '{effective_split}'")
    elif is_eval:
        # For evaluation, use eval_split from config
        effective_split = config.get("eval_split", "test")
        logger.info(f"Using evaluation split '{effective_split}' from config")
    else:
        # For training, use train_split from config
        effective_split = config.get("train_split", "train")
        logger.info(f"Using training split '{effective_split}' from config")
    
    logger.info(f"Loading dataset {dataset_name} (split: {effective_split})")
    
    # Handle different source types
    source = config.get("source")
    
    if source == "local_jsonl":
        if not path:
            raise ValueError(f"Path is required for local dataset {dataset_name}")
        return load_local_json_dataset(path, max_samples, logger)
    elif source == "local_csv":
        if not path:
            raise ValueError(f"Path is required for local dataset {dataset_name}")
        return load_local_csv_dataset(path, max_samples, logger)
    else:
        # Load from HuggingFace datasets
        try:
            dataset_config = config.get("config")
            if dataset_config:
                dataset = load_dataset(source, dataset_config, split=effective_split, revision="main")
            else:
                dataset = load_dataset(source, split=effective_split, revision="main")
                
            # Apply max_samples limit
            if max_samples and max_samples < len(dataset):
                dataset = dataset.select(range(max_samples))
            
            logger.info(f"Loaded {len(dataset)} examples from {dataset_name} ({effective_split} split)")
            return dataset
        except Exception as e:
            logger.warning(f"Error loading HuggingFace dataset {source}: {e}")
            raise

def extract_answer_from_text(dataset_name: str, answer_text: Any) -> Union[float, str, None]:
    """
    Extract answer from text based on dataset-specific extraction method
    
    Args:
        dataset_name: Name of the dataset
        answer_text: Text containing the answer
        
    Returns:
        Extracted answer (float, string, or None if extraction fails)
    """
    try:
        config = get_dataset_config(dataset_name)
        extractor_type = config.get("answer_extractor", "direct")
        
        if extractor_type == "gsm8k":
            # GSM8K format: Extract answer from "#### number" format
            answer_match = re.search(r"####\s*(-?\d+\.?\d*)", str(answer_text))
            if answer_match:
                return float(answer_match.group(1))
            # Fallback to last number in the text
            numbers = re.findall(r"-?\d+\.?\d*", str(answer_text))
            return float(numbers[-1]) if numbers else None
        elif extractor_type == "numina_math":
            boxed_contents = extract_boxed_content(answer_text)
            numbers = extract_numbers_from_boxed(boxed_contents)
            return numbers
        elif extractor_type == "direct":
            # Direct numerical or string answer
            if isinstance(answer_text, (int, float)):
                return float(answer_text)
            try:
                return float(answer_text)
            except (ValueError, TypeError):
                return answer_text
        elif extractor_type == "tagged":
            # Extract from <answer> tags
            answer_match = re.search(r"<answer>(.*?)</answer>", str(answer_text), re.DOTALL)
            if answer_match:
                answer = answer_match.group(1).strip()
                try:
                    return float(answer)
                except (ValueError, TypeError):
                    return answer
            return None
        else:
            # Default: try to parse as number, otherwise return as string
            if isinstance(answer_text, (int, float)):
                return float(answer_text)
            try:
                return float(answer_text)
            except (ValueError, TypeError):
                return answer_text
    except ValueError:
        # For datasets not in registry, try direct conversion
        if isinstance(answer_text, (int, float)):
            return float(answer_text)
        try:
            return float(answer_text)
        except (ValueError, TypeError):
            return answer_text

def get_question_and_answer_fields(dataset_name: str) -> Tuple[str, str]:
    """
    Get the field names for question and answer in a dataset
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Tuple of (question_field, answer_field)
    """
    try:
        config = get_dataset_config(dataset_name)
        return config.get("question_field", "question"), config.get("answer_field", "answer")
    except ValueError:
        # Default field names for unknown datasets
        return "question", "answer"

def format_dataset_for_eval(dataset_name: str, dataset, logger=None) -> Dict:
    """
    Format a dataset for evaluation, extracting questions, answers, and metadata
    
    Args:
        dataset_name: Name of the dataset
        dataset: Dataset object
        logger: Logger instance
        
    Returns:
        Dictionary with questions, answers, and metadata lists
    """
    logger = logger or logging.getLogger('dataset_handler')
    
    question_field, answer_field = get_question_and_answer_fields(dataset_name)
    
    questions = []
    raw_answers = []
    processed_answers = []
    metadata = []
    
    for i, item in enumerate(dataset):
        # Extract question
        if question_field in item:
            questions.append(item[question_field])
        else:
            logger.warning(f"Question field '{question_field}' not found in item {i}")
            questions.append("")
        
        # Extract and process answer
        if answer_field in item:
            raw_answer = item[answer_field]
            raw_answers.append(raw_answer)
            
            # Process the answer according to dataset type
            extracted_answer = extract_answer_from_text(dataset_name, raw_answer)
            processed_answers.append(extracted_answer)
        else:
            logger.warning(f"Answer field '{answer_field}' not found in item {i}")
            raw_answers.append(None)
            processed_answers.append(None)
        
        # Extract metadata (all fields except question and answer)
        meta = {k: v for k, v in item.items() if k not in [question_field, answer_field]}
        metadata.append(meta)
    
    logger.info(f"Formatted {len(questions)} examples from dataset {dataset_name}")
    
    return {
        "questions": questions,
        "raw_answers": raw_answers,
        "answers": processed_answers,
        "metadata": metadata
    }

# Keep the existing local dataset loading functions
def load_local_json_dataset(file_path, max_samples=None, logger=None):
    """
    Load dataset from a local JSON or JSONL file
    
    Args:
        file_path: Path to local JSON file
        max_samples: Maximum number of samples to load
        logger: Logger instance
        
    Returns:
        Dataset object
    """
    logger = logger or logging.getLogger('dataset_handler')
    logger.info(f"Loading local JSON dataset from: {file_path}")
    
    processed_data = []
    
    try:
        # Check if it's a JSONL file (one JSON object per line)
        is_jsonl = False
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            try:
                # Try to parse first line as JSON
                json.loads(first_line)
                is_jsonl = True
            except json.JSONDecodeError:
                pass
        
        # Load the data
        if is_jsonl:
            with open(file_path, 'r', encoding='utf-8') as f:
                line_count = 0
                for line in f:
                    if max_samples and line_count >= max_samples:
                        break
                    
                    data = json.loads(line)
                    processed_data.append(data)
                    line_count += 1
        else:
            # Regular JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                
                if isinstance(data_list, dict):
                    # If it's a dictionary, convert to list
                    data_list = [data_list]
                
                for i, data in enumerate(data_list):
                    if max_samples and i >= max_samples:
                        break
                    
                    processed_data.append(data)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(processed_data)
        
        logger.info(f"Loaded {len(dataset)} items from {file_path}")
        return dataset
    
    except Exception as e:
        error_msg = f"Error loading JSON dataset from {file_path}: {e}"
        logger.warning(error_msg)
        raise ValueError(error_msg)

def load_local_csv_dataset(file_path, max_samples=None, logger=None):
    """
    Load dataset from a local CSV file
    
    Args:
        file_path: Path to local CSV file
        max_samples: Maximum number of samples to load
        logger: Logger instance
        
    Returns:
        Dataset object
    """
    logger = logger or logging.getLogger('dataset_handler')
    logger.info(f"Loading local CSV dataset from: {file_path}")
    
    import csv
    processed_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                processed_data.append(row)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(processed_data)
        
        logger.info(f"Loaded {len(dataset)} examples from {file_path}")
        return dataset
    
    except Exception as e:
        error_msg = f"Error loading CSV dataset from {file_path}: {e}"
        logger.error(error_msg)
        raise ValueError(error_msg)

def load_local_dataset(file_path, max_samples=None, logger=None):
    """
    Load dataset from a local file (auto-detect format)
    
    Args:
        file_path: Path to local file
        max_samples: Maximum number of samples to load
        logger: Logger instance
        
    Returns:
        Dataset object
    """
    logger = logger or logging.getLogger('dataset_handler')
    
    # Determine file type based on extension
    if file_path.endswith('.json') or file_path.endswith('.jsonl'):
        return load_local_json_dataset(file_path, max_samples, logger)
    elif file_path.endswith('.csv'):
        return load_local_csv_dataset(file_path, max_samples, logger)
    else:
        error_msg = f"Unsupported file format: {file_path}"
        logger.error(error_msg)
        raise ValueError(error_msg)