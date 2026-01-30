from typing import Any, List, Dict, Optional
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
import logging
import torch
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel, 
    PeftConfig
)
import os
import json
import re

def load_model(
    model_name_or_path: str,
    model_family: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
    logger=None,
    use_4bit: bool = False,
    use_8bit: bool = False,
    use_flash_attention: bool = True,  # New parameter to toggle Flash Attention
    use_vllm: bool = True,  # New parameter to toggle vLLM
) -> tuple[Any, PreTrainedTokenizer]:
    """
    Load model and tokenizer from the specified path or name,
    with optimizations for speed using vLLM and Flash Attention
    
    Args:
        model_name_or_path: Name or path of the model to load
        model_family: Model family (llama or qwen)
        trust_remote_code: Whether to trust remote code when loading models
        bf16: Whether to use bfloat16 precision
        device_map: Device mapping configuration
        logger: Logger instance
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
        use_flash_attention: Whether to use Flash Attention for faster attention computation
        use_vllm: Whether to use vLLM backend for faster inference

    Returns:
        Tuple of (model, tokenizer)
    """
    logger = logger or logging.getLogger('dpo_training')
    logger.info(f"Loading {model_family} model from {model_name_or_path}")
    
    # Load tokenizer based on model family
    if model_family == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code,
            revision="main"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = '<|endoftext|>'
        if tokenizer.eos_token is None:
            tokenizer.eos_token = '<|endoftext|>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, 
            trust_remote_code=trust_remote_code,
            revision="main"
        )

    logger.info(f"PAD = {tokenizer.pad_token}, EOS = {tokenizer.eos_token}")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Setting pad_token to eos_token: {tokenizer.pad_token}")
    
    # If using vLLM backend, load with vLLM instead of transformers
    if use_vllm:
        try:
            from vllm import LLM
            
            # Check if we're not trying to use conflicting options
            if use_4bit or use_8bit:
                logger.warning("Quantization options are ignored when using vLLM backend")
                
            logger.info("Loading model with vLLM backend for faster inference")
            
            # Configure vLLM parameters
            vllm_kwargs = {
                "model": model_name_or_path,
                "trust_remote_code": trust_remote_code,
                "tensor_parallel_size": 1,  # Adjust based on your GPU count
            }
            
            # Set precision for vLLM
            if bf16:
                vllm_kwargs["dtype"] = "bfloat16"
            else:
                vllm_kwargs["dtype"] = "float16"
                
            # Create the vLLM model
            vllm_model = LLM(**vllm_kwargs)
            
            # Wrap the vLLM model to provide compatibility with the transformers interface
            # This is a placeholder - you would need to implement a proper wrapper class
            # that provides the same API surface as the transformers models
            class VLLMModelWrapper:
                def __init__(self, vllm_model):
                    self.vllm_model = vllm_model
                    # Add additional attributes and methods as needed
                
                def generate(self, **kwargs):
                    # Convert transformers generation parameters to vLLM parameters
                    # Implementation details would depend on your specific needs
                    pass
                    
                # Implement other necessary methods
            
            model = VLLMModelWrapper(vllm_model)
            logger.info("vLLM model loaded successfully")
            return model, tokenizer
            
        except ImportError:
            logger.warning("vLLM not installed. Falling back to standard transformers loading.")
            use_vllm = False
    
    # Set up model loading kwargs based on quantization settings and other optimizations
    model_kwargs = {
        "trust_remote_code": trust_remote_code,
        "device_map": device_map,
    }
    
    # Add Flash Attention if requested and available
    if use_flash_attention:
        try:
            # For transformers >= 4.34.0
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Using Flash Attention 2 for faster attention computation")
        except Exception as e:
            logger.warning(f"Could not enable Flash Attention: {str(e)}. Continuing without it.")
    
    # Add quantization settings
    if use_4bit:
        logger.info("Using 4-bit quantization")
        try:
            import bitsandbytes as bnb
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": torch.bfloat16 if bf16 else torch.float32,
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            })
        except ImportError:
            logger.warning("bitsandbytes not installed. Cannot use 4-bit quantization.")
            raise
    elif use_8bit:
        logger.info("Using 8-bit quantization")
        try:
            import bitsandbytes as bnb
            model_kwargs.update({
                "load_in_8bit": True,
            })
        except ImportError:
            logger.error("bitsandbytes not installed. Cannot use 8-bit quantization.")
            raise
    else:
        logger.info("Using bf16")
        model_kwargs.update({
            "torch_dtype": torch.bfloat16 if bf16 else "auto",
        })
        
    if model_family.lower() == "llama":
        logger.info("Loading LlamaForCausalLM model...")
        model = LlamaForCausalLM.from_pretrained(model_name_or_path, revision="main", **model_kwargs)
    elif model_family.lower() == "qwen":
        logger.info("Loading AutoModelForCausalLM model for Qwen...")
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, revision="main", **model_kwargs)
    else:
        error_msg = f"Unsupported model family: {model_family}. Use 'llama' or 'qwen'."
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Model loaded successfully. Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    return model, tokenizer

# def load_model(
#     model_name_or_path: str,
#     model_family: str,
#     trust_remote_code: bool = False,
#     bf16: bool = True,
#     device_map=None,
#     logger=None,
#     use_4bit: bool = False,
#     use_8bit: bool = False,
# ) -> tuple[Any, PreTrainedTokenizer]:
#     """
#     Load model and tokenizer from the specified path or name,
#     supporting both LLaMA and Qwen model families
    
#     Args:
#         model_name_or_path: Name or path of the model to load
#         model_family: Model family (llama or qwen)
#         trust_remote_code: Whether to trust remote code when loading models
#         bf16: Whether to use bfloat16 precision
#         device_map: Device mapping configuration
#         logger: Logger instance
#         use_4bit: Whether to use 4-bit quantization
#         use_8bit: Whether to use 8-bit quantization

#     Returns:
#         Tuple of (model, tokenizer)
#     """
#     logger = logger or logging.getLogger('dpo_training')
#     logger.info(f"Loading {model_family} model from {model_name_or_path}")
    
#     if model_family == 'qwen':
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name_or_path, 
#             trust_remote_code=trust_remote_code
#         )
#         if tokenizer.pad_token is None:
#             tokenizer.pad_token = '<|endoftext|>'
#         if tokenizer.eos_token is None:
#             tokenizer.eos_token = '<|endoftext|>'
        
#     else:
#         tokenizer = AutoTokenizer.from_pretrained(
#             model_name_or_path, 
#             trust_remote_code=trust_remote_code
#         )

#     logger.info(f"PAD = {tokenizer.pad_token}, EOS = {tokenizer.eos_token}")
    
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         logger.info(f"Setting pad_token to eos_token: {tokenizer.pad_token}")
    
#     # Set up model loading kwargs based on quantization settings
#     model_kwargs = {
#         "trust_remote_code": trust_remote_code,
#         "device_map": device_map,
#     }
    
#     # Add quantization settings
#     if use_4bit:
#         logger.info("Using 4-bit quantization")
#         model_kwargs.update({
#             "load_in_4bit": True,
#             "bnb_4bit_compute_dtype": torch.bfloat16 if bf16 else torch.float32,
#             "bnb_4bit_use_double_quant": True,
#             "bnb_4bit_quant_type": "nf4",
#         })
#     elif use_8bit:
#         logger.info("Using 8-bit quantization")
#         model_kwargs.update({
#             "load_in_8bit": True,
#         })
#     else:
#         model_kwargs.update({
#             "torch_dtype": torch.bfloat16 if bf16 else "auto",
#         })
        
#     if model_family.lower() == "llama":
#         logger.info("Loading LlamaForCausalLM model...")
#         model = LlamaForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
#     elif model_family.lower() == "qwen":
#         logger.info("Loading AutoModelForCausalLM model for Qwen...")
#         model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
#     else:
#         error_msg = f"Unsupported model family: {model_family}. Use 'llama' or 'qwen'."
#         logger.error(error_msg)
#         raise ValueError(error_msg)
    
#     logger.info(f"Model loaded successfully. Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
#     return model, tokenizer

# Function to add LoRA adapter to model
def add_lora_adapter(model, args, logger=None):
    """
    Add LoRA adapter to a model for parameter-efficient fine-tuning
    
    Args:
        model: The model to add the adapter to
        args: Command line arguments with LoRA configuration
        logger: Logger instance
        
    Returns:
        The model with LoRA adapter
    """
    logger = logger or logging.getLogger('dpo_training')
    
    # Define LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.lora_target_modules.split(",") if args.lora_target_modules else None,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    logger.info(f"Adding LoRA adapter to model with the following config:")
    logger.info(f"  - rank (r): {args.lora_r}")
    logger.info(f"  - alpha: {args.lora_alpha}")
    logger.info(f"  - target modules: {args.lora_target_modules}")
    logger.info(f"  - dropout: {args.lora_dropout}")
    
    # Prepare model for quantization if needed
    if args.use_4bit or args.use_8bit:
        logger.info(f"Preparing model for {'4-bit' if args.use_4bit else '8-bit'} training")
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.use_gradient_checkpointing
        )
    
    # Add LoRA adapter
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    model.print_trainable_parameters()
    
    return model

def load_checkpoint_with_lora(checkpoint_dir, model_family, device_map, logger, use_4bit=False, use_8bit=False, trust_remote_code=False, is_trainable=False, use_flash_attention=True, use_vllm=False):
    """
    Load model and tokenizer from a checkpoint directory, specifically handling LoRA checkpoints
    with optimizations for speed
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        model_family: Model family (llama or qwen)
        device_map: Device mapping configuration
        logger: Logger instance
        use_4bit: Whether to use 4-bit quantization
        use_8bit: Whether to use 8-bit quantization
        trust_remote_code: Whether to trust remote code when loading models
        use_flash_attention: Whether to use Flash Attention for faster attention computation
        use_vllm: Whether to use vLLM backend for faster inference
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import PeftModel, PeftConfig
    
    logger.info(f"Loading checkpoint from {checkpoint_dir}")
    
    # Check if this is a PEFT/LoRA checkpoint
    is_peft_checkpoint = os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json"))
    
    if is_peft_checkpoint:
        logger.info("Detected PEFT/LoRA checkpoint")
        
        # Load the PEFT configuration
        peft_config = PeftConfig.from_pretrained(checkpoint_dir)
        base_model_name_or_path = peft_config.base_model_name_or_path
        
        logger.info(f"Base model: {base_model_name_or_path}")
        
        # First load the base model
        base_model, tokenizer = load_model(
            base_model_name_or_path,
            model_family,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            logger=logger,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            use_flash_attention=use_flash_attention,
            use_vllm=use_vllm
        )
        
        # Note: vLLM doesn't currently support PEFT/LoRA out of the box
        if use_vllm:
            logger.warning("vLLM may not fully support PEFT/LoRA. Proceed with caution.")
        
        # Then load the PEFT adapter on top of it
        model = PeftModel.from_pretrained(
            base_model,
            checkpoint_dir,
            is_trainable=True  # Explicitly set to trainable mode
        )

        logger.info("Successfully loaded LoRA adapter on a base model")
        
    else:
        # Regular checkpoint (not PEFT/LoRA)
        logger.info("Loading regular (non-PEFT) checkpoint")
        model, tokenizer = load_model(
            checkpoint_dir,
            model_family,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            logger=logger,
            use_4bit=use_4bit,
            use_8bit=use_8bit,
            use_flash_attention=use_flash_attention,
            use_vllm=use_vllm
        )
        
        logger.info("Successfully loaded a base model")

    # Make sure model is in training mode
    if is_trainable:
        model.train()

        # Ensure all parameters requiring gradients have them enabled
        for param in model.parameters():
            if param.requires_grad:
                param.requires_grad_(True)
    else:
        model.eval()

    return model, tokenizer

def get_local_checkpoints(local_dir, logger):
    """
    Find model checkpoints in local directory
    
    Args:
        local_dir: Local directory containing checkpoints
        logger: Logger instance
        
    Returns:
        List of local checkpoint paths sorted by step number
    """
    # Make sure directory exists
    if not os.path.exists(local_dir):
        logger.error(f"Local directory {local_dir} does not exist")
        return []
    
    # Find all checkpoint folders
    checkpoint_paths = []
    for item in os.listdir(local_dir):
        item_path = os.path.join(local_dir, item)
        if os.path.isdir(item_path) and (item.startswith('checkpoint_') or 'final' in item):
            # Check if this looks like a model checkpoint (has tokenizer.json)
            if os.path.exists(os.path.join(item_path, "tokenizer.json")):
                checkpoint_paths.append(item_path)
            else:
                logger.warning(f"Skipping {item_path} - doesn't look like a model checkpoint")
    
    # Sort checkpoints by step number
    step_pattern = re.compile(r'checkpoint_(\d+)')
    
    def get_step_number(path):
        match = step_pattern.search(path)
        if match:
            return int(match.group(1))
        elif 'final' in path:
            return float('inf')  # Final checkpoint comes last
        return 0
    
    checkpoint_paths.sort(key=get_step_number)
    logger.info(f"Found {len(checkpoint_paths)} checkpoint folders in {local_dir}")
    
    return checkpoint_paths

def load_checkpoint_metadata(checkpoint_dir, logger=None):
    """
    Load checkpoint metadata including remaining problems and metrics history
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        logger: Logger instance
    
    Returns:
        Dictionary with loaded metadata
    """
    logger = logger or logging.getLogger('dpo_training')
    
    metadata = {}
    
    # Load remaining problems
    remaining_path = os.path.join(checkpoint_dir, "remaining_problems.json")
    if os.path.exists(remaining_path):
        with open(remaining_path, "r", encoding="utf-8") as f:
            metadata["remaining_problems"] = json.load(f)
        logger.info(f"Loaded remaining problems: {len(metadata['remaining_problems']['dataset'])} dataset problems, "
                   f"{len(metadata['remaining_problems']['bucket'])} bucket problems")
    else:
        logger.warning(f"No remaining problems file found at {remaining_path}")
        metadata["remaining_problems"] = {"dataset": [], "bucket": []}
    
    # Load metrics history
    metrics_path = os.path.join(checkpoint_dir, "metrics_history.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_history = json.load(f)
        metadata["metrics_history"] = metrics_history
        logger.info(f"Loaded metrics history with {len(metrics_history.keys())} metrics")
    else:
        logger.warning(f"No metrics history file found at {metrics_path}")
        metadata["metrics_history"] = {}
    
    return metadata

def create_model_instance(checkpoint_path, model_family, bits):
    """
    Create and return a model instance for inference, handling LoRA checkpoints
    
    Args:
        checkpoint_path: Path to model checkpoint or Hugging Face model name
        model_family: Model architecture family (llama, qwen)
        bits: Number of bits
        
    Returns:
        Tuple of (model, tokenizer)
    """
    # Create a simple logger if not available
    logger = logging.getLogger("create_model_instance")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
    
    # Check if this is a PEFT/LoRA checkpoint
    is_peft_checkpoint = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))
    
    # Configure device map
    device_map = "auto"
    
    if is_peft_checkpoint:
        # Use the load_checkpoint_with_lora function for LoRA checkpoints
        model, tokenizer = load_checkpoint_with_lora(
            checkpoint_path,
            model_family,
            device_map=device_map,
            logger=logger,
            use_4bit=(bits == 4),
            use_8bit=(bits == 8),
            trust_remote_code=True,
            is_trainable=False
        )
    else:
        model, tokenizer = load_model(
            checkpoint_path,
            model_family,
            device_map=device_map,
            logger=logger,
            use_4bit=(bits == 4),
            use_8bit=(bits == 8),
            trust_remote_code=True,
        )
    
    # Set model to evaluation mode
    model.eval()
    
    return model, tokenizer

def detect_model_family(checkpoint_path):
    """
    Detect model family from config file
    
    Args:
        checkpoint_path: Path to model checkpoint
        
    Returns:
        Model family (llama, qwen, or None if couldn't detect)
    """
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        model_type = config.get("model_type", "").lower()
        architectures = config.get("architectures", [])
        
        # Check for Llama
        if model_type == "llama" or any("llama" in arch.lower() for arch in architectures):
            return "llama"
        
        # Check for Qwen
        if model_type == "qwen" or any("qwen" in arch.lower() for arch in architectures):
            return "qwen"
            
        # Check for other indicators in config
        if "hidden_act" in config and config["hidden_act"] == "silu":
            return "llama"  # Likely Llama
            
    except Exception as e:
        print(f"Error detecting model family: {e}")
    
    return None  # Could not determine