import re
import torch
import torch.optim as optim
import torch.nn.functional as F
import logging
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
    GenerationConfig,
)
from config import system_prompt
from datasets import Dataset

import gzip

def calculate_gzip_distance(a: str, b: str) -> float:
    """
    Calculate the distance between two strings using gzip compression.
    
    This function computes a normalized distance metric based on the compression
    of concatenated strings versus individual compressions:
    distance = (gzip(a+b) - min(gzip(a), gzip(b))) / max(gzip(a), gzip(b))
    
    Parameters:
    -----------
    a : str
        First input string
    b : str
        Second input string
    
    Returns:
    --------
    float
        The normalized distance between the two strings
    """
    # Encode strings to bytes for gzip compression
    a_bytes = a.encode('utf-8')
    b_bytes = b.encode('utf-8')
    
    # Concatenate strings
    ab_bytes = a_bytes + b_bytes
    
    # Compress individual strings and concatenated string
    gzip_a_size = len(gzip.compress(a_bytes))
    gzip_b_size = len(gzip.compress(b_bytes))
    gzip_ab_size = len(gzip.compress(ab_bytes))
    
    # Calculate minimum and maximum compressed sizes
    min_size = min(gzip_a_size, gzip_b_size)
    max_size = max(gzip_a_size, gzip_b_size)
    
    # Calculate distance using the formula
    distance = (gzip_ab_size - min_size) / max_size
    
    return distance

def extract_answer_from_text(text):
    """
    Extract answer from text using regex pattern matching
    
    Args:
        text: Text to extract answer from
        
    Returns:
        Extracted answer or None if no answer found
    """
    # Try to extract from <answer> tags
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # If no tags found, return None
    return None

def check_format_correctness(text):
    """
    Check if the format of the text is correct:
    - Has both reasoning and answer tags
    - Tags are not duplicated
    - Tags have content between them
    
    Args:
        text: Text to check format of
        
    Returns:
        Boolean indicating whether the format is correct
    """
    # Check if the required tags are present
    reasoning_tags = re.findall(r"<reasoning>", text)
    reasoning_close_tags = re.findall(r"</reasoning>", text)
    answer_tags = re.findall(r"<answer>", text)
    answer_close_tags = re.findall(r"</answer>", text)
    
    # Check if we have exactly one of each tag (no duplicates)
    if len(reasoning_tags) != 1 or len(reasoning_close_tags) != 1 or len(answer_tags) != 1 or len(answer_close_tags) != 1:
        return False
    
    # Check if tags are properly nested (no overlapping)
    reasoning_start = text.find("<reasoning>")
    reasoning_end = text.find("</reasoning>")
    answer_start = text.find("<answer>")
    answer_end = text.find("</answer>")
    
    # Check proper nesting and ordering
    if not (reasoning_start < reasoning_end and answer_start < answer_end):
        return False
    
    # It's usually expected that reasoning comes before answer, but this is optional
    # Uncommenting this would enforce that reasoning must come before answer
    # if not (reasoning_start < reasoning_end < answer_start < answer_end):
    #     return False
    
    # Extract content between tags and check if it's non-empty
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    if not reasoning_match or not answer_match:
        return False
    
    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip()
    
    # Both tags must have non-empty content
    return len(reasoning) > 0 and len(answer) > 0

def extract_numbers(text):
    pattern = r'-?[\d,]+\.?\d*'
    matches = re.findall(pattern, text)
    if matches:
        try:
            # Remove commas for conversion to float
            clean_number = matches[0].replace(',', '')
            return float(clean_number)
        except ValueError:
            return None

# Function to calculate answer correctness
def calculate_answer_correctness(model_answer, oracle_answer, is_symbolic = False, logger = None):
    """
    Calculate answer correctness (exact match or partial match)
    
    Args:
        model_answer: The extracted answer from the model's output
        oracle_answer: The ground truth answer
        
    Returns:
        Boolean indicating whether the answer is correct
    """
    logger = logger or logging.getLogger('dpo_training')

    if model_answer is None:
        return False
    
    # Check if the answer still contains placeholder text
    if "[The final numerical answer]" in model_answer:
        return False
        
    # Clean up answer and oracle_answer for comparison
    if isinstance(model_answer, str) and not is_symbolic:
        answer_clean = extract_numbers(model_answer)
    else:
        answer_clean = model_answer  # or handle non-string case appropriately

    if isinstance(oracle_answer, str) and not is_symbolic:
        oracle_clean = extract_numbers(oracle_answer)
    else:
        oracle_clean = oracle_answer  # or handle non-string case appropriately
    
    # Empty strings should never match
    if not answer_clean or not oracle_clean:
        return False
    
    # Check if answers match
    if is_symbolic:
        gzip_dist = abs(calculate_gzip_distance(answer_clean.lower(), oracle_clean.lower()))
        logger.info(f"gzip dist: {gzip_dist}")
        if gzip_dist < 0.15: #check approximate kolmogorov complexity
            return True
        else:
            return False

    if answer_clean == oracle_clean:
        return True
    elif abs(answer_clean-oracle_clean) < 1e-5:
        return True
    else:
        return False

from transformers import StoppingCriteriaList, StoppingCriteria

class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids
    
    def __call__(self, input_ids, scores, **kwargs):
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

class StopOnTextAfter(StoppingCriteria):
    def __init__(self, tokenizer, stop_texts, input_length):
        self.tokenizer = tokenizer
        self.stop_texts = stop_texts
        self.input_length = input_length
    
    def __call__(self, input_ids, scores, **kwargs):
        generated_text = self.tokenizer.decode(input_ids[0][self.input_length:])
        for stop_text in self.stop_texts:
            if stop_text in generated_text:
                stop_pos = generated_text.find(stop_text) + len(stop_text)
                
                if stop_pos >= len(generated_text) - 1:
                    return True
                
                after_stop = generated_text[stop_pos:].strip()
                if after_stop: 
                    return True
        return False
    
@torch.no_grad()
def generate_answer(
    model,
    tokenizer: PreTrainedTokenizer,
    question: str,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    logger=None
) -> str:
    """
    Generate an answer for a given question
    
    Args:
        model: Model to generate responses with
        tokenizer: Tokenizer for the model
        question: Question to generate responses for
        max_new_tokens: Maximum new tokens of generated responses
        temperature: Temperature for generation
        top_p: Top-p parameter for generation
        logger: Logger instance
        
    Returns:
        Generated answer
    """
    logger = logger or logging.getLogger('dpo_training')
    model.eval()

    # Format prompt
    chat_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": question,
        },
    ]
    
    # Check if tokenizer has a chat template
    prefix_applied = False
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        chat_prompt = tokenizer.apply_chat_template(
            chat_messages, tokenize=False, add_generation_prompt=True
        )
    else:
        # Fallback for tokenizers without chat template
        chat_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n<reasoning>"
        prefix_applied = True
    
    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        return_attention_mask=True,
    ).to(model.device)

    stop_token_ids = [tokenizer.eos_token_id] 
    stop_on_tokens = StopOnTokens(stop_token_ids)

    stop_texts = ["</answer>"] 
    stop_on_text = StopOnTextAfter(tokenizer, stop_texts, model_inputs.input_ids.shape[1])

    stopping_criteria = StoppingCriteriaList([stop_on_tokens, stop_on_text])
    
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    logger.debug(f"Generating answer for question: {question[:50]}...")
    
    output_ids = model.generate(
        **model_inputs, 
        generation_config=generation_config,
        stopping_criteria=stopping_criteria,
    )
    
    # Extract only the generated part, not the prompt
    prompt_length = model_inputs["input_ids"].shape[1]
    completion = tokenizer.decode(
        output_ids[0, prompt_length:], 
        skip_special_tokens=True
    )
    
    logger.debug(f"Generated answer of length {len(completion)}")

    model.train()

    if prefix_applied:
        return "<reasoning>" + completion
    return completion

def perform_dpo_training(model, tokenizer, formatted_question, better_answer, worse_answer, beta=0.1, learning_rate=5e-6, optimizer=None, logger=None):
    """Improved DPO training implementation with explicit EOS token handling"""
    logger = logger or logging.getLogger('dpo_training')
    
    try:
        original_training_mode = model.training
        
        # Clean up the answers to ensure proper formatting
        clean_better = extract_clean_answer(better_answer)
        clean_worse = extract_clean_answer(worse_answer)

        # Log the winning and losing answers
        logger.info("=== DPO TRAINING PAIR ===")
        logger.info("--- WINNING (CHOSEN) ANSWER ---")
        logger.info(clean_better)
        logger.info("--- LOSING (REJECTED) ANSWER ---")
        logger.info(worse_answer)
        
        # Format prompt with system and user messages
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": formatted_question},
        ]
        
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            prompt = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{formatted_question}\n<|assistant|>\n"
        
        # Tokenize inputs
        inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=True).to(model.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Format chosen and rejected completions
        chosen_messages = chat_messages + [{"role": "assistant", "content": clean_better}]
        rejected_messages = chat_messages + [{"role": "assistant", "content": clean_worse}]
        
        # Tokenize full sequences
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            chosen_text = tokenizer.apply_chat_template(chosen_messages, tokenize=False, add_generation_prompt=False)
            rejected_text = tokenizer.apply_chat_template(rejected_messages, tokenize=False, add_generation_prompt=False)
        else:
            chosen_text = f"{prompt}{clean_better}"
            rejected_text = f"{prompt}{clean_worse}"
        
        chosen_tokens = tokenizer(chosen_text, return_tensors="pt", return_attention_mask=True).to(model.device)
        rejected_tokens = tokenizer(rejected_text, return_tensors="pt", return_attention_mask=True).to(model.device)
        
        # --- BEGIN: EXPLICIT EOS TOKEN HANDLING ---
        # Check if EOS token is present at the end of chosen sequence
        needs_eos_chosen = chosen_tokens.input_ids[0, -1] != tokenizer.eos_token_id
        if needs_eos_chosen:
            logger.info("Adding missing EOS token to chosen sequence")
            # Add EOS token to chosen sequence
            chosen_tokens.input_ids = torch.cat([
                chosen_tokens.input_ids, 
                torch.tensor([[tokenizer.eos_token_id]]).to(model.device)
            ], dim=1)
            chosen_tokens.attention_mask = torch.cat([
                chosen_tokens.attention_mask,
                torch.ones((1, 1), dtype=torch.long).to(model.device)
            ], dim=1)
            
        # Check if EOS token is present at the end of rejected sequence
        needs_eos_rejected = rejected_tokens.input_ids[0, -1] != tokenizer.eos_token_id
        if needs_eos_rejected:
            logger.info("Adding missing EOS token to rejected sequence")
            # Add EOS token to rejected sequence
            rejected_tokens.input_ids = torch.cat([
                rejected_tokens.input_ids, 
                torch.tensor([[tokenizer.eos_token_id]]).to(model.device)
            ], dim=1)
            rejected_tokens.attention_mask = torch.cat([
                rejected_tokens.attention_mask,
                torch.ones((1, 1), dtype=torch.long).to(model.device)
            ], dim=1)
            
        # Log confirmation of EOS tokens
        logger.info(f"Chosen sequence ends with EOS: {chosen_tokens.input_ids[0, -1] == tokenizer.eos_token_id}")
        logger.info(f"Rejected sequence ends with EOS: {rejected_tokens.input_ids[0, -1] == tokenizer.eos_token_id}")
        # --- END: EXPLICIT EOS TOKEN HANDLING ---
        
        # Create masks for response tokens (non-padding tokens after prompt)
        chosen_mask = (chosen_tokens.input_ids[:, prompt_length:] != tokenizer.pad_token_id).float()
        rejected_mask = (rejected_tokens.input_ids[:, prompt_length:] != tokenizer.pad_token_id).float()
        
        # 1. REFERENCE LOGPROBS FIRST
        with torch.no_grad():
            model.eval()
            
            ref_chosen_output = model(
                input_ids=chosen_tokens.input_ids,
                attention_mask=chosen_tokens.attention_mask
            )
            ref_rejected_output = model(
                input_ids=rejected_tokens.input_ids,
                attention_mask=rejected_tokens.attention_mask
            )
            
            ref_chosen_logprobs = compute_token_logprobs(
                ref_chosen_output.logits, 
                chosen_tokens.input_ids,
                prompt_length, 
                chosen_mask
            )
            
            ref_rejected_logprobs = compute_token_logprobs(
                ref_rejected_output.logits, 
                rejected_tokens.input_ids,
                prompt_length, 
                rejected_mask
            )
        
        # 2. POLICY LOGPROBS SECOND
        model.train() 
        
        # Policy model forward passes
        policy_chosen_output = model(
            input_ids=chosen_tokens.input_ids,
            attention_mask=chosen_tokens.attention_mask
        )
        policy_rejected_output = model(
            input_ids=rejected_tokens.input_ids,
            attention_mask=rejected_tokens.attention_mask
        )
        
        # Calculate log probabilities for policy model
        policy_chosen_logprobs = compute_token_logprobs(
            policy_chosen_output.logits, 
            chosen_tokens.input_ids, 
            prompt_length,
            chosen_mask
        )
        
        policy_rejected_logprobs = compute_token_logprobs(
            policy_rejected_output.logits, 
            rejected_tokens.input_ids,
            prompt_length, 
            rejected_mask
        )
        
        # 3. DPO LOSS CALCULATION
        model_logratios = policy_chosen_logprobs - policy_rejected_logprobs
        reference_logratios = ref_chosen_logprobs - ref_rejected_logprobs
        logits = model_logratios - reference_logratios
        loss = -F.logsigmoid(beta * logits).mean()
        
        # Track rewards (detached from computation graph)
        chosen_rewards = (policy_chosen_logprobs - ref_chosen_logprobs).detach()
        rejected_rewards = (policy_rejected_logprobs - ref_rejected_logprobs).detach()
        
        # 4. OPTIMIZATION STEP
        if optimizer is None:
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
            logger.info("Created new DPO optimizer (not using persistent one)")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log DPO training results with detailed info about rewards
        logger.info(f"DPO training completed successfully. Loss: {loss.item():.4f}")
        logger.info(f"Chosen reward: {chosen_rewards.item():.4f}, Rejected reward: {rejected_rewards.item():.4f}")
        logger.info(f"Reward difference: {(chosen_rewards - rejected_rewards).item():.4f}")
        
        if not original_training_mode:
            model.eval()
            
        return {
            "loss": loss.item(),
            "chosen_reward": chosen_rewards.item(),
            "rejected_reward": rejected_rewards.item(),
            "reward_diff": (chosen_rewards - rejected_rewards).item(),
            "success": True,
            "winning_answer": clean_better,
            "losing_answer": clean_worse
        }
        
    except Exception as e:
        if original_training_mode:
            model.train()
        else:
            model.eval()
            
        logger.error(f"Error during DPO training: {e}")
        logger.exception(e)
        return {
            "loss": None,
            "chosen_reward": None,
            "rejected_reward": None,
            "success": False,
            "error": str(e),
            "winning_answer": clean_better if 'clean_better' in locals() else None,
            "losing_answer": clean_worse if 'clean_worse' in locals() else None
        }

def compute_token_logprobs(logits, input_ids, prompt_length, mask):
    """
    Compute log probabilities for response tokens with careful handling of prompt boundaries
    
    Args:
        logits: Model output logits of shape [batch_size, sequence_length, vocab_size]
        input_ids: Input token ids of shape [batch_size, sequence_length]
        prompt_length: Length of the prompt (non-completion) portion
        mask: Mask for completion tokens of shape [batch_size, completion_length]
        
    Returns:
        Per-sequence log probabilities for completion tokens
    """
    batch_size, _ = input_ids.size()
    
    # For each position, we want to use the logits to predict the next token
    # Shift logits to align with the tokens they should predict
    shift_logits = logits[:, :-1, :]  # All logits except the last position
    
    # These are the tokens we want to predict, starting from the second position
    shift_labels = input_ids[:, 1:]   # All labels except the first position
    
    # Calculate the length of the shifted sequence
    shifted_seq_length = shift_labels.size(1)
    
    # Adjust mask if needed to match the shifted sequence length
    if mask.size(1) != shifted_seq_length:
        adjusted_mask = torch.zeros((batch_size, shifted_seq_length), 
                                   device=mask.device, 
                                   dtype=mask.dtype)
        min_len = min(mask.size(1), shifted_seq_length)
        # Copy as much of the mask as will fit
        adjusted_mask[:, :min_len] = mask[:, :min_len]
        mask = adjusted_mask
    
    # Create a mask that focuses only on the completion portion
    # We need to account for the shift when applying the mask based on prompt_length
    position_mask = torch.zeros_like(shift_labels, dtype=torch.float)
    adjusted_prompt_pos = max(0, prompt_length - 1)
    position_mask[:, adjusted_prompt_pos:] = 1.0
    
    # Compute log probabilities for each token prediction
    log_probs = F.log_softmax(shift_logits, dim=-1)
    
    # For each position, get the log probability of the actual next token
    token_logprobs = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)
    
    # Apply both the completion token mask and position mask
    # This ensures we only count log probs for the completion tokens
    masked_logprobs = token_logprobs * mask * position_mask
    
    # Sum log probabilities and normalize by the number of tokens considered
    return masked_logprobs.sum(dim=1) / ((mask * position_mask).sum(dim=1) + 1e-6)

def perform_sft_training(model, tokenizer, question, answer, learning_rate, optimizer=None, logger=None):
    """
    Perform Supervised Fine-Tuning (SFT) with improved special token handling
    and memory error handling to gracefully skip problematic examples
    """
    logger = logger or logging.getLogger('dpo_training')
    
    try:
        model.train()
        # Clean up the answer to ensure proper formatting
        clean_answer = extract_clean_answer(answer)
        
        # Log the SFT training data
        logger.info("=== SFT TRAINING ===")
        logger.info("--- QUESTION ---")
        logger.info(question + "...")
        logger.info("--- TARGET ANSWER ---")
        logger.info(clean_answer + "...")
        
        # Format prompt with system and user messages
        chat_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": clean_answer}
        ]
        
        # Apply chat template for the full conversation
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
            formatted_input = tokenizer.apply_chat_template(
                chat_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            formatted_input = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n{clean_answer}"
        
        try:
            # Tokenize the full conversation
            tokens = tokenizer(
                formatted_input, 
                return_tensors="pt", 
                padding=True,
                truncation=True,
                max_length=2048,
                return_attention_mask=True
            ).to(model.device)
            
            # Identify where the assistant's response begins
            prompt_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
                prompt_only = tokenizer.apply_chat_template(
                    prompt_messages, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
            else:
                prompt_only = f"<|system|>\n{system_prompt}\n<|user|>\n{question}\n<|assistant|>\n"
            
            # Get the prompt length precisely
            prompt_tokens = tokenizer(
                prompt_only,
                return_tensors="pt",
                add_special_tokens=True
            ).to(model.device)
            prompt_length = prompt_tokens.input_ids.shape[1]
            
            # Create labels with -100 for the prompt part to exclude it from loss calculation
            labels = tokens.input_ids.clone()
            labels[:, :prompt_length] = -100  # Don't predict the prompt
            
            # Ensure we have the EOS token at the end
            needs_eos = False
            if tokens.input_ids[0, -1] != tokenizer.eos_token_id:
                needs_eos = True
                logger.info("Adding missing EOS token to training sequence")
                
                # Add EOS token if not present
                input_ids_with_eos = torch.cat([
                    tokens.input_ids, 
                    torch.tensor([[tokenizer.eos_token_id]]).to(model.device)
                ], dim=1)
                
                attention_mask_with_eos = torch.cat([
                    tokens.attention_mask,
                    torch.ones((1, 1), dtype=torch.long).to(model.device)
                ], dim=1)
                
                labels_with_eos = torch.cat([
                    labels,
                    torch.tensor([[tokenizer.eos_token_id]]).to(model.device)  # EOS should be predicted
                ], dim=1)
                
                tokens.input_ids = input_ids_with_eos
                tokens.attention_mask = attention_mask_with_eos
                labels = labels_with_eos
            
            # Double-check the labels have been properly set up
            # EOS token should be predicted (not -100)
            logger.info(f"Input length: {tokens.input_ids.shape[1]}, Prompt length: {prompt_length}")
            logger.info(f"Number of tokens being predicted: {(labels != -100).sum().item()}")
            if needs_eos:
                logger.info(f"Last token is EOS: {tokens.input_ids[0, -1] == tokenizer.eos_token_id}")
                logger.info(f"Last label is EOS: {labels[0, -1] == tokenizer.eos_token_id}")
            
            # Forward pass with properly masked labels
            outputs = model(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # Optimization step
            if optimizer is None:
                optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
                logger.info("Created new SFT optimizer (not using persistent one)")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Log SFT training results
            logger.info(f"SFT training completed successfully. Loss: {loss.item():.4f}")
            
            return {
                "loss": loss.item(),
                "success": True,
                "question": question,
                "target_answer": clean_answer
            }
            
        except torch.cuda.OutOfMemoryError as e:
            # Specifically catch CUDA out of memory errors
            torch.cuda.empty_cache()  # Try to free up memory
            logger.warning(f"CUDA out of memory error during SFT training: {e}")
            logger.warning("Skipping this SFT example due to memory constraints")
            
            return {
                "loss": None,
                "success": False,
                "error": "CUDA out of memory",
                "question": question,
                "target_answer": clean_answer
            }
            
    except Exception as e:
        logger.error(f"Error during SFT training: {e}")
        return {
            "loss": None,
            "success": False,
            "error": str(e),
            "question": question,
            "target_answer": clean_answer if 'clean_answer' in locals() else answer
        }

def extract_clean_answer(answer_text):
    """
    Extract clean answer with reasoning and answer tags from text
    
    Args:
        answer_text: Original text containing the answer
        
    Returns:
        Clean formatted answer with proper tags
    """
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", answer_text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
    
    if reasoning_match and answer_match:
        reasoning = reasoning_match.group(1).strip()
        answer = answer_match.group(1).strip()
        return f"<reasoning>{reasoning}</reasoning>\n<answer>{answer}</answer>"
    
    return answer_text  # Return original if extraction fails

def improvement_cycle(model, tokenizer, question, original_answer, expected_answer, feedback_provider, args, last_winning_answer=None, logger=None, cycle_name="Initial", is_bucket_problem=False):
    """
    Run an improvement cycle with feedback and DPO
    
    Args:
        model: Model to train
        tokenizer: Tokenizer for the model
        question: The math problem question
        original_answer: The initial answer to improve
        expected_answer: The expected correct answer
        feedback_provider: FeedbackProvider instance
        args: Command line arguments
        logger: Optional logger
        cycle_name: Name of the improvement cycle (e.g., "Initial", "Verification")
        is_bucket_problem: Whether this is a problem from the bucket list
        
    Returns:
        Dictionary with results of improvement cycle
    """
    logger = logger or logging.getLogger('dpo_training')
    
    logger.info(f"=== Starting {cycle_name} Improvement Cycle ===")
    logger.info(f"Problem type: {'Bucket list' if is_bucket_problem else 'Standard dataset'}")
    
    # Format question with instructions about required format
    formatted_question = question
    
    improved_answers = []
    improved_extracted = []
    improved_format_correct = []
    improved_answer_correct = []
    feedback = None
    dpo_results = []
    
    # Get initial feedback using the appropriate model based on problem type
    logger.info(f"Getting feedback for {cycle_name} cycle...")
    feedback = feedback_provider.get_feedback(question, original_answer, expected_answer, last_winning_answer, is_bucket_problem)
    logger.info(f"Feedback: {feedback}...")
    
    # Try to improve answer with feedback up to max_attempts times
    for attempt in range(args.improvement_max_attempts):
        logger.info(f"{cycle_name} improvement attempt {attempt+1}/{args.improvement_max_attempts}")
        
        # Generate improved answer based on feedback
        prev_answer = improved_answers[-1] if improved_answers else original_answer
        
        # Choose which model to use for generating improvements
        improved_answer = feedback_provider.generate_improved_answer(
                question, 
                prev_answer, 
                feedback,
                expected_answer,
                is_bucket_problem
            )
        ## Clean the improved answer before checking
        improved_answer = extract_clean_answer(improved_answer)
        
        # Extract answer and check correctness
        extracted_answer = extract_answer_from_text(improved_answer)
        format_correct = check_format_correctness(improved_answer)
        answer_correct = calculate_answer_correctness(extracted_answer, expected_answer, 'symbolic' in args.dataset_source, logger)
        
        # Add to results
        improved_answers.append(improved_answer)
        improved_extracted.append(extracted_answer)
        improved_format_correct.append(format_correct)
        improved_answer_correct.append(answer_correct)
        
        logger.info(f"{cycle_name} attempt {attempt+1} - Format correct: {format_correct}, Answer correct: {answer_correct}")
        
        # Modified condition: Only perform DPO if the format is correct AND the improved answer is better
        # This ensures we don't train on examples with incorrect formatting
        if format_correct and answer_correct: 
            logger.info(f"{cycle_name} cycle: Improved answer has correct format and is better than original. Performing DPO...")

            # Perform DPO between improved and original answers
            dpo_result = perform_dpo_training(
                model=model,
                tokenizer=tokenizer,
                formatted_question=formatted_question,
                better_answer=improved_answer,
                worse_answer=original_answer,
                beta=args.beta,
                learning_rate=args.dpo_learning_rate,
                optimizer=feedback_provider.dpo_optimizer,
                logger=logger
            )
            dpo_results.append(dpo_result)
            
            logger.info(f"{cycle_name} cycle DPO result: Success={dpo_result['success']}, Loss={dpo_result.get('loss')}")
            logger.info(f"{cycle_name} cycle found a correct answer with correct format. Stopping improvement cycle.")
            break
        else:
            logger.info(f"{cycle_name} cycle: Improved answer is not complete. Skipping DPO.")
            logger.info("--- IMPROVED ANSWER ---")
            logger.info(improved_answers[-1])
            logger.info("--- END OF IMPROVED ANSWER ---")
        
        feedback = feedback_provider.get_feedback(question, improved_answer, expected_answer, is_bucket_problem)
        logger.info(f"New feedback: {feedback}...")
    
    logger.info(f"=== Completed {cycle_name} Improvement Cycle ===")
    logger.info(f"Total attempts: {len(improved_answers)}, DPO operations: {len(dpo_results)}")
    
    # Log summary of improvement cycle results
    correct_formats = sum(improved_format_correct)
    correct_answers = sum(improved_answer_correct)
    perfect_solutions = sum(1 for fc, ac in zip(improved_format_correct, improved_answer_correct) if fc and ac)
    
    logger.info(f"{cycle_name} cycle summary: {correct_formats}/{len(improved_answers)} correct formats, "
               f"{correct_answers}/{len(improved_answers)} correct answers, "
               f"{perfect_solutions}/{len(improved_answers)} perfect solutions")
    
    return {
        "improved_answers": improved_answers,
        "improved_extracted": improved_extracted, 
        "improved_format_correct": improved_format_correct,
        "improved_answer_correct": improved_answer_correct,
        "dpo_results": dpo_results,
        "final_feedback": feedback
    }

def run_dpo_step(model, tokenizer, problem, feedback_provider, args, logger=None, is_bucket_problem=False):
    """
    Run a single DPO step for one problem
    
    Args:
        model: Model being trained
        tokenizer: Tokenizer for the model
        problem: Problem to work on
        feedback_provider: Feedback provider object
        args: Command line arguments
        logger: Logger instance
        is_bucket_problem: Whether this is a problem from the bucket list
        
    Returns:
        Dictionary with results and metrics
    """
    logger = logger or logging.getLogger('dpo_training')
    
    logger.info("==============================")
    logger.info(f"STARTING NEW DPO PROBLEM ({'BUCKET LIST' if is_bucket_problem else 'STANDARD'})")
    logger.info("==============================")
    
    # Format the question with explicit instructions about the response format
    question = problem["question"]
    formatted_question = question
    expected_answer = problem["answer"]
    
    logger.info(f"Working on problem: {question}")
    logger.info(f"Expected answer: {expected_answer}")
    
    # Generate initial answer
    original_answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        question=formatted_question,
        logger=logger
    )

    logger.info(f"Original answer: {original_answer}")
    
    # Extract answer and check correctness
    original_extracted = extract_answer_from_text(original_answer)
    original_answer_correct = calculate_answer_correctness(original_extracted, expected_answer, "symbolic" in args.dataset_source)
    original_format_correct = check_format_correctness(original_answer)
    
    logger.info(f"Original extracted answer: {original_extracted}")
    logger.info(f"Format correct: {original_format_correct}")
    logger.info(f"Answer correct: {original_answer_correct}")
    
    # If answer is already correct AND format is correct, no need to improve
    if original_answer_correct and original_format_correct:
        logger.info("Original answer and format are correct. Skipping improvement.")
        return {
            "question": question,
            "expected_answer": expected_answer,
            "original_answer": original_answer,
            "original_extracted": original_extracted,
            "original_format_correct": original_format_correct,
            "original_answer_correct": original_answer_correct,
            "improved_answers": [],
            "improved_extracted": [],
            "improved_format_correct": [],
            "improved_answer_correct": [],
            "feedback": "No feedback needed (correct answer and format)",
            "dpo_performed": False,
            "sft_performed": False,
            "sft_loss": None,
            "attempts": 0,
            "is_correct": True,
            "format_correct": True,
            "combined_score": 1.0,
            "verification_results": [],
            "dpo_pairs": [],  # Track all DPO pairs
            "verification_attempts": 0,
            "first_time_correct": original_answer_correct,
            "winning_answer": original_answer
        }
    
    # Keep track of the most recent successful DPO's winning answer
    last_winning_answer = None

    # Run improvement cycle to get better answers
    improvement_results = improvement_cycle(
        model=model,
        tokenizer=tokenizer,
        question=question,
        original_answer=original_answer,
        expected_answer=expected_answer,
        feedback_provider=feedback_provider,
        args=args,
        last_winning_answer=last_winning_answer,
        logger=logger,
        cycle_name="Initial",
        is_bucket_problem=is_bucket_problem
    )
    
    # Extract results from improvement cycle
    improved_answers = improvement_results["improved_answers"]
    improved_extracted = improvement_results["improved_extracted"]
    improved_format_correct = improvement_results["improved_format_correct"]
    improved_answer_correct = improvement_results["improved_answer_correct"]
    feedback = improvement_results["final_feedback"]

    # Track DPO results
    initial_dpo_results = improvement_results["dpo_results"]
    dpo_pairs = []
    
    # Extract DPO pairs from initial improvement and track last winning answer
    for dpo_result in initial_dpo_results:
        if dpo_result.get("success", False):
            dpo_pairs.append({
                "phase": "Initial",
                "winning": dpo_result.get("winning_answer"),
                "losing": dpo_result.get("losing_answer"),
                "loss": dpo_result.get("loss")
            })
            last_winning_answer = dpo_result.get("winning_answer")
    
    dpo_performed = len(initial_dpo_results) > 0
    dpo_loss = None if not dpo_performed else initial_dpo_results[-1].get("loss")
    
    # Start with initial correctness
    final_answer_correct = False
    final_format_correct = False
    
    # Find the best improved answer, if any (for metrics, not for SFT)
    best_improved_index = -1
    
    # First look for an answer with both correct format and answer
    for i, (is_format_correct, is_answer_correct) in enumerate(zip(improved_format_correct, improved_answer_correct)):
        if is_format_correct and is_answer_correct:
            best_improved_index = i
            break
    
    # If none found, look for one with at least correct answer
    if best_improved_index == -1:
        for i, is_answer_correct in enumerate(improved_answer_correct):
            if is_answer_correct:
                best_improved_index = i
                break
    
    # If still none found, look for one with at least correct format
    if best_improved_index == -1:
        for i, is_format_correct in enumerate(improved_format_correct):
            if is_format_correct:
                best_improved_index = i
                break
    
    # Log the best improved answer if found
    if best_improved_index != -1:
        logger.info(f"Best improved answer found at attempt {best_improved_index+1}:")
        logger.info(f"Format correct: {improved_format_correct[best_improved_index]}")
        logger.info(f"Answer correct: {improved_answer_correct[best_improved_index]}")
    else:
        logger.info("No improvement found over original answer.")
    
    # After DPO training, verify if the model has improved
    verification_results = []
    verification_attempts = 0
    last_verification_format_correct = False
    last_verification_answer_correct = False
    
    logger.info("==============================")
    logger.info("STARTING VERIFICATION PHASE")
    logger.info("==============================")
    
    new_answer = original_answer
    # Proceed with verification steps
    for attempt in range(args.verification_max_attempts):
        verification_attempts += 1
        logger.info(f"Verification attempt {attempt+1}/{args.verification_max_attempts}")
        
        new_answer = generate_answer(
            model=model,
            tokenizer=tokenizer,
            question=formatted_question,
            logger=logger
        )
        
        # Extract answer and check correctness
        extracted_answer = extract_answer_from_text(new_answer)
        answer_correct = calculate_answer_correctness(extracted_answer, expected_answer, "symbolic" in args.dataset_source)
        format_correct = check_format_correctness(new_answer)
        
        # Update last verification result
        last_verification_format_correct = format_correct
        last_verification_answer_correct = answer_correct
        
        # Record verification result
        verification_result = {
            "attempt": attempt + 1,
            "answer": new_answer,
            "extracted_answer": extracted_answer,
            "format_correct": format_correct,
            "answer_correct": answer_correct,
            "dpo_performed": False,
            "dpo_results": []
        }
        
        logger.info(f"Verification {attempt+1} - Answer: {extracted_answer}")
        logger.info(f"Verification {attempt+1} - Format correct: {format_correct}")
        logger.info(f"Verification {attempt+1} - Answer correct: {answer_correct}")
        
        # If we got both format and answer correct, we're done
        if format_correct and answer_correct:
            logger.info("Model successfully learned to solve the problem correctly!")
            verification_results.append(verification_result)

            # Apply SFT if enabled AND we have a winning answer from the most recent successful DPO
            sft_performed = False
            sft_loss = None

            if args.use_sft > 0 and dpo_performed and last_winning_answer is not None:
                logger.info("==============================")
                logger.info("PERFORMING SFT FINAL")
                logger.info("==============================")
                
                sft_performed1 = False
                sft_loss1 = 0
                
                # Determine learning rate based on verification results
                sft_lr = args.sft_learning_rate
                logger.info(f"Using standard learning rate ({sft_lr}) because verification succeeded")
                
                if args.use_sft == 1 or args.use_sft >= 3:
                    logger.info("STEP 1: SFT ON LAST DPO'S WINNING ANSWER")
                    sft_result = perform_sft_training(
                        model=model,
                        tokenizer=tokenizer,
                        question=formatted_question,
                        answer=last_winning_answer,  # Using the most recent successful DPO's winning answer
                        learning_rate=sft_lr,  # Use adjusted learning rate
                        optimizer=feedback_provider.sft_optimizer,
                        logger=logger
                    )
                    
                    sft_performed1 = sft_result.get("success", False)
                    sft_loss1 = sft_result.get("loss")

                    logger.info(f"SFT performed: {sft_performed1}, Loss: {sft_loss1}")

                sft_performed2 = False
                sft_loss2 = 0
                if args.use_sft >= 2:
                    logger.info("STEP 2: SFT ON LAST CORRECT VERIFICATION ANSWER")
                    sft_result = perform_sft_training(
                        model=model,
                        tokenizer=tokenizer,
                        question=formatted_question,
                        answer=new_answer,  # Using the last verification answer
                        learning_rate=args.sft_learning_rate * 0.1,
                        optimizer=feedback_provider.sft_optimizer,
                        logger=logger
                    )

                    sft_performed2 = sft_result.get("success", False)
                    sft_loss2 = sft_result.get("loss")
                    
                    logger.info(f"SFT performed: {sft_performed2}, Loss: {sft_loss2}")

                if sft_loss1 is None:
                    sft_loss1 = 0

                if sft_loss2 is None:
                    sft_loss2 = 0

                sft_performed = sft_performed1 or sft_performed2
                sft_loss = sft_loss1 + sft_loss2

                if sft_loss1 is None and sft_loss2 is None:
                    sft_performed = False
                
            elif args.use_sft:
                if last_winning_answer is None:
                    logger.info("Skipping SFT because no successful DPO was performed")

            break

        # If we still haven't succeeded, run another improvement cycle with this answer
        if attempt < args.verification_max_attempts - 1 and (not format_correct or not answer_correct):
            verification_improvement = improvement_cycle(
            model=model,
            tokenizer=tokenizer,
            question=question,
            original_answer=new_answer,
            expected_answer=expected_answer,
            feedback_provider=feedback_provider,
            args=args,
            last_winning_answer=last_winning_answer,
            logger=logger,
            cycle_name=f"Verification-{attempt+1}",
            is_bucket_problem=is_bucket_problem
        )
            
            # Update verification result with DPO details
            verification_result["dpo_results"] = verification_improvement["dpo_results"]
            verification_result["dpo_performed"] = len(verification_improvement["dpo_results"]) > 0
            dpo_performed = verification_result["dpo_performed"]
            
            # Extract DPO pairs from verification and track last winning answer
            for dpo_result in verification_improvement["dpo_results"]:
                if dpo_result.get("success", False):
                    dpo_pairs.append({
                        "phase": f"Verification-{attempt+1}",
                        "winning": dpo_result.get("winning_answer"),
                        "losing": dpo_result.get("losing_answer"),
                        "loss": dpo_result.get("loss")
                    })
                    # Update the most recent successful DPO's winning answer
                    last_winning_answer = dpo_result.get("winning_answer")
        
        verification_results.append(verification_result)

        # Apply SFT if enabled AND we have a winning answer from the most recent successful DPO
        sft_performed = False
        sft_loss = None

        if args.use_sft > 0 and dpo_performed and last_winning_answer is not None:
            logger.info("==============================")
            logger.info("PERFORMING SFT")
            logger.info("==============================")
            
            sft_performed1 = False
            sft_loss1 = 0
            
            # Determine learning rate based on verification results
            sft_lr = args.sft_learning_rate * 0.01
            logger.info(f"Using reduced learning rate ({sft_lr}) because verification failed")
            
            if args.use_sft == 1 or args.use_sft >= 3:
                logger.info("STEP 1: SFT ON LAST DPO'S WINNING ANSWER")
                sft_result = perform_sft_training(
                    model=model,
                    tokenizer=tokenizer,
                    question=formatted_question,
                    answer=last_winning_answer,  # Using the most recent successful DPO's winning answer
                    learning_rate=sft_lr,  # Use adjusted learning rate
                    optimizer=feedback_provider.sft_optimizer,
                    logger=logger
                )
                
                sft_performed1 = sft_result.get("success", False)
                sft_loss1 = sft_result.get("loss")

                logger.info(f"SFT performed: {sft_performed1}, Loss: {sft_loss1}")

            sft_performed2 = False
            sft_loss2 = 0

            if sft_loss1 is None:
                    sft_loss1 = 0

            if sft_loss2 is None:
                sft_loss2 = 0

            sft_performed = sft_performed1 or sft_performed2
            sft_loss = sft_loss1 + sft_loss2

            if sft_loss1 is None and sft_loss2 is None:
                sft_performed = False
            
        elif args.use_sft:
            if last_winning_answer is None:
                logger.info("Skipping SFT because no successful DPO was performed")
    
    # Use the last verification results to determine final correctness
    if verification_attempts > 0:
        final_answer_correct = last_verification_answer_correct
        final_format_correct = last_verification_format_correct
    else:
        # If no verification was done, use the best from improvement phase
        if best_improved_index != -1:
            final_answer_correct = improved_answer_correct[best_improved_index]
            final_format_correct = improved_format_correct[best_improved_index]
    
    # Calculate combined score
    combined_score = 0.0
    if final_answer_correct and final_format_correct:
        combined_score = 1.0
    elif final_answer_correct or final_format_correct:
        combined_score = 0.5
    
    # Log final summary
    logger.info("==============================")
    logger.info("DPO PROBLEM SUMMARY")
    logger.info("==============================")
    logger.info(f"Total DPO pairs: {len(dpo_pairs)}")
    logger.info(f"Final answer correct: {final_answer_correct}")
    logger.info(f"Final format correct: {final_format_correct}")
    logger.info(f"Combined score: {combined_score}")
    logger.info(f"Verification attempts: {verification_attempts}")
    logger.info(f"SFT performed: {sft_performed}")
    
    # Return results
    return {
        "question": question,
        "expected_answer": expected_answer,
        "original_answer": original_answer,
        "original_extracted": original_extracted,
        "original_format_correct": original_format_correct,
        "original_answer_correct": original_answer_correct,
        "improved_answers": improved_answers,
        "improved_extracted": improved_extracted,
        "improved_format_correct": improved_format_correct,
        "improved_answer_correct": improved_answer_correct,
        "feedback": feedback,
        "dpo_performed": dpo_performed,
        "dpo_loss": dpo_loss,
        "sft_performed": sft_performed,
        "sft_loss": sft_loss,
        "attempts": len(improved_answers),
        "is_correct": final_answer_correct,
        "format_correct": final_format_correct,
        "combined_score": combined_score,
        "verification_results": verification_results,
        "dpo_pairs": dpo_pairs,
        "verification_attempts": verification_attempts,
        "first_time_correct": original_answer_correct,
        "winning_answer": last_winning_answer  # Return the last winning answer from DPO
    }