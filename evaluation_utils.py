import math
import re

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
        answer_text = answer_match.group(1).strip()
        # Extract numerical value with commas
        number_match = re.search(r'-?[\d,]+\.?\d*', answer_text)
        if number_match:
            try:
                # Remove commas for conversion to float
                clean_number = number_match.group(0).replace(',', '')
                value = float(clean_number)
                
                # Check for infinity or NaN values
                if math.isinf(value) or math.isnan(value):
                    return None
                return value
            except ValueError:
                return None
    
    return None

def extract_last_number(text):
    """
    Extract the last number mentioned in the text.
    Used as a fallback when formal answer extraction fails.
    
    Args:
        text: Text to extract the last number from
        
    Returns:
        Last number found in the text as float or None if no number found
    """
    # Find all numbers in the text (including those with commas)
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    
    # If numbers are found, return the last one
    if numbers:
        try:
            # Remove commas for conversion to float
            clean_number = numbers[-1].replace(',', '')
            value = float(clean_number)
            
            # Check for infinity or NaN values
            if math.isinf(value) or math.isnan(value):
                return None
            return value
        except ValueError:
            return None
    
    return None

def extract_answer_from_gsm8k(text):
    """
    Extract answer from GSM8K answer format (#### followed by number)
    
    Args:
        text: GSM8K answer text
        
    Returns:
        Extracted answer as float or None if no answer found
    """
    # Look for the pattern #### followed by a number
    answer_match = re.search(r"####\s*(-?\d+\.?\d*)", text)
    if answer_match:
        try:
            return float(answer_match.group(1))
        except ValueError:
            return None
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
    
    # Extract content between tags and check if it's non-empty
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    
    if not reasoning_match or not answer_match:
        return False
    
    reasoning = reasoning_match.group(1).strip()
    answer = answer_match.group(1).strip()
    
    # Both tags must have non-empty content
    return len(reasoning) > 0 and len(answer) > 0

def evaluate_model_response(model_response, target_answer):
    # Check format correctness
    is_format_correct = check_format_correctness(model_response)
    
    # Extract answer
    extracted_answer = extract_answer_from_text(model_response)
    
    # Extract last number as fallback
    last_number_answer = extract_last_number(model_response)
    
    # Check correctness
    is_correct = False
    last_number_correct = False
    is_valid = extracted_answer is not None
    
    if target_answer is not None:
        # Convert target answer to float if it's a string representation of a number
        if isinstance(target_answer, str):
            target_answer = float(target_answer)
        else:
            target_answer = target_answer
        
        if isinstance(target_answer, (int, float)) and is_valid and isinstance(extracted_answer, (int, float)):
            is_correct = abs(extracted_answer - target_answer) < 1e-5
            print("is_correct", is_correct)

        elif isinstance(target_answer, str) and isinstance(extracted_answer, str):
            is_correct = extracted_answer.strip() == target_answer.strip()
            
        if isinstance(target_answer, (int, float)) and last_number_answer is not None and isinstance(last_number_answer, (int, float)):
            last_number_correct = abs(last_number_answer - target_answer) < 1e-5
            print("last_number_correct", last_number_correct)

        elif isinstance(target_answer, str) and isinstance(last_number_answer, str):
            last_number_correct = last_number_answer.strip() == target_answer.strip()

    return {
        "model_response": model_response,
        "extracted_answer": extracted_answer,
        "last_number_answer": last_number_answer,
        "format_correct": is_format_correct,
        "answer_correct": is_correct,
        "last_number_correct": last_number_correct,
        "is_valid": is_valid,
    }
