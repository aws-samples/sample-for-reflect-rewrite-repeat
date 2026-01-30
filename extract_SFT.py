import re
import json
from config import system_prompt

experiment_name = "20250414-171529-Qwen2.5-Math-7B-math_train_4digit-bedrock-bedrock-nova_premier-nova_premier-lora-r16"

def clean_question(raw_question):
    # Remove timestamp and log level patterns
    timestamp_pattern = r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - \w+ - '
    cleaned = re.sub(timestamp_pattern, '', raw_question)
    # Remove any leading/trailing whitespace
    return cleaned.strip()

def extract_answer_content(raw_answer):
    # Extract content including <reasoning> and </answer> tags
    pattern = r'(<reasoning>.*?</answer>)'
    match = re.search(pattern, raw_answer, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def extract_sft_data(log_file, output_file):

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split the content by SFT TRAINING sections
    sft_sections = content.split("=== SFT TRAINING ===")[1:]
    print(f"Total SFT sections found: {len(sft_sections)}")
    
    idx = 0
    
    for section in sft_sections:
        try:
            # Extract question
            question_match = re.search(r"--- QUESTION ---\n(.*?)\n", 
                                     section, re.DOTALL)
            if question_match:
                question = clean_question(question_match.group(1))
            else:
                continue
            
            # Extract target answer
            answer_match = re.search(r"--- TARGET ANSWER ---\n(.*?)\n---", 
                                   section, re.DOTALL)
            if answer_match:
                raw_answer = answer_match.group(1)
                clean_answer = extract_answer_content(raw_answer)
            else:
                continue
            
            # Skip if either question or answer is empty
            if not question or not clean_answer:
                print(f"Skipping section due to empty question or answer: {question} - {clean_answer}")
                continue
            
            conversation = {
                "id": idx,
                "question": question,
                "answer": clean_answer,
                "experiment_name": experiment_name
            }
             
            # Save to JSONL file
            with open(output_file, 'a', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False)
                f.write('\n')
            print(f"Processed section {idx}: {question} - {clean_answer}")
            # Increment index for unique ID
            idx += 1
            
        except Exception as e:
            print(f"Error processing section: {e}")
            continue
    
    return conversation 

def main():
    # Input log file
    log_file = "20250414-171529-Qwen2.5-Math-7B-math_train_4digit-bedrock-bedrock-nova_premier-nova_premier-lora-r16.log"
    
    # Output JSON file
    output_file = "datasets/sft_dataset.jsonl"
    
    # Extract and compile SFT data
    sft_dataset = extract_sft_data(log_file, output_file)
    
    # Print statistics
    print(f"Total examples extracted: {len(sft_dataset)}")
    
    # Print first example as sample
    if sft_dataset:
        print("\nSample entry:")
        print(json.dumps(sft_dataset[0], indent=2))


if __name__ == "__main__":
    main()
