import argparse
from typing import Any, List, Dict, Optional, Union
from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import torch
from transformers import (
    GenerationConfig,
    StoppingCriteriaList,
    StoppingCriteria,
)
from model_utils import create_model_instance
from evaluation_utils import (
    extract_answer_from_text,
    check_format_correctness,
    extract_last_number,
)

app = FastAPI()

def parse_args():
    """
    Parse command line arguments for evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate math problem performance across model checkpoints')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for generation')
    parser.add_argument('--bits', type=int, choices=[0, 4, 8], default=0, 
                        help='Quantization bits (0 for no quantization, 4 or 8 for quantized)')
    parser.add_argument('--checkpoint_path', type=str, default="", help='checkpoint_path')
    parser.add_argument('--model_family', type=str, choices=['llama', 'qwen', 'auto'], default='auto', 
                        help='Model family (llama, qwen, or auto for automatic detection)')
    parser.add_argument('--worker_id', type=int, default=0, help='Worker ID')
    parser.add_argument('--api_hostname', type=str, default="127.0.0.1", help='Hostname for API server')
    parser.add_argument('--api_port', type=int, default=8000, help='Port for API server')
                        
    return parser.parse_args()


# FastAPI models for request/response
class GenerationRequest(BaseModel):
    system_prompt: str
    question: str
    target_answer: Optional[Union[float, str]] = None
    worker_id: int = 0
    temperature: float = 0.1
    top_p: float = 0.9
    max_new_tokens: int = 1024
    question_id: Optional[int] = None
    question_metadata: Optional[Dict[str, Any]] = None

class GenerationResponse(BaseModel):
    question: str
    model_response: str
    extracted_answer: Optional[Union[float, str]] = None
    last_number_answer: Optional[Union[float, str]] = None
    target_answer: Optional[Union[float, str]] = None
    format_correct: bool
    answer_correct: bool
    last_number_correct: bool
    is_valid: bool
    error: Optional[str] = None
    question_id: Optional[int] = None
    question_metadata: Optional[Dict[str, Any]] = None

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


@app.on_event("startup")
async def load_models():
    """Initialize models on startup"""
    # Set random seed for reproducibility
    torch.manual_seed(args.seed + args.worker_id)  # Different seed per worker
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + args.worker_id)
    
    try:
        print(f"Loading model instance {args.worker_id} on port {args.api_port}")
        model, tokenizer = create_model_instance(args.checkpoint_path, args.model_family, args.bits)
        app.model = model
        app.tokenizer = tokenizer
        print(f"Model (worker_id={args.worker_id}) loaded successfully on port {args.api_port}")
    except Exception as e:
        print(f"Error loading model instance {args.worker_id}: {e}")
        raise e

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok", "worker_id": args.worker_id}

@app.post("/generate", response_model=GenerationResponse)
async def generate_response(request: GenerationRequest):
    """Generate a response for a math question"""
    try:
        # Prepare prompt using chat template
        if hasattr(app.tokenizer, "chat_template") and app.tokenizer.chat_template:
            # Use chat template if available
            chat_messages = [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.question}
            ]
            prompt = app.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for app.tokenizers without chat template
            prompt = f"<|system|>\n{request.system_prompt}\n<|user|>\n{request.question}\n<|assistant|>"
        
        # Tokenize
        inputs = app.tokenizer(prompt, return_tensors="pt", padding=True).to(app.model.device)
        
        # Generation configuration
        generation_config = GenerationConfig(
            do_sample=True,
            top_p=request.top_p,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            pad_token_id=app.tokenizer.pad_token_id,
            eos_token_id=app.tokenizer.eos_token_id,
        )

        stop_token_ids = [app.tokenizer.eos_token_id] 
        stop_on_tokens = StopOnTokens(stop_token_ids)

        stop_texts = ["</answer>"] 
        stop_on_text = StopOnTextAfter(app.tokenizer, stop_texts, inputs.input_ids.shape[1])

        stopping_criteria = StoppingCriteriaList([stop_on_tokens, stop_on_text])

        # Generate response
        with torch.no_grad():
            outputs = app.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )
        
        # Extract generated text
        input_length = inputs.input_ids.shape[1]
        generated_text = app.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Check format correctness
        is_format_correct = check_format_correctness(generated_text)
        
        # Extract answer
        extracted_answer = extract_answer_from_text(generated_text)
        
        # Extract last number as fallback
        last_number_answer = extract_last_number(generated_text)
        
        # Check correctness
        is_correct = False
        last_number_correct = False
        is_valid = extracted_answer is not None
        
        if request.target_answer is not None:
            # Convert target answer to float if it's a string representation of a number
            target_answer = None
            try:
                if isinstance(request.target_answer, str):
                    target_answer = float(request.target_answer)
                else:
                    target_answer = request.target_answer
            except (ValueError, TypeError):
                # Non-numeric target answer
                target_answer = request.target_answer
            
            if isinstance(target_answer, (int, float)) and is_valid and isinstance(extracted_answer, (int, float)):
                is_correct = abs(extracted_answer - target_answer) < 1e-5
                
            if isinstance(target_answer, (int, float)) and last_number_answer is not None and isinstance(last_number_answer, (int, float)):
                last_number_correct = abs(last_number_answer - target_answer) < 1e-5
                
            # Handle string comparison for non-numeric answers
            elif isinstance(target_answer, str) and isinstance(extracted_answer, str):
                is_correct = extracted_answer.strip() == target_answer.strip()
                
            if isinstance(target_answer, str) and isinstance(last_number_answer, str):
                last_number_correct = last_number_answer.strip() == target_answer.strip()
        
        return GenerationResponse(
            question=request.question,
            model_response=generated_text,
            extracted_answer=extracted_answer,
            last_number_answer=last_number_answer,
            target_answer=request.target_answer,
            format_correct=is_format_correct,
            answer_correct=is_correct,
            last_number_correct=last_number_correct,
            is_valid=is_valid,
            question_id=request.question_id,
            question_metadata=request.question_metadata
        )
        
    except Exception as e:
        return GenerationResponse(
            question=request.question,
            model_response=f"ERROR: {str(e)}",
            target_answer=request.target_answer,
            format_correct=False,
            answer_correct=False,
            last_number_correct=False,
            is_valid=False,
            error=str(e),
            question_id=request.question_id,
            question_metadata=request.question_metadata
        )


if __name__ == "__main__":
    args = parse_args()
    
    # Start the server
    uvicorn.run(app, host=args.api_hostname, port=args.api_port)
