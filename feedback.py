import logging
import time
import random
from config import MODEL_IDS
import torch
from transformers import GenerationConfig
from config import system_prompt

# Class for handling feedback via Bedrock API
# Modify FeedbackProvider class to handle self-feedback
class FeedbackProvider:
    """
    Get feedback on model responses using AWS Bedrock models or self-feedback
    """
    def __init__(self, bedrock_client, model, tokenizer, 
             feedback_provider="bedrock", improvement_generator="bedrock",
             basic_model_key='c3_haiku', advanced_model_key=None, 
             max_retries=3, dpo_optimizer=None, sft_optimizer=None, logger=None):
        """
        Initialize the feedback provider with basic and advanced models
        
        Args:
            bedrock_client: AWS Bedrock client
            model: Target model for self-feedback
            tokenizer: Tokenizer for the target model
            feedback_provider: Which provider to use ("bedrock" or "self")
            basic_model_key: Key for the basic model ID in MODEL_IDS (used for standard problems)
            advanced_model_key: Key for the advanced model ID in MODEL_IDS (used for bucket list problems)
            max_retries: Maximum retries for API calls
            logger: Logger instance
            dpo_optimizer: Optimizer for DPO training (persistent)
            sft_optimizer: Optimizer for SFT training (persistent)
            logger: Logger instance
        """
        self.bedrock = bedrock_client
        self.model = model
        self.tokenizer = tokenizer
        self.feedback_provider = feedback_provider
        self.improvement_generator = improvement_generator
        self.basic_model_id = MODEL_IDS.get(basic_model_key, MODEL_IDS['c3_haiku'])
        self.advanced_model_key = advanced_model_key
        self.advanced_model_id = MODEL_IDS.get(advanced_model_key, self.basic_model_id) if advanced_model_key else None
        self.max_retries = max_retries
        self.logger = logger or logging.getLogger('dpo_training')
        self.dpo_optimizer = dpo_optimizer  # Store the DPO optimizer
        self.sft_optimizer = sft_optimizer  # Store the SFT optimizer

        # Log setup information
        if self.feedback_provider == "bedrock":
            self.logger.info(f"Using Bedrock for feedback with basic model: {basic_model_key} ({self.basic_model_id})")
            if advanced_model_key:
                self.logger.info(f"Using advanced Bedrock model for bucket list problems: {advanced_model_key} ({self.advanced_model_id})")
            else:
                self.logger.info("No advanced model specified, will use basic model for all problems")
        else:
            self.logger.info("Using target model for self-feedback")
        
        self.logger.info(f"Max retries for API calls: {self.max_retries}")
        
        # Current active model (will be switched between basic and advanced)
        self.current_model_id = self.basic_model_id
        self.current_model_key = basic_model_key
    
    def set_model_for_problem(self, is_bucket_problem=False):
        """
        Set the active model based on whether this is a bucket list problem
        
        Args:
            is_bucket_problem: Whether this is a problem from the bucket list
        """
        if self.feedback_provider == "bedrock" and is_bucket_problem and self.advanced_model_id:
            self.current_model_id = self.advanced_model_id
            self.current_model_key = self.advanced_model_key
            self.logger.debug(f"Switched to advanced model: {self.advanced_model_key} for bucket list problem")
        else:
            self.current_model_id = self.basic_model_id
            self.current_model_key = self.advanced_model_key
            self.logger.debug(f"Using basic model: {self.current_model_key} for standard problem")
    
    def get_feedback(self, question, answer, expected_answer, last_winning_answer=None, is_bucket_problem=False):
        """
        Get feedback on a response using appropriate model
        
        Args:
            question: The original math question
            answer: The model's answer to get feedback on
            expected_answer: The expected answer for reference
            is_bucket_problem: Whether this is a problem from the bucket list
            
        Returns:
            Feedback on the reasoning and solution
        """
        # Set the appropriate model based on problem type
        self.set_model_for_problem(is_bucket_problem)
        
        if self.feedback_provider == "bedrock":
            # Create common feedback prompt
            prompt = self._create_feedback_prompt(question, answer, expected_answer)
            return self._get_bedrock_response(prompt, is_bucket_problem)
        else:
            prompt = self._create_feedback_prompt_self(question, answer, expected_answer, last_winning_answer)
            return self._get_self_model_response(prompt, is_system_prompt=True)
    
    def _create_feedback_prompt(self, question, answer, expected_answer):
        """
        Create a prompt for feedback
        
        Args:
            question: The original math question
            answer: The model's answer to get feedback on
            expected_answer: The expected answer for reference
            
        Returns:
            Prompt for getting feedback
        """
        return f"""Review and give constructive feedback to this proposed solution based on the following:

QUESTION: {question}
PROPOSED SOLUTION: {answer}
EXPECTED ANSWER: {expected_answer}

Check if the solution:
1. Has exactly one set of <reasoning>...</reasoning> tags containing all work
2. Has exactly one set of <answer>...</answer> tags with only the final answer
3. Follows proper format (begins with <reasoning> and ends with </answer>)
4. Provides clear, logical steps that lead to the correct final answer
5. DO NOT provide any revised solution.

Provide concise feedback ONCE focusing on these specific sections (only include sections where issues exist):
1. Formatting issues:
The solution should have proper formatting structure with:
a. All reasoning contained within <reasoning>...</reasoning> tags
b. Only the final numerical answer within <answer>...</answer> tags
c. No text outside these tags

2. Reasoning issues:
a. Check if all steps are logical and lead to the correct result
b. Look for calculation errors in arithmetic operations
c. Verify that the approach is appropriate for this type of problem
d. Identify any conceptual misunderstandings
e. FOR EVERY MULTIPLICATION STEP (CRITICAL): ALWAYS re-verify, break down each step of multiplication
   Example: 38 * 7 = (30 * 7) + (8 * 7) = 210 + 56 = 266
   Example: 124 * 36 = (124 * 30) + (124 * 6) = 3,720 + 744 = 4,464
   
Example: "Your reasoning contains an error when working with negative numbers. When subtracting a negative number, the operation becomes addition."

3. Answer Issues (CRITICAL):
a. MOST IMPORTANT: Check if the final answer matches the expected answer
b. The answer should be concise and not include unnecessary explanations
c. Only the final result should appear in the answer tags

Example: "Your final answer of -736 doesn't match the expected answer of 326. Review your calculations for errors."
"""
    
    def _create_feedback_prompt_self(self, question, answer, expected_answer, last_winning_answer):
        """
        Create a prompt for feedback
        
        Args:
            question: The original math question
            answer: The model's answer to get feedback on
            expected_answer: The expected answer for reference
            last_winning_answer: The last correct solution for reference
            
        Returns:
            Prompt for getting feedback
        """
        return f"""Review and give constructive feedback to this proposed solution based on the following:

QUESTION: {question}
PROPOSED SOLUTION: {answer}
EXPECTED ANSWER: {expected_answer}
LAST CORRECTED SOLUTION: {last_winning_answer}

Check if the solution:
1. Has exactly one set of <reasoning>...</reasoning> tags containing all work
2. Has exactly one set of <answer>...</answer> tags with only the final answer
3. Follows proper format (begins with <reasoning> and ends with </answer>)
4. Provides clear, logical steps that lead to the correct final answer
5. DO NOT provide any revised solution.

Compare the proposed solution with the last corrected solution to identify areas needing improvement.

Provide concise feedback ONCE focusing on these specific sections (only include sections where issues exist):
1. Formatting issues:
The solution should have proper formatting structure with:
a. All reasoning contained within <reasoning>...</reasoning> tags
b. Only the final numerical answer within <answer>...</answer> tags
c. No text outside these tags

2. Reasoning issues:
a. Check if all steps are logical and lead to the correct result
b. Look for calculation errors in arithmetic operations
c. Verify that the approach is appropriate for this type of problem
d. Identify any conceptual misunderstandings
e. FOR EVERY MULTIPLICATION STEP (CRITICAL): ALWAYS re-verify, break down each step of multiplication
   Example: 38 * 7 = (30 * 7) + (8 * 7) = 210 + 56 = 266
   Example: 124 * 36 = (124 * 30) + (124 * 6) = 3,720 + 744 = 4,464
f. If the last corrected solution exists, note if it had a more effective approach

Example: "Your reasoning contains an error when working with negative numbers. When subtracting a negative number, the operation becomes addition."
Example: "In your multiplication calculation, check the digit placement carefully. The product of 38 * 7 should be 266, not 246."

3. Answer Issues (CRITICAL):
a. MOST IMPORTANT: Check if the final answer matches the expected answer
b. The answer should be concise and not include unnecessary explanations
c. Only the final result should appear in the answer tags
d. Verify if the form of the answer matches the expected format

Example: "Your final answer of -736 doesn't match the expected answer of 326. Review your calculations for errors."
"""
    
    def _create_improvement_prompt(self, question, original_answer, feedback, expected_answer):
        """
        Create a prompt for generating improved answers
        
        Args:
            question: The original math question
            original_answer: The original model's answer
            feedback: The feedback provided on the original answer
            expected_answer: The expected answer for reference
            
        Returns:
            Prompt for generating improved answer
        """
        return f"""Please improve the following math solution based on the provided feedback:

QUESTION: {question}

ORIGINAL SOLUTION: {original_answer}

FEEDBACK: {feedback}

CORRECT ANSWER: {expected_answer}

Your improved solution should:
1. Use the exact tags <reasoning>...</reasoning> and <answer>...</answer>
2. Follow the ORIGINAL SOLUTION's format and structure exactly until the error point
3. At the error point, create a natural "aha moment" transition:
   a. Don't explicitly mention the "original approach" or that you're correcting something
   b. Write as if you're genuinely discovering the insight during your problem-solving process
   c. Use transition phrases like "Wait, I need to reconsider..." or "Actually, I just realized..."
4. After the transition, work through the problem step-by-step:
   a. Show your reasoning and calculations as you go, don't skip steps
   b. Don't jump straight to the expected answer - derive it naturally
   c. Include any false starts or corrections that would happen in real problem solving
5. Maintain the same level of detail and explanation style throughout
6. Present your final answer within the <answer> tags
7. Write as if you're solving this problem for the first time without prior knowledge of the answer

IMPORTANT: Don't act like you already know the answer! Show authentic problem-solving where you genuinely work through the logic to arrive at the correct solution.

GOOD TRANSITION AND REASONING EXAMPLES:
- "Wait, I need to reconsider this calculation. Let me try a different approach... [show work step-by-step]"
- "Actually, I just realized that I need to account for... Let me recalculate this... [show detailed calculations]"
- "Hold on, I should be more careful with this part. Let me double-check... [work through the logic]"

GOOD CONCLUDING EXAMPLES:
- "Well, I think I've checked through all the potential mistakes now. This approach seems correct."
- "Hmm, I keep arriving at the same answer after verifying my work, so I'll go with this result."
- "Let me double-check... yes, everything adds up correctly. This must be the answer."
- "I was unsure at first, but after careful verification, I can see this solution works."
- "Wait, let me verify this last calculation... OK, it's correct. That gives us our final answer."

IMPROVED SOLUTION:
"""
    
    def _get_bedrock_response(self, prompt, is_bucket_problem=False):
        """
        Get response from Bedrock models
        
        Args:
            prompt: The prompt to send to Bedrock
            is_bucket_problem: Whether this is a bucket list problem (for model selection)
            
        Returns:
            Response from Bedrock
        """
        # Try up to max_retries times if there's an error
        for retry_count in range(self.max_retries):
            try:
                # Add a small delay to prevent rate limiting
                time.sleep(random.uniform(0.1, 0.5))
                
                model_type = "advanced" if is_bucket_problem and self.advanced_model_id else "basic"
                self.logger.debug(f"Getting response using {model_type} model: {self.current_model_key}")
                
                response = self.bedrock.converse(
                    modelId=self.current_model_id if model_type =="basic" else self.advanced_model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    inferenceConfig={
                        "maxTokens": 2000,
                        "temperature": 0.99,
                    }
                )
                
                # Extract response text
                result = response['output']['message']['content'][0]['text']
                return result
                        
            except Exception as e:
                if retry_count < self.max_retries - 1:
                    self.logger.warning(f"Error from Bedrock (attempt {retry_count+1}/{self.max_retries}): {e}. Retrying...")
                    # time.sleep(2.0)  # Exponential backoff
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed. Error: {e}")
                    return "Unable to provide response due to technical issues."
        
        # This should never be reached
        return "Unable to provide response due to technical issues."
    
    def _get_self_model_response(self, prompt, is_system_prompt=False, check_tags=False):
        """
        Get response from the target model
        
        Args:
            prompt: The prompt to send to the model
            is_system_prompt: Whether to include system prompt for chat
            check_tags: Whether to check for required tags in response
            
        Returns:
            Response from the target model
        """
        # Format as chat
        if is_system_prompt:
            # system_msg = "You are a helpful tutor providing feedback on student work. Be specific about formatting issues and reasoning errors, but don't solve the problem completely. Your goal is to guide the student to find and correct their own mistakes."
            chat_messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        else:
            chat_messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
        
        # Check if tokenizer has a chat template
        if hasattr(self.tokenizer, "apply_chat_template"):
            chat_prompt = self.tokenizer.apply_chat_template(
                chat_messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template
            if is_system_prompt:
                # system_msg = "You are a helpful tutor providing feedback on student work. Be specific about formatting issues and reasoning errors, but don't solve the problem completely. Your goal is to guide the student to find and correct their own mistakes."
                chat_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>"
            else:
                chat_prompt = f"<|user|>\n{prompt}\n<|assistant|>"
        
        # Try up to max_retries times if there's an error
        for retry_count in range(self.max_retries):
            try:
                # Add a small delay to prevent rate limiting
                # time.sleep(random.uniform(0.1, 0.5))
                
                self.logger.debug("Getting response using target model")
                
                model_inputs = self.tokenizer(
                    [chat_prompt],
                    return_tensors="pt",
                    padding=True,
                    return_attention_mask=True,
                ).to(self.model.device)
                
                generation_config = GenerationConfig(
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.99,# if is_system_prompt else 0.7,
                    max_new_tokens=2000,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **model_inputs, 
                        generation_config=generation_config
                    )
                
                # Extract only the generated part, not the prompt
                prompt_length = model_inputs["input_ids"].shape[1]
                result = self.tokenizer.decode(
                    output_ids[0, prompt_length:], 
                    skip_special_tokens=True
                )
                
                # Check if the response has the required tags
                if check_tags:
                    required_tags = ["<reasoning>", "</reasoning>", "<answer>", "</answer>"]
                    missing_tags = [tag for tag in required_tags if tag not in result]

                    if missing_tags and set(missing_tags) == {"<reasoning>", "</reasoning>"} and "<answer>" in result and "</answer>" in result:
                        answer_start = result.find("<answer>")
                        answer_end = result.find("</answer>") + len("</answer>")
                        answer_part = result[answer_start:answer_end]
                        
                        remaining_text = result[:answer_start].strip()
                        modified_result = f"<reasoning>{remaining_text}</reasoning>\n{answer_part}"
                        result = modified_result
                        missing_tags = [tag for tag in required_tags if tag not in result]

                    if missing_tags and retry_count < self.max_retries - 1:
                        self.logger.warning(f"Missing tags: {missing_tags}. Retrying...")
                        continue
                
                return result
                        
            except Exception as e:
                if retry_count < self.max_retries - 1:
                    self.logger.warning(f"Error from target model (attempt {retry_count+1}/{self.max_retries}): {e}. Retrying...")
                    time.sleep(min(2 ** retry_count, 5))  # Exponential backoff
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed. Error: {e}")
                    return "Unable to provide response due to technical issues."
        
        # This should never be reached
        return "Unable to provide response due to technical issues."
    
    def generate_improved_answer(self, question, original_answer, feedback, expected_answer, is_bucket_problem=False):
        """
        Generate improved answer based on feedback using appropriate model
        
        Args:
            question: The original math question
            original_answer: The original model's answer
            feedback: The feedback provided on the original answer
            expected_answer: The expected answer for reference
            is_bucket_problem: Whether this is a problem from the bucket list
            
        Returns:
            Improved answer with better reasoning
        """
        # Set the appropriate model based on problem type
        self.set_model_for_problem(is_bucket_problem)
        # Create common improvement prompt
        prompt = self._create_improvement_prompt(question, original_answer, feedback, expected_answer)
        
        if self.improvement_generator == "bedrock":
            return self._get_bedrock_response(prompt, is_bucket_problem)
        else:
            return self._get_self_model_response(prompt, is_system_prompt=False, check_tags=True)
