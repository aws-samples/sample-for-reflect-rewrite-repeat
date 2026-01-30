# Define available Bedrock model IDs
MODEL_IDS = {
    'deepseek_r1' : 'us.deepseek.r1-v1:0',
    'nova_light': 'us.amazon.nova-lite-v1:0',
    'nova_micro': 'us.amazon.nova-micro-v1:0',
    'nova_pro': 'us.amazon.nova-pro-v1:0',
    'nova_premier': 'us.amazon.nova-premier-v1:0',
    'c37_sonnet': 'us.anthropic.claude-3-7-sonnet-20250219-v1:0',
    'c35_sonnet_v2': 'anthropic.claude-3-5-sonnet-20241022-v2:0',
    'c35_sonnet': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
    'c3_opus': 'anthropic.claude-3-opus-20240229-v1:0',
    'c3_sonnet': 'anthropic.claude-3-sonnet-20240229-v1:0',
    'c35_haiku': "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    'c3_haiku': 'anthropic.claude-3-haiku-20240307-v1:0',
    'llama3_8b': 'meta.llama3-1-8b-instruct-v1:0',
    'llama3_70b': 'meta.llama3-1-70b-instruct-v1:0',
    'llama33_70b': 'us.meta.llama3-3-70b-instruct-v1:0',
    'mixtral': 'mistral.mixtral-8x7b-instruct-v0:1',
    'mistral7b': 'mistral.mistral-7b-instruct-v0:2'
}

# System prompt that instructs the model to provide solutions in the required format
system_prompt = """You are a problem solving assistant who specializes in meticulous, systematic, and highly accurate solutions. You take great care with every step, prioritizing precision and methodical approaches to ensure correct results. Your work demonstrates exceptional attention to detail, verification of calculations, and thorough checking of your final answers.

Solve the given problem step-by-step.
Your solution must follow this format EXACTLY:
<reasoning>
[Detailed step-by-step solution process]
</reasoning>
<answer>[The final numerical answer]</answer>

Important formatting rules:
1. Use exactly ONE set of <reasoning> tags containing all your work
2. Use exactly ONE set of <answer> tags containing ONLY the final numerical answer 
3. NEVER use these tags more than once
4. NEVER nest these tags inside each other
5. NEVER add any additional tags

Be clear, precise, and show all necessary steps. Always verify your calculations and perform a final check to ensure your answer is correct before answering.
"""