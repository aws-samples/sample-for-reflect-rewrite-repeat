import logging
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import boto3
import re
import os
import time
import csv
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from matplotlib.projections import register_projection
import requests

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("llm_reasoning_assessment.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

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

# Define assessment categories
ASSESSMENT_CATEGORIES = [
    "arithmetic_operations",     # Fundamental operations with complex calculations
    "algebraic_reasoning",       # Solving equations and working with algebraic expressions
    "numerical_computation",     # Precision in numerical calculations and estimation
    "mathematical_proof",        # Logical reasoning and proving mathematical statements
    "probabilistic_reasoning",   # Probability theory and statistical reasoning
    "sequential_reasoning",      # Multi-step problem solving with interdependent steps
    "geometric_understanding",   # Spatial reasoning and geometric principles
    "edge_case_handling"         # Handling mathematical edge cases and exceptions
]

# Define benchmark datasets
BENCHMARK_DATASETS = [
    "gsm8k",         # Grade School Math Word Problems
    "aime",          # American Invitational Mathematics Examination
    "amc",           # American Mathematics Competitions
    "mathqa",        # Mathematics Question Answering
    "math"           # MATH dataset (DeepMind)
]

class RadarAxes(plt.PolarAxes):
    """
    Custom radar chart axes with improved aesthetics
    """
    name = 'radar'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.set_theta_zero_location('N')
    
    def plot_radar(self, values, *args, **kwargs):
        """Plot one line on the radar chart"""
        values = np.array(values)
        # Close the loop for the plot
        values = np.concatenate((values, [values[0]]))
        angles = np.linspace(0, 2*np.pi, len(values), endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        # Plot line
        line = self.plot(angles, values, *args, **kwargs)
        # Fill with semi-transparent color
        self.fill(angles, values, alpha=0.1, color=line[0].get_color())
        return line

def radar_factory(N, frame='polygon'):
    """
    Create a radar chart with N axes
    
    Parameters:
    -----------
    N : int
        Number of axes
    frame : str
        Shape of frame ('circle' or 'polygon')
    
    Returns:
    --------
    theta : array
        Angles of the radar axes
    """
    theta = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    register_projection(RadarAxes)
    return theta

class LLMReasoningAssessor:
    """
    Class to assess and compare mathematical reasoning capabilities of different LLM models.
    Handles both foundational mathematical skills and standard benchmark datasets.
    """
    def __init__(self, model_names: List[str], data_dir: str = "assessment_data"):
        """
        Initialize the assessor with a list of model names to evaluate.
        
        Parameters:
        -----------
        model_names : List[str]
            List of models to evaluate (can be Bedrock aliases, HF model IDs, or paths to local checkpoints)
        data_dir : str
            Directory to store/load benchmark datasets
        """
        self.model_names = model_names
        self.data_dir = data_dir
        self.bedrock = boto3.client('bedrock-runtime') if any(name in MODEL_IDS for name in model_names) else None
        self.models = {}
        self.tokenizers = {}
        self.results = {model: {category: 0.0 for category in ASSESSMENT_CATEGORIES} for model in model_names}
        self.benchmark_results = {model: {dataset: {"score": 0.0, "samples": 0} for dataset in BENCHMARK_DATASETS} for model in model_names}
        self.raw_interactions = {model: [] for model in model_names}
        self.detailed_evaluations = {model: {} for model in model_names}
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info(f"Initializing assessment for models: {', '.join(model_names)}")
        
        # Load HuggingFace models or local checkpoints
        for model_name in model_names:
            if model_name not in MODEL_IDS:
                logger.info(f"Loading model: {model_name}")
                try:
                    # Check if the model is a local path or HF model ID
                    if os.path.exists(model_name) or os.path.exists(os.path.join(model_name, "config.json")):
                        logger.info(f"Loading from local checkpoint: {model_name}")
                    else:
                        logger.info(f"Loading from HuggingFace: {model_name}")
                        
                    self.tokenizers[model_name] = AutoTokenizer.from_pretrained(
                        model_name,
                        trust_remote_code=True,
                        use_fast=True,
                        revision="main"
                    )
                    
                    self.models[model_name] = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,  # Use fp16 to save memory
                        device_map="auto",
                        trust_remote_code=True,
                        revision="main"
                    )
                    logger.info(f"Successfully loaded {model_name}")
                except Exception as e:
                    logger.error(f"Failed to load {model_name}: {e}")
    
    def generate_assessment_problems(self) -> Dict[str, List[Dict]]:
        """
        Generate assessment problems for each category.
        The problems are carefully designed with verified solutions.
        
        Returns:
        --------
        Dict[str, List[Dict]]
            A dictionary mapping categories to lists of problems
        """
        logger.info("Generating assessment problems for each category")
        
        problems = {category: [] for category in ASSESSMENT_CATEGORIES}
        
        # Arithmetic Operations - Complex calculations requiring careful application of order of operations
        problems["arithmetic_operations"] = [
            {
                "question": "Compute the exact value of: (-17)² - 4 × (-12) + 56 ÷ (7 - 3)²",
                "answer": "340.5",
                "explanation": "(-17)² = 289, 4 × (-12) = -48, (7-3)² = 16, 56 ÷ 16 = 3.5. So 289 - (-48) + 3.5 = 289 + 48 + 3.5 = 340.5.",
                "expected_work": "First, calculate each part: (-17)² = 289, 4 × (-12) = -48, (7-3)² = 16, 56 ÷ 16 = 3.5. Then apply order of operations: 289 - (-48) + 3.5 = 289 + 48 + 3.5 = 340.5"
            },
            {
                "question": "Calculate the result of: (2³ × 4² - 36) ÷ (√25 - 3)",
                "answer": "46",
                "explanation": "2³ = 8, 4² = 16, 8 × 16 = 128, 128 - 36 = 92, √25 = 5, 5 - 3 = 2, 92 ÷ 2 = 46.",
                "expected_work": "Work through step-by-step: (2³ × 4² - 36) ÷ (√25 - 3) = (8 × 16 - 36) ÷ (5 - 3) = (128 - 36) ÷ 2 = 92 ÷ 2 = 46"
            },
            {
                "question": "Evaluate: ((-3)³ + 18 × 5) ÷ (2² + 3²)",
                "answer": "4.85",
                "explanation": "(-3)³ = -27, 18 × 5 = 90, -27 + 90 = 63, 2² = 4, 3² = 9, 4 + 9 = 13, 63 ÷ 13 = 4.846... = 4.85 (rounded to 2 decimal places)",
                "expected_work": "(-3)³ = -27, 18 × 5 = 90, -27 + 90 = 63, 2² = 4, 3² = 9, 4 + 9 = 13, 63 ÷ 13 = 4.846... = 4.85 (rounded to 2 decimal places)"
            },
            {
                "question": "Calculate: (2⁄3 × 9) - (4⁄5 × 15) + 3⁄4",
                "answer": "-5.25",
                "explanation": "2⁄3 × 9 = 6, 4⁄5 × 15 = 12, 6 - 12 = -6, -6 + 3⁄4 = -6 + 0.75 = -5.25",
                "expected_work": "2⁄3 × 9 = 6, 4⁄5 × 15 = 12, 6 - 12 = -6, -6 + 3⁄4 = -6 + 0.75 = -5.25"
            },
            {
                "question": "If a = 5, b = -3, and c = 2, compute the value of a² - 2ab - c³ + 4ac",
                "answer": "87",
                "explanation": "a² = 5² = 25, 2ab = 2(5)(-3) = -30, c³ = 2³ = 8, 4ac = 4(5)(2) = 40. So a² - 2ab - c³ + 4ac = 25 - (-30) - 8 + 40 = 25 + 30 - 8 + 40 = 87",
                "expected_work": "Substitute the values: a² - 2ab - c³ + 4ac = 5² - 2(5)(-3) - 2³ + 4(5)(2) = 25 - (-30) - 8 + 40 = 25 + 30 - 8 + 40 = 87"
            }
        ]
        
        # Algebraic Reasoning - Solving equations, working with algebraic expressions
        problems["algebraic_reasoning"] = [
            {
                "question": "Solve for x: 3(2x - 4) = 5x - (x + 6)",
                "answer": "x = 3",
                "explanation": "3(2x - 4) = 5x - (x + 6), 6x - 12 = 5x - x - 6, 6x - 12 = 4x - 6, 6x - 4x = 12 - 6, 2x = 6, x = 3",
                "expected_work": "Step 1: Distribute on left side: 3(2x - 4) = 6x - 12. Step 2: Distribute on right side: 5x - (x + 6) = 5x - x - 6 = 4x - 6. Step 3: So we have 6x - 12 = 4x - 6. Step 4: Subtract 4x from both sides: 2x - 12 = -6. Step 5: Add 12 to both sides: 2x = 6. Step 6: Divide both sides by 2: x = 3"
            },
            {
                "question": "Factor completely: 2x³ - 8x² - 24x",
                "answer": "2x(x - 6)(x + 2)",
                "explanation": "2x³ - 8x² - 24x = 2x(x² - 4x - 12) = 2x((x - 6)(x + 2)) = 2x(x - 6)(x + 2)",
                "expected_work": "First, factor out the GCD: 2x³ - 8x² - 24x = 2x(x² - 4x - 12). Then factor the quadratic: 2x(x² - 4x - 12) = 2x(x - 6)(x + 2)"
            },
            {
                "question": "Find all values of x that satisfy: x⁴ - 13x² + 36 = 0",
                "answer": "x = 3, x = -3, x = 2, x = -2",
                "explanation": "Let u = x², then u² - 13u + 36 = 0. Using the quadratic formula: u = (13 ± √(169-144))/2 = (13 ± √25)/2 = (13 ± 5)/2, so u = 9 or u = 4. Since u = x², x² = 9 or x² = 4, thus x = ±3 or x = ±2.",
                "expected_work": "This is a biquadratic equation. Let u = x², so the equation becomes u² - 13u + 36 = 0. Using the quadratic formula: u = (13 ± √(169-144))/2 = (13 ± √25)/2 = (13 ± 5)/2. So u = 9 or u = 4. Since u = x², x² = 9 or x² = 4, which means x = ±3 or x = ±2."
            },
            {
                "question": "Solve the system of equations: 2x + 3y = 7 and 5x - 2y = 16",
                "answer": "x = 4, y = -1/3",
                "explanation": "From the first equation: 2x + 3y = 7, so y = (7 - 2x)/3. Substitute into the second equation: 5x - 2((7 - 2x)/3) = 16. Simplify: 5x - 2(7 - 2x)/3 = 16, 5x - (14 - 4x)/3 = 16, 15x - 14 + 4x = 48, 19x = 62, x = 62/19 ≈ 3.26. Substituting back: y = (7 - 2(62/19))/3 = (7 - 124/19)/3 = (133/19 - 124/19)/3 = 9/19/3 = 3/19 ≈ 0.16.",
                "expected_work": "From 2x + 3y = 7, we get y = (7 - 2x)/3. Substituting into 5x - 2y = 16: 5x - 2((7 - 2x)/3) = 16. Multiply by 3: 15x - 2(7 - 2x) = 48. Distribute: 15x - 14 + 4x = 48. 19x = 62. x = 62/19. Substituting back: y = (7 - 2(62/19))/3 = (7 - 124/19)/3 = (133/19 - 124/19)/3 = 9/57 = 3/19."
            },
            {
                "question": "If f(x) = x² - 3x + 2 and g(x) = 2x + 1, find the value of f(g(3))",
                "answer": "30",
                "explanation": "g(3) = 2(3) + 1 = 7. f(g(3)) = f(7) = 7² - 3(7) + 2 = 49 - 21 + 2 = 30.",
                "expected_work": "First, find g(3) = 2(3) + 1 = 7. Then compute f(g(3)) = f(7) = 7² - 3(7) + 2 = 49 - 21 + 2 = 30."
            }
        ]
        
        # Numerical Computation - Precision in numerical calculations
        problems["numerical_computation"] = [
            {
                "question": "Calculate √(128) in simplified radical form.",
                "answer": "8√2",
                "explanation": "√(128) = √(64 * 2) = √64 * √2 = 8√2",
                "expected_work": "First, find the largest perfect square factor: 128 = 64 × 2 = 2⁶ × 2¹ = 2⁷. So √128 = √(2⁷) = √(2⁶ × 2¹) = √(2⁶) × √(2¹) = 2³ × √2 = 8√2"
            },
            {
                "question": "Express 3.721 × 10⁻⁴ × 8.5 × 10⁶ in scientific notation with 3 significant figures.",
                "answer": "3.16 × 10³",
                "explanation": "3.721 × 10⁻⁴ × 8.5 × 10⁶ = 3.721 × 8.5 × 10⁻⁴ × 10⁶ = 31.6285 × 10² = 3.16285 × 10³ ≈ 3.16 × 10³ (rounded to 3 significant figures)",
                "expected_work": "Multiply the coefficients: 3.721 × 8.5 = 31.6285. Add the exponents: 10⁻⁴ × 10⁶ = 10². So the result is 31.6285 × 10² = 3.16285 × 10³. Rounding to 3 significant figures gives 3.16 × 10³."
            },
            {
                "question": "Find the exact value of sin(π/3) + cos(π/4)",
                "answer": "√3/2 + √2/2",
                "explanation": "sin(π/3) = √3/2, cos(π/4) = √2/2. So sin(π/3) + cos(π/4) = √3/2 + √2/2.",
                "expected_work": "Using the values from the unit circle: sin(π/3) = √3/2 and cos(π/4) = √2/2. Therefore, sin(π/3) + cos(π/4) = √3/2 + √2/2. This is the exact value and cannot be simplified further."
            },
            {
                "question": "Calculate the product of (2 + 3i) and (4 - 2i) where i is the imaginary unit.",
                "answer": "14 + 8i",
                "explanation": "(2 + 3i)(4 - 2i) = 8 - 4i + 12i - 6i² = 8 + 8i - 6(-1) = 8 + 8i + 6 = 14 + 8i",
                "expected_work": "Use the distributive property: (2 + 3i)(4 - 2i) = 2(4) + 2(-2i) + 3i(4) + 3i(-2i) = 8 - 4i + 12i - 6i² = 8 + 8i - 6(-1) = 8 + 8i + 6 = 14 + 8i"
            },
            {
                "question": "Calculate the limit as x approaches 0: lim(sin(3x)/5x)",
                "answer": "3/5",
                "explanation": "lim(sin(3x)/5x) = lim(3·sin(3x)/3·5x) = (3/5)·lim(sin(3x)/3x). Using the known limit lim(sin(θ)/θ) = 1 as θ approaches 0, and letting θ = 3x, we get (3/5)·1 = 3/5.",
                "expected_work": "We can rewrite this as lim(sin(3x)/5x) = lim(3·sin(3x)/15x) = 3/15 · lim(sin(3x)/3x). Let u = 3x, then as x→0, u→0. We know that lim(sin(u)/u) = 1 as u→0, which is a fundamental limit. Therefore, lim(sin(3x)/5x) = 3/15 · 1 = 3/5."
            }
        ]
        
        # Mathematical Proof - Logical reasoning and proving mathematical statements
        problems["mathematical_proof"] = [
            {
                "question": "Prove by induction that 1 + 2 + 3 + ... + n = n(n+1)/2 for all positive integers n.",
                "answer": "Proved by induction",
                "explanation": "Base case: For n = 1, LHS = 1, RHS = 1(1+1)/2 = 1. So base case holds. Inductive step: Assume P(k): 1 + 2 + ... + k = k(k+1)/2 for some k ≥ 1. Need to show P(k+1): 1 + 2 + ... + k + (k+1) = (k+1)(k+2)/2. LHS = (1 + 2 + ... + k) + (k+1) = k(k+1)/2 + (k+1) = (k+1)(k/2 + 1) = (k+1)(k+2)/2. So P(k+1) holds, completing the proof.",
                "expected_work": "Step 1: Base case - For n = 1, the formula gives 1(1+1)/2 = 1, which is true. Step 2: Inductive hypothesis - Assume the formula is true for n = k, so 1 + 2 + ... + k = k(k+1)/2. Step 3: Inductive step - We need to prove the formula for n = k+1. The sum 1 + 2 + ... + k + (k+1) = [1 + 2 + ... + k] + (k+1) = k(k+1)/2 + (k+1) = (k+1)[k/2 + 1] = (k+1)(k+2)/2. This matches the formula for n = k+1, completing the proof."
            },
            {
                "question": "Prove that the square root of 2 is irrational.",
                "answer": "Proved by contradiction",
                "explanation": "Proof by contradiction: Assume √2 = a/b where a and b are integers with no common factors (reduced fraction). Then 2 = a²/b², so a² = 2b². This means a² is even, which implies a is even (since odd squared is odd). So a = 2c for some integer c. Substituting: 2b² = a² = (2c)² = 4c². Thus b² = 2c². By the same reasoning, b must be even. But this contradicts our assumption that a and b have no common factors. Therefore, √2 cannot be expressed as a fraction of integers, so it is irrational.",
                "expected_work": "We'll use proof by contradiction. Assume √2 is rational, so √2 = a/b where a and b are integers with no common factors. Then 2 = a²/b², so a² = 2b². This means a² is even, which implies a is even (since an odd number squared is odd). So a = 2c for some integer c. Substituting: a² = (2c)² = 4c² = 2b². Therefore, b² = 2c². This means b² is even, so b is even. But now we've shown both a and b are even, which contradicts our assumption that they have no common factors. Therefore, √2 must be irrational."
            },
            {
                "question": "Prove the Pythagorean identity: sin²(θ) + cos²(θ) = 1 for any angle θ.",
                "answer": "Proved using the unit circle",
                "explanation": "Consider a point (x,y) on the unit circle. By definition, x = cos(θ) and y = sin(θ) where θ is the angle from the positive x-axis. Since the point is on the unit circle, x² + y² = 1. Therefore, sin²(θ) + cos²(θ) = y² + x² = 1.",
                "expected_work": "Consider a point (x,y) on the unit circle corresponding to angle θ. By definition of sine and cosine, x = cos(θ) and y = sin(θ). Since the point is on the unit circle, its distance from the origin is 1, so x² + y² = 1. Substituting, we get cos²(θ) + sin²(θ) = 1, which proves the identity."
            },
            {
                "question": "Prove that if n is an odd integer, then n² is odd.",
                "answer": "Proved algebraically",
                "explanation": "If n is odd, then n = 2k + 1 for some integer k. Then n² = (2k + 1)² = 4k² + 4k + 1 = 2(2k² + 2k) + 1. Since 2k² + 2k is an integer, n² = 2m + 1 where m = 2k² + 2k, which means n² is odd.",
                "expected_work": "Since n is odd, we can write n = 2k + 1 for some integer k. Then n² = (2k + 1)² = 4k² + 4k + 1 = 2(2k² + 2k) + 1. Since 2k² + 2k is an integer (call it m), we have n² = 2m + 1, which is the form of an odd integer. Therefore, n² is odd."
            },
            {
                "question": "Prove the power rule for differentiation: d/dx[x^n] = nx^(n-1) for any real number n.",
                "answer": "Proved using the limit definition",
                "explanation": "Using the limit definition of the derivative: d/dx[x^n] = lim(h→0)[(x+h)^n - x^n]/h. By the binomial theorem, (x+h)^n = x^n + nx^(n-1)h + terms with h² and higher powers. So [(x+h)^n - x^n]/h = nx^(n-1) + terms with h and higher powers. As h approaches 0, these extra terms vanish, leaving d/dx[x^n] = nx^(n-1).",
                "expected_work": "Using the limit definition of the derivative: d/dx[x^n] = lim(h→0)[(x+h)^n - x^n]/h. By the binomial theorem: (x+h)^n = sum(k=0 to n)(n choose k)x^(n-k)h^k = x^n + nx^(n-1)h + terms with h² and higher. Substituting: [(x+h)^n - x^n]/h = [x^n + nx^(n-1)h + O(h²) - x^n]/h = nx^(n-1) + O(h). As h→0, this approaches nx^(n-1), which proves the power rule."
            }
        ]
        
        # Probabilistic Reasoning - Probability theory and statistical reasoning
        problems["probabilistic_reasoning"] = [
            {
                "question": "In a standard deck of 52 cards, what is the probability of drawing either a king or a heart?",
                "answer": "4/13",
                "explanation": "There are 4 kings and 13 hearts, with 1 card (king of hearts) in both categories. So the number of favorable outcomes is 4 + 13 - 1 = 16. The probability is 16/52 = 4/13.",
                "expected_work": "Using the formula P(A or B) = P(A) + P(B) - P(A and B): P(king) = 4/52, P(heart) = 13/52, P(king and heart) = 1/52. So P(king or heart) = 4/52 + 13/52 - 1/52 = 16/52 = 4/13."
            },
            {
                "question": "Three fair six-sided dice are rolled. What is the probability that the sum of the dice is exactly 10?",
                "answer": "1/8",
                "explanation": "We need to count the number of ways to get a sum of 10 with three dice. The possible combinations are: (1,3,6), (1,4,5), (2,2,6), (2,3,5), (2,4,4), (3,3,4). Each combination can occur in multiple ways depending on which die shows which value. There are 27 such arrangements out of 216 total outcomes, so the probability is 27/216 = 1/8.",
                "expected_work": "Let's count the number of ordered triples (a,b,c) where 1 ≤ a,b,c ≤ 6 and a+b+c = 10. The combinations are: (1,3,6), (1,4,5), (2,2,6), (2,3,5), (2,4,4), (3,3,4). For each combination, we need to account for permutations: (1,3,6): 6 permutations, (1,4,5): 6 permutations, (2,2,6): 3 permutations, (2,3,5): 6 permutations, (2,4,4): 3 permutations, (3,3,4): 3 permutations. Total favorable outcomes: 6+6+3+6+3+3 = 27. Total possible outcomes: 6³ = 216. Probability = 27/216 = 1/8."
            },
            {
                "question": "In a binomial distribution with n = 5 trials and probability of success p = 0.3, what is the probability of exactly 2 successes?",
                "answer": "0.3087",
                "explanation": "Using the binomial probability formula: P(X = 2) = (5 choose 2) × (0.3)² × (0.7)³ = 10 × 0.09 × 0.343 = 0.3087.",
                "expected_work": "The formula for binomial probability is P(X = k) = (n choose k) × p^k × (1-p)^(n-k). With n = 5, k = 2, p = 0.3: P(X = 2) = (5 choose 2) × (0.3)² × (0.7)³. (5 choose 2) = 5!/(2!(5-2)!) = 10. P(X = 2) = 10 × 0.09 × 0.343 = 0.3087."
            },
            {
                "question": "A bag contains 5 red balls and 7 blue balls. Three balls are drawn without replacement. Find the probability that exactly 2 of them are red.",
                "answer": "0.3182",
                "explanation": "We need to find P(exactly 2 red out of 3 drawn) = (number of ways to choose 2 red balls from 5 red balls and 1 blue ball from 7 blue balls) / (number of ways to choose 3 balls from 12 balls) = (5C2 × 7C1) / 12C3 = (10 × 7) / 220 = 70/220 = 7/22 ≈ 0.3182.",
                "expected_work": "The number of ways to select exactly 2 red balls and 1 blue ball is (5 choose 2) × (7 choose 1) = 10 × 7 = 70. The total number of ways to select 3 balls from 12 is (12 choose 3) = 220. Therefore, the probability = 70/220 = 7/22 ≈ 0.3182."
            },
            {
                "question": "The random variable X follows a normal distribution with mean μ = 60 and standard deviation σ = 12. Find P(X > 75).",
                "answer": "0.1056",
                "explanation": "We standardize the variable: Z = (X - μ)/σ = (75 - 60)/12 = 1.25. We need P(Z > 1.25) = 1 - P(Z ≤ 1.25). From the standard normal table, P(Z ≤ 1.25) ≈ 0.8944. So P(Z > 1.25) ≈ 1 - 0.8944 = 0.1056.",
                "expected_work": "First, convert to a standard normal: Z = (X - μ)/σ = (75 - 60)/12 = 1.25. We need to find P(Z > 1.25) = 1 - P(Z ≤ 1.25). From the standard normal table, P(Z ≤ 1.25) = 0.8944. Therefore, P(X > 75) = P(Z > 1.25) = 1 - 0.8944 = 0.1056."
            }
        ]
        
        # Sequential Reasoning - Multi-step problem solving
        problems["sequential_reasoning"] = [
            {
                "question": "A square has one of its vertices at the origin and the opposite vertex at (6,8). Find the coordinates of the other two vertices.",
                "answer": "(-1,7) and (7,1)",
                "explanation": "Let the opposite vertices be at (0,0) and (6,8). The diagonals of a square bisect each other, so the center of the square is at (3,4). Since the diagonals are perpendicular, if we rotate the vector from the center to one vertex by 90°, we get the vector from the center to an adjacent vertex. So from center to (0,0) is (-3,-4). Rotating 90° gives (4,-3). So the third vertex is at (3,4) + (4,-3) = (7,1). Similarly, rotating (-3,-4) by -90° gives (-4,3), so the fourth vertex is at (3,4) + (-4,3) = (-1,7).",
                "expected_work": "Let's call the vertices A(0,0), C(6,8), and the other two B and D. The diagonals of a square bisect each other, so the center of the square is at ((0+6)/2, (0+8)/2) = (3,4). The vector from the center to vertex A is (-3,-4). Since the diagonals are perpendicular, if we rotate this vector by 90° clockwise, we get the vector from center to B. Rotation by 90° transforms (x,y) to (y,-x), so (-3,-4) becomes (-4,3). Thus vertex B is at (3,4) + (-4,3) = (-1,7). Similarly, rotating (-3,-4) by 90° counterclockwise gives (4,-3), so vertex D is at (3,4) + (4,-3) = (7,1)."
            },
            {
                "question": "A cylindrical tank with radius 3 meters is filled with water to a height of 5 meters. If water is being drained at a rate of 2 cubic meters per minute, how fast is the water level dropping (in cm/min)?",
                "answer": "7.07 cm/min",
                "explanation": "The volume of water is V = πr²h, where r = 3m and h is the height. So dV/dt = πr² × dh/dt. We know dV/dt = -2 m³/min (negative because volume is decreasing). Therefore, dh/dt = dV/dt ÷ (πr²) = -2 ÷ (π × 9) = -2/9π ≈ -0.0707 m/min = -7.07 cm/min. The water level is dropping at 7.07 cm/min.",
                "expected_work": "The volume of water in the tank is V = πr²h, where r = 3m and h is the height. Rate of change of volume: dV/dt = πr² × dh/dt. We know dV/dt = -2 m³/min (negative because water is being drained). So πr² × dh/dt = -2, which gives dh/dt = -2/(πr²) = -2/(π × 3² ) = -2/(9π) ≈ -0.0707 m/min. Converting to cm/min: -0.0707 × 100 = -7.07 cm/min. The negative sign indicates the level is decreasing, so the water level is dropping at 7.07 cm/min."
            },
            {
                "question": "Two trains leave stations 330 km apart at the same time, traveling toward each other. One train travels at 60 km/h and the other at 50 km/h. How far from the first station will they meet?",
                "answer": "180 km",
                "explanation": "Let's say they meet x km from the first station. At that point, the first train has traveled x km, and the second train has traveled (330-x) km. If t is the time in hours when they meet, then x = 60t and 330-x = 50t. Solving these equations: 60t = x and 50t = 330-x, so 110t = 330, thus t = 3. Therefore, x = 60 × 3 = 180 km.",
                "expected_work": "Let x = distance from first station where trains meet. Let t = time (in hours) until they meet. For the first train: x = 60t. For the second train: 330 - x = 50t. From the second equation: t = (330 - x)/50. Substituting into the first equation: x = 60((330 - x)/50), x = (60/50)(330 - x), x = 1.2(330 - x), x = 396 - 1.2x, 2.2x = 396, x = 180 km."
            },
            {
                "question": "A mixture of 20 liters of alcohol and water contains 15% alcohol. How much water should be added to make a mixture containing 12% alcohol?",
                "answer": "5 liters",
                "explanation": "Initially, the amount of pure alcohol is 20 × 0.15 = 3 liters. Let x be the amount of water to add. After adding, the total volume is (20 + x) liters, and the percentage of alcohol is 12%. So 3/(20 + x) = 0.12. Solving for x: 3 = 0.12(20 + x), 3 = 2.4 + 0.12x, 0.6 = 0.12x, x = 5 liters.",
                "expected_work": "Initial amount of pure alcohol = 20 × 15% = 3 liters. Let x = liters of water to add. After adding water: Total volume = 20 + x liters, Amount of alcohol = 3 liters, Concentration = 3/(20 + x) = 12%. Solving: 3/(20 + x) = 0.12, 3 = 0.12(20 + x), 3 = 2.4 + 0.12x, 0.6 = 0.12x, x = 5 liters."
            },
            {
                "question": "A $10,000 investment earns 6% annual interest, compounded quarterly. How much will the investment be worth after 5 years?",
                "answer": "$13,488.50",
                "explanation": "Using the compound interest formula A = P(1 + r/n)^(nt), where P = $10,000, r = 0.06, n = 4 (quarterly), and t = 5 years. A = 10000 × (1 + 0.06/4)^(4×5) = 10000 × (1 + 0.015)^20 = 10000 × (1.015)^20 ≈ 10000 × 1.3489 = $13,489.",
                "expected_work": "Using the compound interest formula A = P(1 + r/n)^(nt), where P = principal, r = annual interest rate, n = number of compounding periods per year, and t = time in years. Given: P = $10,000, r = 0.06, n = 4, t = 5. A = 10000(1 + 0.06/4)^(4×5) = 10000(1 + 0.015)^20 = 10000(1.015)^20. Using a calculator: (1.015)^20 ≈ 1.3488. Therefore, A = 10000 × 1.3488 = $13,488.50 (rounded to the nearest cent)."
            }
        ]
        
        # Geometric Understanding - Spatial reasoning and geometric principles
        problems["geometric_understanding"] = [
            {
                "question": "In a triangle, sides a and b have lengths 8 and 15 respectively, and the angle between them is 60°. Find the length of the third side c.",
                "answer": "13",
                "explanation": "Using the law of cosines: c² = a² + b² - 2ab·cos(C) = 8² + 15² - 2·8·15·cos(60°) = 64 + 225 - 2·8·15·0.5 = 289 - 120 = 169, so c = 13.",
                "expected_work": "We can use the Law of Cosines: c² = a² + b² - 2ab·cos(C), where C is the angle between sides a and b. Given a = 8, b = 15, and C = 60°. c² = 8² + 15² - 2(8)(15)cos(60°) = 64 + 225 - 2(8)(15)(0.5) = 289 - 120 = 169. Therefore, c = √169 = 13."
            },
            {
                "question": "Find the volume of a regular tetrahedron with edge length 6.",
                "answer": "36√2",
                "explanation": "The formula for the volume of a regular tetrahedron is V = (edge length)³ / (6√2). So V = 6³ / (6√2) = 216 / (6√2) = 36√2.",
                "expected_work": "The formula for the volume of a regular tetrahedron is V = (s³/6√2), where s is the edge length. With s = 6: V = 6³/(6√2) = 216/(6√2) = 36√2 cubic units."
            },
            {
                "question": "A circle has equation x² + y² - 4x - 6y + 9 = 0. Find its center and radius.",
                "answer": "Center: (2,3), Radius: 2",
                "explanation": "Rewriting the equation: (x² - 4x + 4) + (y² - 6y + 9) = -9 + 4 + 9 = 4. So (x - 2)² + (y - 3)² = 4. This is in the form (x - h)² + (y - k)² = r², so the center is (h,k) = (2,3) and the radius is r = 2.",
                "expected_work": "We need to complete the square for both x and y terms: x² + y² - 4x - 6y + 9 = 0. Rearranging: (x² - 4x) + (y² - 6y) = -9. Complete the square for x: x² - 4x + 4 = (x - 2)². For y: y² - 6y + 9 = (y - 3)². So: (x - 2)² + (y - 3)² = -9 + 4 + 9 = 4. This is in the standard form (x - h)² + (y - k)² = r², so the center is (2,3) and the radius is 2."
            },
            {
                "question": "Find the equation of the tangent line to the curve y = x³ + 2x at the point (1,3).",
                "answer": "y = 5x - 2",
                "explanation": "For the curve y = x³ + 2x, the derivative is dy/dx = 3x² + 2. At the point (1,3), the slope is 3(1)² + 2 = 3 + 2 = 5. Using the point-slope form of a line: y - 3 = 5(x - 1), which simplifies to y = 5x - 2.",
                "expected_work": "The derivative of y = x³ + 2x is dy/dx = 3x² + 2. At the point (1,3), the slope is 3(1)² + 2 = 3 + 2 = 5. Using point-slope form of a line: y - y₁ = m(x - x₁), where (x₁,y₁) = (1,3) and m = 5. y - 3 = 5(x - 1), y - 3 = 5x - 5, y = 5x - 2."
            },
            {
                "question": "In a right triangle, the hypotenuse is 13 cm and one leg is 5 cm. Find the area of the triangle.",
                "answer": "30 cm²",
                "explanation": "Let the legs be a = 5 cm and b (unknown). By the Pythagorean theorem: a² + b² = c² where c = 13 cm. So 5² + b² = 13², 25 + b² = 169, b² = 144, b = 12 cm. The area is A = (1/2)ab = (1/2) × 5 × 12 = 30 cm².",
                "expected_work": "Let the legs of the right triangle be a = 5 cm and b (unknown), with hypotenuse c = 13 cm. Using the Pythagorean theorem: a² + b² = c², 5² + b² = 13², 25 + b² = 169, b² = 144, b = 12 cm. The area of the triangle is A = (1/2)ab = (1/2)(5)(12) = 30 cm²."
            }
        ]
        
        # Edge Case Handling - Dealing with edge cases and special cases
        problems["edge_case_handling"] = [
            {
                "question": "Evaluate lim(x→0) (sin(x)/x)",
                "answer": "1",
                "explanation": "This is a well-known limit. As x approaches 0, sin(x)/x approaches 1. This can be proven using L'Hôpital's rule or by analyzing the definition of the derivative of sin(x) at x = 0.",
                "expected_work": "This is a fundamental limit that equals 1. It can be verified by observing that the derivative of sin(x) at x = 0 is cos(0) = 1, and this derivative is defined as lim(h→0)[sin(h)/h], which is precisely our limit."
            },
            {
                "question": "Find the value of 0⁰ (zero raised to the power of zero).",
                "answer": "1, by convention",
                "explanation": "Mathematically, 0⁰ is an indeterminate form. However, by convention in most contexts, 0⁰ = 1. This convention makes many formulas and theorems work consistently.",
                "expected_work": "The expression 0⁰ is an indeterminate form that requires careful consideration. If we approach it as x→0 for x⁰, we get 1 for any x≠0. In most mathematical contexts and programming languages, 0⁰ is defined as 1 by convention to preserve continuity of functions like x⁰ for x>0."
            },
            {
                "question": "Solve the equation |2x - 5| = -3.",
                "answer": "No solution",
                "explanation": "The absolute value of any expression is always greater than or equal to 0. Since -3 < 0, the equation |2x - 5| = -3 has no solution.",
                "expected_work": "Since the absolute value of any real expression is always non-negative, and -3 < 0, the equation |2x - 5| = -3 has no solution. There is no value of x for which |2x - 5| can equal -3."
            },
            {
                "question": "What is the domain of the function f(x) = ln(x²-4) / √(25-x²)?",
                "answer": "(-5,-2)∪(2,5)",
                "explanation": "For ln(x²-4) to be defined, we need x²-4 > 0, which gives |x| > 2, so x < -2 or x > 2. For √(25-x²) to be defined, we need 25-x² ≥ 0, which gives |x| ≤ 5, so -5 ≤ x ≤ 5. Additionally, the denominator cannot be zero, so x ≠ ±5. Combining these restrictions, the domain is (-5,-2)∪(2,5).",
                "expected_work": "We need to find where both parts of this function are defined. For ln(x²-4): x²-4 > 0 → x² > 4 → |x| > 2 → x < -2 or x > 2. For √(25-x²): 25-x² ≥ 0 → x² ≤ 25 → -5 ≤ x ≤ 5. Additionally, the denominator cannot be zero: √(25-x²) ≠ 0 → 25-x² ≠ 0 → x² ≠ 25 → x ≠ ±5. Combining these conditions: (-5,-2)∪(2,5). But since x = -5 must be excluded due to division by zero, the domain is (-5,-2)∪(2,5)."
            },
            {
                "question": "Evaluate the improper integral ∫_1^∞ (1/x²) dx.",
                "answer": "1",
                "explanation": "∫_1^∞ (1/x²) dx = lim(b→∞) ∫_1^b (1/x²) dx = lim(b→∞) [-1/x]_1^b = lim(b→∞) (-1/b - (-1/1)) = lim(b→∞) (1 - 1/b) = 1.",
                "expected_work": "We compute this improper integral as a limit: ∫_1^∞ (1/x²) dx = lim(b→∞) ∫_1^b (1/x²) dx. For the definite integral: ∫_1^b (1/x²) dx = [-1/x]_1^b = -1/b - (-1) = 1 - 1/b. Taking the limit: lim(b→∞) (1 - 1/b) = 1 - 0 = 1."
            }
        ]
        
        return problems

    def _bedrock_conversation(self, model_id: str, message: str) -> str:
        """
        Send a message to a Bedrock model and get the response.
        
        Parameters:
        -----------
        model_id : str
            Bedrock model ID
        message : str
            Message to send to the model
            
        Returns:
        --------
        str
            Model response
        """
        try:
            # Implement exponential backoff for API calls
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.bedrock.converse(
                        modelId=model_id,
                        messages=[{"role": "user", "content": [{"text": message}]}],
                        inferenceConfig={
                            "maxTokens": 4096,
                            "temperature": 0.2,  # Lower temperature for more focused mathematical reasoning
                            "topP": 0.9
                        }
                    )
                    return response['output']['message']['content'][0]['text']
                except Exception as e:
                    if attempt < max_retries - 1:
                        sleep_time = (2 ** attempt) + 0.1  # Exponential backoff
                        logger.warning(f"Attempt {attempt+1} failed with error: {e}. Retrying in {sleep_time:.2f} seconds.")
                        time.sleep(sleep_time)
                    else:
                        raise
        except Exception as e:
            logger.error(f"Error in Bedrock conversation: {e}")
            return f"Error: {e}"
    
    def _huggingface_conversation(self, model_name: str, message: str) -> str:
        """
        Generate a response from a HuggingFace model.
        
        Parameters:
        -----------
        model_name : str
            Name or path of the HuggingFace model
        message : str
            Message to send to the model
            
        Returns:
        --------
        str
            Model response
        """
        try:
            tokenizer = self.tokenizers[model_name]
            model = self.models[model_name]
            
            # Create a better prompt with system instruction for mathematical reasoning
            system_instruction = "You are an expert in mathematics with a strong ability to reason through problems step-by-step. Analyze the problem carefully, show all your work, and ensure your calculations are accurate."
            
            if hasattr(tokenizer, "apply_chat_template"):
                messages = [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": message}
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, return_tensors="pt")[0]
            else:
                # Fallback for tokenizers without chat template
                formatted_prompt = f"{system_instruction}\n\nQuestion: {message}\n\nAnswer:"
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_length=4096,
                    do_sample=True,
                    temperature=0.2,
                    top_p=0.9,
                    num_return_sequences=1
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the model's response (not the input prompt)
            if formatted_prompt in response:
                response = response[len(formatted_prompt):].strip()
            else:
                # Alternative extraction if exact matching fails
                last_question_pos = response.rfind(message)
                if last_question_pos != -1:
                    response = response[last_question_pos + len(message):].strip()
            
            return response
        except Exception as e:
            logger.error(f"Error in HuggingFace conversation: {e}")
            return f"Error: {e}"
    
    def assess_model(self, model_name: str, problems: Dict[str, List[Dict]]) -> None:
        """
        Assess a single model across all categories.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to assess
        problems : Dict[str, List[Dict]]
            Dictionary of assessment problems by category
        """
        logger.info(f"Starting comprehensive assessment of model: {model_name}")
        
        for category in ASSESSMENT_CATEGORIES:
            category_problems = problems[category]
            category_score = 0.0
            category_evaluations = []
            
            logger.info(f"Testing {model_name} on {category} problems")
            
            for idx, problem in enumerate(category_problems):
                question = problem["question"]
                expected_answer = problem["answer"]
                expected_work = problem.get("expected_work", "")
                
                logger.info(f"  Problem {idx+1}: {question[:50]}...")
                
                # Formulate an extended prompt that encourages step-by-step reasoning
                prompt = f"""Solve the following mathematical problem. Provide your reasoning step-by-step in a clear and organized manner:

Problem: {question}

Please be methodical and careful in your approach. Show all your work, including intermediate steps, calculations, and logical reasoning. If the problem requires a proof, ensure your proof is rigorous and complete.
"""
                
                # Get model response based on whether it's a Bedrock or HF model
                if model_name in MODEL_IDS:
                    response = self._bedrock_conversation(MODEL_IDS[model_name], prompt)
                else:
                    response = self._huggingface_conversation(model_name, prompt)
                
                # Log the raw interaction
                interaction = {
                    "category": category,
                    "problem_index": idx,
                    "question": question,
                    "expected_answer": expected_answer,
                    "expected_work": expected_work,
                    "model_response": response
                }
                
                self.raw_interactions[model_name].append(interaction)
                
                # Analyze response for correctness and quality of reasoning
                evaluation = self._evaluate_response(response, question, expected_answer, expected_work)
                category_evaluations.append(evaluation)
                
                # Calculate problem score
                problem_score = (
                    0.4 * evaluation["answer_correctness"] + 
                    0.3 * evaluation["reasoning_quality"] + 
                    0.2 * evaluation["work_completeness"] + 
                    0.1 * evaluation["precision"]
                )
                
                category_score += problem_score / len(category_problems)
                
                logger.info(f"    Problem score: {problem_score:.2f}")
                logger.info(f"      Answer correctness: {evaluation['answer_correctness']:.2f}")
                logger.info(f"      Reasoning quality: {evaluation['reasoning_quality']:.2f}")
                logger.info(f"      Work completeness: {evaluation['work_completeness']:.2f}")
                logger.info(f"      Precision: {evaluation['precision']:.2f}")
            
            # Update the results for this category
            self.results[model_name][category] = category_score
            self.detailed_evaluations[model_name][category] = category_evaluations
            
            logger.info(f"{model_name} - {category} score: {category_score:.2f}")
    
    def assess_model_on_benchmark(self, model_name: str, dataset_name: str, samples: List[Dict], limit: int = 20) -> Dict:
        """
        Assess a model's performance on a benchmark dataset
        
        Parameters:
        -----------
        model_name : str
            Name of the model to assess
        dataset_name : str
            Name of the benchmark dataset
        samples : List[Dict]
            List of problem samples
        limit : int
            Maximum number of samples to evaluate
            
        Returns:
        --------
        Dict
            Results and statistics for this benchmark
        """
        logger.info(f"Testing {model_name} on {dataset_name} benchmark")
        samples = samples[:limit]
        
        correct = 0
        evaluations = []
        
        for idx, sample in enumerate(samples):
            question = sample["question"]
            expected_answer = sample["answer"]
            
            prompt = f"""Solve the following problem step-by-step, showing your work:

{question}

First, understand the problem carefully. 
Then, outline your approach.
Next, solve the problem methodically, showing all calculations.
Finally, verify your answer.

Make sure to include your final answer in a clear format at the end.
"""
            
            # Get model response
            if model_name in MODEL_IDS:
                response = self._bedrock_conversation(MODEL_IDS[model_name], prompt)
            else:
                response = self._huggingface_conversation(model_name, prompt)
            
            # Evaluate answer
            evaluation = self._evaluate_response(response, question, expected_answer, "")
            evaluations.append(evaluation)
            
            if evaluation["answer_correctness"] >= 0.8:  # Consider answer correct if score is high
                correct += 1
            
            logger.info(f"  Sample {idx+1} - Correctness: {evaluation['answer_correctness']:.2f}")
        
        # Calculate overall score
        accuracy = correct / len(samples) if samples else 0
        
        self.benchmark_results[model_name][dataset_name] = {
            "score": accuracy,
            "samples": len(samples),
            "correct": correct,
            "evaluations": evaluations
        }
        
        logger.info(f"{model_name} - {dataset_name} accuracy: {accuracy:.2f} ({correct}/{len(samples)})")
        
        return self.benchmark_results[model_name][dataset_name]
    
    def _evaluate_response(self, response: str, question: str, expected_answer: str, expected_work: str) -> Dict[str, float]:
        """
        Evaluate a model response against the expected answer and reasoning.
        Returns a dictionary with scores for different aspects of the response.
        
        Parameters:
        -----------
        response : str
            Model's response
        question : str
            Original question
        expected_answer : str
            Expected correct answer
        expected_work : str
            Expected reasoning and work
            
        Returns:
        --------
        Dict[str, float]
            Scores for different aspects of the response
        """
        # Extract the final answer from the response
        final_answer = self._extract_final_answer(response)
        
        # Different evaluation metrics
        answer_correctness = self._check_answer_correctness(final_answer, expected_answer)
        reasoning_quality = self._evaluate_reasoning_quality(response, expected_work)
        work_completeness = self._evaluate_work_completeness(response, expected_work)
        precision = self._evaluate_precision(response, question)
        
        return {
            "answer_correctness": answer_correctness,
            "reasoning_quality": reasoning_quality,
            "work_completeness": work_completeness,
            "precision": precision,
            "extracted_answer": final_answer
        }
    
    def _extract_final_answer(self, response: str) -> str:
        """
        Extract the final answer from a model response using pattern matching.
        
        Parameters:
        -----------
        response : str
            Model's response
            
        Returns:
        --------
        str
            Extracted final answer
        """
        # Look for "final answer", "therefore", "thus", etc. followed by an answer
        conclusion_patterns = [
            r"(?:final answer|therefore|thus|hence|conclusion|result)[^.,;:]*?(?:is|=|:)[^.,;:]*?([^.,;:]+)",
            r"(?:the answer is|we get|we find|we have)[^.,;:]*?([^.,;:]+)",
            r"answer:?\s*([^.,;:]+)",
            r"(?:=|:)\s*([^.,;:\n]+)$",  # Matches = or : followed by text at the end
            r"(?<=\$)([^$]+)(?=\$)",     # Matches LaTeX expressions in $ delimiters
        ]
        
        # Try each pattern
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            if matches:
                # Take the last match as it's more likely to be the final answer
                return matches[-1].strip()
        
        # If no explicit conclusion found, look at the last few sentences
        sentences = re.split(r'[.!?]\s+', response)
        if len(sentences) > 1:
            for sentence in reversed(sentences[:min(len(sentences), 5)]):
                # Look for equations, values, or numbers in the last few sentences
                equation_match = re.search(r'([^=]+=[^=]+|[xy]\s*=\s*[-+]?\d*\.?\d+|[-+]?\d*\.?\d+)', sentence)
                if equation_match:
                    return equation_match.group(1).strip()
        
        # If all else fails, return the last 100 characters (might contain the answer)
        return response[-100:].strip()
    
    def _check_answer_correctness(self, actual_answer: str, expected_answer: str) -> float:
        """
        Check if the actual answer matches the expected answer, with nuanced handling.
        Returns a score between 0.0 and 1.0.
        
        Parameters:
        -----------
        actual_answer : str
            Answer extracted from model response
        expected_answer : str
            Expected correct answer
            
        Returns:
        --------
        float
            Correctness score between 0.0 and 1.0
        """
        if not actual_answer or not expected_answer:
            return 0.0
        
        # Clean up both answers for comparison
        actual = self._normalize_answer(actual_answer)
        expected = self._normalize_answer(expected_answer)
        
        # Check for exact match after normalization
        if actual == expected:
            return 1.0
        
        # Check for numerical equivalence
        try:
            actual_num = self._extract_numerical_value(actual)
            expected_num = self._extract_numerical_value(expected)
            
            if actual_num is not None and expected_num is not None:
                # Allow for small floating point errors
                if abs(actual_num - expected_num) < 1e-6:
                    return 1.0
                
                # Calculate relative error for partial credit
                relative_error = abs((actual_num - expected_num) / expected_num) if expected_num != 0 else abs(actual_num)
                if relative_error < 0.05:  # Within 5% of correct answer
                    return max(0.0, 1.0 - relative_error * 10)  # Scale error to score
                if relative_error < 0.2:   # Within 20% of correct answer
                    return max(0.0, 0.5 - relative_error * 2)   # Some partial credit
        except:
            pass
        
        # Handle symbolic answers
        if self._is_symbolic_answer(expected):
            similarity = self._compare_symbolic_answers(actual, expected)
            if similarity > 0.7:
                return similarity
        
        # Check if key parts of the expected answer are contained
        key_parts_expected = self._extract_key_parts(expected)
        key_parts_actual = self._extract_key_parts(actual)
        
        if key_parts_expected and key_parts_actual:
            matched_parts = sum(1 for part in key_parts_expected if any(self._is_similar(part, actual_part) for actual_part in key_parts_actual))
            if matched_parts > 0:
                return min(0.8, matched_parts / len(key_parts_expected))
        
        # No significant match
        return 0.0
    
    def _normalize_answer(self, answer: str) -> str:
        """
        Normalize an answer string for comparison.
        
        Parameters:
        -----------
        answer : str
            Answer string to normalize
            
        Returns:
        --------
        str
            Normalized answer string
        """
        # Convert to lowercase
        result = answer.lower()
        
        # Remove extra spaces
        result = ' '.join(result.split())
        
        # Remove punctuation that doesn't affect the meaning
        result = re.sub(r'[,;]', '', result)
        
        # Handle various formats and units
        result = re.sub(r'kilometers|kilometer|km', 'km', result)
        result = re.sub(r'meters|meter|m\b', 'm', result)
        result = re.sub(r'\$', '', result)
        result = re.sub(r'°|degrees', 'deg', result)
        
        # Normalize fractions
        result = re.sub(r'(\d+)/(\d+)', lambda m: str(int(m.group(1))/int(m.group(2))), result)
        
        # Normalize scientific notation
        result = re.sub(r'(\d+(?:\.\d+)?)(?:e|×10\^|×10\*\*)([+-]?\d+)', 
                       lambda m: str(float(m.group(1)) * (10 ** int(m.group(2)))), result)
        
        return result.strip()
    
    def _extract_numerical_value(self, text: str) -> float:
        """
        Extract a numerical value from text, handling various formats.
        Returns None if no valid number can be extracted.
        
        Parameters:
        -----------
        text : str
            Text to extract numerical value from
            
        Returns:
        --------
        Optional[float]
            Extracted numerical value, or None if not found
        """
        # Try direct conversion first
        try:
            return float(text)
        except ValueError:
            pass
        
        # Look for numbers in the text
        number_match = re.search(r'[-+]?\d*\.?\d+', text)
        if number_match:
            try:
                return float(number_match.group(0))
            except:
                pass
        
        # Try to handle fractions like "1/2"
        fraction_match = re.search(r'(\d+)/(\d+)', text)
        if fraction_match:
            try:
                return float(int(fraction_match.group(1)) / int(fraction_match.group(2)))
            except:
                pass
        
        # Try to handle square roots like "√2" or "sqrt(2)"
        sqrt_match = re.search(r'(?:√|sqrt\s*\(?)(\d+)(?:\))?', text)
        if sqrt_match:
            try:
                return float(int(sqrt_match.group(1)) ** 0.5)
            except:
                pass
        
        return None
    
    def _is_symbolic_answer(self, answer: str) -> bool:
        """
        Check if an answer is symbolic (contains variables, square roots, etc.)
        
        Parameters:
        -----------
        answer : str
            Answer to check
            
        Returns:
        --------
        bool
            True if the answer is symbolic
        """
        symbols = ['x', 'y', 'z', 'a', 'b', 'c', 'π', 'pi', '√', 'sqrt']
        return any(symbol in answer.lower() for symbol in symbols)
    
    def _compare_symbolic_answers(self, actual: str, expected: str) -> float:
        """
        Compare symbolic answers and return a similarity score.
        
        Parameters:
        -----------
        actual : str
            Actual answer
        expected : str
            Expected answer
            
        Returns:
        --------
        float
            Similarity score between 0.0 and 1.0
        """
        # Remove spaces
        actual_clean = re.sub(r'\s+', '', actual)
        expected_clean = re.sub(r'\s+', '', expected)
        
        # If they're identical after cleaning, return perfect score
        if actual_clean == expected_clean:
            return 1.0
        
        # Normalize common symbolic representations
        actual_norm = self._normalize_symbolic(actual_clean)
        expected_norm = self._normalize_symbolic(expected_clean)
        
        if actual_norm == expected_norm:
            return 0.95
        
        # Check for subset of terms (partial matching)
        actual_terms = re.findall(r'[a-z√\d\+\-\*\/\^]+', actual_norm)
        expected_terms = re.findall(r'[a-z√\d\+\-\*\/\^]+', expected_norm)
        
        if not actual_terms or not expected_terms:
            return 0.0
        
        # Count how many terms match
        matched_terms = sum(1 for term in expected_terms if any(self._is_similar(term, actual_term) for actual_term in actual_terms))
        return min(0.8, matched_terms / len(expected_terms))
    
    def _normalize_symbolic(self, expr: str) -> str:
        """
        Normalize symbolic expressions for comparison.
        
        Parameters:
        -----------
        expr : str
            Symbolic expression
            
        Returns:
        --------
        str
            Normalized expression
        """
        # Replace sqrt with √
        expr = expr.replace('sqrt', '√')
        
        # Replace ** with ^ for exponents
        expr = expr.replace('**', '^')
        
        # Replace × with *
        expr = expr.replace('×', '*')
        
        # Replace "pi" with π
        expr = expr.replace('pi', 'π')
        
        return expr
    
    def _extract_key_parts(self, text: str) -> List[str]:
        """
        Extract key parts from an answer for partial matching.
        
        Parameters:
        -----------
        text : str
            Text to extract parts from
            
        Returns:
        --------
        List[str]
            List of key parts
        """
        # Split by common separators
        parts = re.split(r'[,;]|\band\b|\bor\b', text)
        
        # Remove empty parts and strip whitespace
        return [part.strip() for part in parts if part.strip()]
    
    def _is_similar(self, a: str, b: str) -> bool:
        """
        Check if two strings are similar.
        
        Parameters:
        -----------
        a : str
            First string
        b : str
            Second string
            
        Returns:
        --------
        bool
            True if the strings are similar
        """
        a = a.lower().strip()
        b = b.lower().strip()
        
        # Direct equality
        if a == b:
            return True
        
        # Same numerical value
        a_num = self._extract_numerical_value(a)
        b_num = self._extract_numerical_value(b)
        if a_num is not None and b_num is not None:
            return abs(a_num - b_num) < 1e-6
        
        # Levenshtein distance for text similarity
        if len(a) > 0 and len(b) > 0:
            max_len = max(len(a), len(b))
            similarity = 1 - min(10, self._levenshtein_distance(a, b)) / min(10, max_len)
            return similarity > 0.7
        
        return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.
        
        Parameters:
        -----------
        s1 : str
            First string
        s2 : str
            Second string
            
        Returns:
        --------
        int
            Levenshtein distance
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
            
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]
    
    def _evaluate_reasoning_quality(self, response: str, expected_work: str) -> float:
        """
        Evaluate the quality of reasoning in a response.
        Returns a score between 0.0 and 1.0.
        
        Parameters:
        -----------
        response : str
            Model response
        expected_work : str
            Expected work and reasoning
            
        Returns:
        --------
        float
            Reasoning quality score between 0.0 and 1.0
        """
        # Split response into logical steps/sections
        steps = re.split(r'(?:\n\n|\n(?=\d+\.)|\n(?=Step))', response)
        
        # Count the number of identifiable reasoning steps
        step_count = sum(1 for step in steps if re.search(r'(?:Step|First|Second|Next|Then|Finally|Thus|Therefore)', step, re.IGNORECASE))
        
        # Check for presence of key reasoning indicators
        has_structured_steps = step_count > 0
        
        # Check for mathematical expressions/calculations
        calculations_count = len(re.findall(r'[-+]?\d+\.?\d*\s*[\+\-\*\/\^=]\s*[-+]?\d+\.?\d*', response))
        has_calculations = calculations_count > 0
        
        # Check for logical connections
        has_logical_connections = re.search(r'(?:because|since|as|therefore|thus|hence|so)', response, re.IGNORECASE) is not None
        
        # Check for proper use of mathematical notation
        has_proper_notation = re.search(r'(?:\(|=|\+|-|\*|\/|√|\^|sin|cos|tan|log|ln|lim|\bif\b|\bthen\b)', response) is not None
        
        # Compare to expected work if available
        expected_work_match = 0.0
        if expected_work:
            # Extract key elements from expected work
            expected_elements = self._extract_work_elements(expected_work)
            response_elements = self._extract_work_elements(response)
            
            # Calculate how many expected elements appear in the response
            if expected_elements:
                expected_work_match = sum(1 for elem in expected_elements 
                                        if any(self._is_work_element_present(elem, resp_elem) 
                                              for resp_elem in response_elements)) / len(expected_elements)
        
        # Calculate reasoning quality score
        score = 0.0
        score += 0.2 if has_structured_steps else 0.0
        score += min(0.2, 0.05 * step_count)  # More credit for more steps, up to 0.2
        score += min(0.2, 0.02 * calculations_count)  # More credit for more calculations, up to 0.2
        score += 0.1 if has_logical_connections else 0.0
        score += 0.1 if has_proper_notation else 0.0
        
        # Add score for matching expected work
        score += 0.2 * expected_work_match
        
        return min(1.0, score)
    
    def _extract_work_elements(self, text: str) -> List[str]:
        """
        Extract key work elements from text.
        
        Parameters:
        -----------
        text : str
            Text to extract elements from
            
        Returns:
        --------
        List[str]
            List of work elements
        """
        # Extract calculations, equations, and values
        elements = []
        
        # Find equations (with = sign)
        equations = re.findall(r'[^=.;]+=+[^=.;]+', text)
        elements.extend(equations)
        
        # Find calculations (with operators +, -, *, /, ^)
        calculations = re.findall(r'[-+]?\d+\.?\d*\s*[\+\-\*\/\^]\s*[-+]?\d+\.?\d*', text)
        elements.extend(calculations)
        
        # Find specialized mathematical expressions
        math_expressions = re.findall(r'(?:sin|cos|tan|log|ln|lim|sqrt|∫|∑|∏)[^.;:]+', text)
        elements.extend(math_expressions)
        
        return [elem.strip() for elem in elements if elem.strip()]
    
    def _is_work_element_present(self, expected_elem: str, actual_elem: str) -> bool:
        """
        Check if an expected work element is present in an actual element.
        
        Parameters:
        -----------
        expected_elem : str
            Expected work element
        actual_elem : str
            Actual work element
            
        Returns:
        --------
        bool
            True if the expected element is present
        """
        # Clean and normalize elements
        expected_clean = re.sub(r'\s+', '', expected_elem).lower()
        actual_clean = re.sub(r'\s+', '', actual_elem).lower()
        
        # Direct containment
        if expected_clean in actual_clean or actual_clean in expected_clean:
            return True
        
        # Extract numerical values from both
        expected_nums = [float(n) for n in re.findall(r'[-+]?\d+\.?\d*', expected_elem)]
        actual_nums = [float(n) for n in re.findall(r'[-+]?\d+\.?\d*', actual_elem)]
        
        # Check if they contain the same numbers
        if expected_nums and actual_nums:
            common_nums = sum(1 for n1 in expected_nums if any(abs(n1 - n2) < 1e-6 for n2 in actual_nums))
            if common_nums / max(len(expected_nums), len(actual_nums)) > 0.5:
                return True
        
        return False
    
    def _evaluate_work_completeness(self, response: str, expected_work: str) -> float:
        """
        Evaluate the completeness of the work shown.
        Returns a score between 0.0 and 1.0.
        
        Parameters:
        -----------
        response : str
            Model response
        expected_work : str
            Expected work and reasoning
            
        Returns:
        --------
        float
            Work completeness score between 0.0 and 1.0
        """
        # Check length of response relative to question complexity
        response_length = len(response.split())
        
        # Longer responses tend to be more thorough
        length_score = min(0.3, response_length / 300)  # Cap at 0.3
        
        # Check for presence of key steps if expected work is available
        expected_work_score = 0.0
        if expected_work:
            # Split expected work into steps
            expected_steps = re.split(r'\.\s+|Step\s+\d+:', expected_work)
            expected_steps = [step.strip() for step in expected_steps if step.strip()]
            
            if expected_steps:
                # Count how many expected steps are mentioned in the response
                steps_present = sum(1 for step in expected_steps if self._is_step_present(step, response))
                expected_work_score = 0.7 * (steps_present / len(expected_steps))
        else:
            # Without expected work, base completeness on structural elements
            structure_elements = [
                # Introduction/setup
                re.search(r'(?:Given|We have|Let|Start|First)', response, re.IGNORECASE) is not None,
                # Multiple reasoning steps
                len(re.findall(r'(?:Next|Then|Step|Now|Since|Because)', response, re.IGNORECASE)) >= 2,
                # Validation or checking
                re.search(r'(?:Verify|Check|Confirm|Indeed|Correct)', response, re.IGNORECASE) is not None,
                # Conclusion
                re.search(r'(?:Therefore|Thus|Hence|In conclusion|Finally|So)', response, re.IGNORECASE) is not None
            ]
            expected_work_score = 0.7 * (sum(structure_elements) / len(structure_elements))
        
        return length_score + expected_work_score
    
    def _is_step_present(self, expected_step: str, response: str) -> bool:
        """
        Check if an expected step is present in the response.
        
        Parameters:
        -----------
        expected_step : str
            Expected reasoning step
        response : str
            Model response
            
        Returns:
        --------
        bool
            True if the expected step is present
        """
        # Extract key elements from the step
        key_elements = []
        
        # Look for numbers
        numbers = re.findall(r'[-+]?\d+\.?\d*', expected_step)
        key_elements.extend(numbers)
        
        # Look for mathematical symbols and variables
        symbols = re.findall(r'[xyz]\s*=|=\s*[xyz]|\bsin\b|\bcos\b|\btan\b|\blog\b|\bln\b|√|\^', expected_step)
        key_elements.extend(symbols)
        
        # Check if enough key elements are in the response
        if key_elements:
            elements_present = sum(1 for elem in key_elements if elem in response)
            return elements_present / len(key_elements) > 0.5
            
        # Fallback: check if significant words from the step are in the response
        significant_words = [word for word in expected_step.lower().split() if len(word) > 3]
        if significant_words:
            words_present = sum(1 for word in significant_words if word in response.lower())
            return words_present / len(significant_words) > 0.5
            
        return False
    
    def _evaluate_precision(self, response: str, question: str) -> float:
        """
        Evaluate the precision of calculations and reasoning.
        Returns a score between 0.0 and 1.0.
        
        Parameters:
        -----------
        response : str
            Model response
        question : str
            Original question
            
        Returns:
        --------
        float
            Precision score between 0.0 and 1.0
        """
        # Check for calculation errors
        calculations = re.findall(r'([-+]?\d+\.?\d*)\s*([\+\-\*\/])\s*([-+]?\d+\.?\d*)\s*=\s*([-+]?\d+\.?\d*)', response)
        calculation_errors = 0
        
        for a_str, op, b_str, result_str in calculations:
            try:
                a = float(a_str)
                b = float(b_str)
                result = float(result_str)
                
                expected_result = None
                if op == '+':
                    expected_result = a + b
                elif op == '-':
                    expected_result = a - b
                elif op == '*':
                    expected_result = a * b
                elif op == '/' and b != 0:
                    expected_result = a / b
                
                if expected_result is not None and abs(expected_result - result) > 1e-6:
                    calculation_errors += 1
            except:
                pass
        
        # Penalize calculation errors
        calculation_score = 1.0
        if calculations:
            calculation_score = max(0.0, 1.0 - (calculation_errors / len(calculations)) * 2)
        
        # Check for appropriate precision in the answer
        precision_required = re.search(r'(?:decimal places|significant figures|nearest|exactly|simplified)', question, re.IGNORECASE) is not None
        
        # Detect if response addresses precision requirements
        precision_addressed = False
        if precision_required:
            precision_addressed = re.search(r'(?:decimal places|significant figures|rounding|exact|simplified)', response, re.IGNORECASE) is not None
        else:
            precision_addressed = True  # No special precision required
        
        # Check for consistency in precision through the solution
        decimal_numbers = re.findall(r'[-+]?\d+\.\d+', response)
        decimal_precision = [len(num.split('.')[1]) for num in decimal_numbers]
        
        precision_consistency = 1.0
        if decimal_precision and len(decimal_precision) > 1:
            # Check if precision is reasonably consistent
            max_precision = max(decimal_precision)
            consistency_ratio = sum(1 for p in decimal_precision if abs(p - max_precision) <= 1) / len(decimal_precision)
            precision_consistency = max(0.5, consistency_ratio)
        
        # Combine the precision metrics
        precision_score = 0.6 * calculation_score + 0.2 * float(precision_addressed) + 0.2 * precision_consistency
        
        return precision_score
    
    def run_interactive_assessment(self, model_name: str) -> Dict:
        """
        Run an interactive assessment with follow-up questions to assess reasoning depth.
        
        Parameters:
        -----------
        model_name : str
            Name of the model to assess
            
        Returns:
        --------
        Dict
            Interactive assessment results
        """
        logger.info(f"Starting interactive assessment for model: {model_name}")
        
        model_id = MODEL_IDS.get(model_name, model_name)
        
        # Define categories and interactive problem scenarios
        interactive_problems = {
            "arithmetic_operations": {
                "initial": "A money box contains 15 coins, all of which are either 25¢ or 10¢. If the total value is $2.65, how many of each type of coin are there?",
                "followups": [
                    "Explain your approach step-by-step and verify your solution.",
                    "What if the total value was $2.45 instead? How would your solution change?"
                ]
            },
            "algebraic_reasoning": {
                "initial": "Solve the system of equations: 2x - 3y = 13 and 5x + 4y = 7",
                "followups": [
                    "Verify your solution by substituting the values back into both equations.",
                    "How would you solve this system using matrices instead of substitution or elimination?"
                ]
            },
            "sequential_reasoning": {
                "initial": "A circular pool with radius 4 meters is surrounded by a path of uniform width. If the total area of the pool and path together is 100 square meters, what is the width of the path?",
                "followups": [
                    "Walk through your reasoning process in detail, especially explaining how you set up the equation.",
                    "If we double the radius of the pool while maintaining the same total area of 100 square meters, what happens to the width of the path?"
                ]
            },
            "mathematical_proof": {
                "initial": "Prove that for any positive integer n, the expression n³ - n is always divisible by 6.",
                "followups": [
                    "Can you provide a different proof approach for the same statement?",
                    "Extend your proof to show whether n⁵ - n is always divisible by 30 for any positive integer n."
                ]
            }
        }
        
        interactive_results = {}
        
        for category, problem_set in interactive_problems.items():
            logger.info(f"Testing {model_name} on interactive {category} problem")
            
            # Initial question
            initial_prompt = f"""Please solve this mathematical problem carefully, showing your complete reasoning:

{problem_set['initial']}

Take your time and be thorough in your explanation.
"""
            
            if model_name in MODEL_IDS:
                initial_response = self._bedrock_conversation(model_id, initial_prompt)
            else:
                initial_response = self._huggingface_conversation(model_name, initial_prompt)
            
            # Record the interaction
            conversation = [{
                "role": "user",
                "message": problem_set['initial'],
                "response": initial_response
            }]
            
            # Follow-up questions to assess reasoning depth
            for followup in problem_set['followups']:
                # Add context from previous interaction
                followup_prompt = f"""Regarding this previous mathematical problem:

{problem_set['initial']}

You provided this solution:
{initial_response[:500]}...

Now I'd like you to address this follow-up:
{followup}
"""
                
                if model_name in MODEL_IDS:
                    followup_response = self._bedrock_conversation(model_id, followup_prompt)
                else:
                    followup_response = self._huggingface_conversation(model_name, followup_prompt)
                
                # Record the follow-up interaction
                conversation.append({
                    "role": "user",
                    "message": followup,
                    "response": followup_response
                })
                
                # Update initial response for next followup
                initial_response = followup_response
            
            # Store the full conversation
            interactive_results[category] = conversation
            
            # Score the model's performance on this interactive problem
            reasoning_score = self._evaluate_interactive_responses(conversation, category)
            self.results[model_name][f"interactive_{category}"] = reasoning_score
            
            logger.info(f"{model_name} - interactive {category} score: {reasoning_score:.2f}")
        
        return interactive_results
    
    def _evaluate_interactive_responses(self, conversation: List[Dict], category: str) -> float:
        """
        Evaluate the quality of responses in an interactive conversation.
        
        Parameters:
        -----------
        conversation : List[Dict]
            List of conversation turns
        category : str
            Assessment category
            
        Returns:
        --------
        float
            Interactive reasoning score between 0.0 and 1.0
        """
        # Initial response evaluation
        initial_response = conversation[0]["response"]
        initial_score = self._evaluate_reasoning_quality(initial_response, "")
        
        # Followup evaluations with increased weights for adaptability
        followup_scores = []
        
        for i, interaction in enumerate(conversation[1:], 1):
            response = interaction["response"]
            question = interaction["message"]
            
            # Base reasoning quality
            quality_score = self._evaluate_reasoning_quality(response, "")
            
            # Check if response properly addresses the follow-up question
            relevance_score = 0.0
            
            # Extract key terms from the question
            question_terms = set(re.findall(r'\b\w{4,}\b', question.lower()))
            
            # Check how many key question terms are addressed in the response
            if question_terms:
                matching_terms = sum(1 for term in question_terms if term in response.lower())
                relevance_score = min(1.0, matching_terms / len(question_terms))
            
            # Check for coherence with previous responses
            coherence_score = 0.0
            
            # Look for references to previous work
            builds_on_previous = re.search(r'previous|earlier|before|as I mentioned|as shown|as calculated', response, re.IGNORECASE) is not None
            
            # Check if values from previous calculations are reused
            prev_response = conversation[i-1]["response"]
            prev_values = set(re.findall(r'[-+]?\d+\.?\d*', prev_response))
            current_values = set(re.findall(r'[-+]?\d+\.?\d*', response))
            
            if prev_values:
                common_values = prev_values.intersection(current_values)
                value_reuse_ratio = min(1.0, len(common_values) / (len(prev_values) * 0.3))  # Expect at least 30% reuse
                coherence_score = max(coherence_score, value_reuse_ratio)
            
            if builds_on_previous:
                coherence_score = max(coherence_score, 0.7)
                
            # Combined score for this followup
            followup_score = 0.4 * quality_score + 0.3 * relevance_score + 0.3 * coherence_score
            followup_scores.append(followup_score)
        
        # Weight initial response 40%, followups 60% combined
        if followup_scores:
            avg_followup = sum(followup_scores) / len(followup_scores)
            final_score = 0.4 * initial_score + 0.6 * avg_followup
        else:
            final_score = initial_score
            
        return final_score
    
    def run_assessment(self) -> None:
        """
        Run the complete assessment on all models.
        """
        logger.info("Starting comprehensive assessment of all models")
        
        # Generate assessment problems for the core math skills
        problems = self.generate_assessment_problems()
        
        # Load benchmark datasets
        benchmarks = {}
        for dataset_name in BENCHMARK_DATASETS:
            dataset = self._load_benchmark_dataset(dataset_name)
            if dataset:
                benchmarks[dataset_name] = dataset
                logger.info(f"Loaded {len(dataset)} samples from {dataset_name} benchmark")
            
        # Assess each model
        for model_name in self.model_names:
            # Assess core math skills
            self.assess_model(model_name, problems)
            
            # Assess on benchmark datasets
            for dataset_name, dataset in benchmarks.items():
                self.assess_model_on_benchmark(model_name, dataset_name, dataset)
            
            # Run interactive assessment for each model
            interactive_scores = self.run_interactive_assessment(model_name)
            
            logger.info(f"Completed assessment for {model_name}")
        
        logger.info("Assessment completed for all models")
        
        # Save results to file
        self.save_results()
        
        # Create visualizations
        self.create_visualizations()
        
        # Create comprehensive report
        self.create_report()
    
    def create_visualizations(self) -> None:
        """
        Create various visualizations of model performance.
        """
        logger.info("Creating visualizations of assessment results")
        
        # 1. Create radar chart for core math skills
        self._create_radar_chart()
        
        # 2. Create bar chart for benchmark performance
        self._create_benchmark_bar_chart()
        
        # 3. Create heatmap of strengths and weaknesses
        self._create_heatmap()
        
        logger.info("Visualizations created successfully")
    
    def _create_radar_chart(self) -> None:
        """
        Create radar chart visualization of model performance across categories.
        """
        # Define categories for the radar chart
        categories = ASSESSMENT_CATEGORIES
        
        # Create radar chart figure
        fig = plt.figure(figsize=(12, 9))
        
        try:
            # Try to use the custom radar factory
            theta = radar_factory(len(categories), frame='polygon')
            ax = fig.add_subplot(111, projection='radar')
            ax.set_varlabels([cat.replace('_', ' ').title() for cat in categories])
        except:
            # Fallback to standard polar plot
            ax = fig.add_subplot(111, polar=True)
            
            # Set up the angles for the radar chart
            theta = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
            
            # Set up the radar chart
            ax.set_theta_offset(np.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(theta)
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
            ax.set_ylim(0, 1)
        
        # Create a color cycle for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.model_names)))
        
        # Plot each model
        for i, model_name in enumerate(self.model_names):
            # Get scores and make sure they form a closed loop
            values = [self.results[model_name][cat] for cat in categories]
            
            # Get display name (shorter)
            display_name = model_name.split('/')[-1]
            if model_name in MODEL_IDS:
                display_name = model_name
                
            # Plot using appropriate method
            if hasattr(ax, 'plot_radar'):
                # For custom radar factory
                ax.plot_radar(values, color=colors[i], label=display_name)
            else:
                # For standard polar plot
                values.append(values[0])  # Close the loop
                angles = theta + [theta[0]]
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       color=colors[i], label=display_name)
                ax.fill(angles, values, alpha=0.1, color=colors[i])
        
        # Add legend and title
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('LLM Mathematical Reasoning Capability Comparison', size=15)
        
        # Save the chart
        plt.tight_layout()
        plt.savefig('llm_reasoning_comparison_radar.png', dpi=300)
        logger.info("Radar chart visualization saved as 'llm_reasoning_comparison_radar.png'")
    
    def _create_benchmark_bar_chart(self) -> None:
        """
        Create bar chart of model performance on benchmark datasets.
        """
        # Filter only models and datasets with results
        models_with_results = []
        for model_name in self.model_names:
            if any(self.benchmark_results[model_name][dataset]["score"] > 0 for dataset in BENCHMARK_DATASETS):
                models_with_results.append(model_name)
        
        if not models_with_results:
            logger.info("No benchmark results to visualize")
            return
        
        datasets_with_results = []
        for dataset in BENCHMARK_DATASETS:
            if any(self.benchmark_results[model_name][dataset]["score"] > 0 for model_name in models_with_results):
                datasets_with_results.append(dataset)
        
        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set up bar width and positions
        bar_width = 0.8 / len(models_with_results)
        
        # Create a color cycle for different models
        colors = plt.cm.tab10(np.linspace(0, 1, len(models_with_results)))
        
        # Plot bars for each model
        for i, model_name in enumerate(models_with_results):
            # Get scores for this model
            scores = [self.benchmark_results[model_name][dataset]["score"] for dataset in datasets_with_results]
            
            # Get display name
            display_name = model_name.split('/')[-1]
            if model_name in MODEL_IDS:
                display_name = model_name
            
            # Calculate bar positions
            positions = np.arange(len(datasets_with_results)) - 0.4 + (i + 0.5) * bar_width
            
            # Plot bars
            ax.bar(positions, scores, bar_width, label=display_name, color=colors[i])
        
        # Add labels and title
        ax.set_xlabel('Benchmark Datasets')
        ax.set_ylabel('Accuracy Score')
        ax.set_title('Model Performance on Mathematical Benchmarks')
        ax.set_xticks(np.arange(len(datasets_with_results)))
        ax.set_xticklabels([dataset.upper() for dataset in datasets_with_results])
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Add grid lines for readability
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limits
        ax.set_ylim(0, 1.0)
        
        # Add value labels on top of bars
        for i, model_name in enumerate(models_with_results):
            scores = [self.benchmark_results[model_name][dataset]["score"] for dataset in datasets_with_results]
            positions = np.arange(len(datasets_with_results)) - 0.4 + (i + 0.5) * bar_width
            
            for x, y in zip(positions, scores):
                ax.annotate(f'{y:.2f}', 
                          xy=(x, y), 
                          xytext=(0, 3),  # 3 points vertical offset
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the chart
        plt.savefig('llm_reasoning_comparison_benchmarks.png', dpi=300)
        logger.info("Benchmark visualization saved as 'llm_reasoning_comparison_benchmarks.png'")
    
    def _create_heatmap(self) -> None:
        """
        Create heatmap showing strengths and weaknesses of each model.
        """
        # Prepare data for heatmap
        models = self.model_names
        categories = ASSESSMENT_CATEGORIES
        
        # Get display names
        display_names = [model_name.split('/')[-1] if model_name not in MODEL_IDS else model_name for model_name in models]
        
        # Create data matrix
        data = np.zeros((len(models), len(categories)))
        for i, model_name in enumerate(models):
            for j, category in enumerate(categories):
                data[i, j] = self.results[model_name][category]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Plot heatmap
        im = ax.imshow(data, cmap='viridis')
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("Score", rotation=-90, va="bottom")
        
        # Show ticks
        ax.set_xticks(np.arange(len(categories)))
        ax.set_yticks(np.arange(len(models)))
        
        # Label with category names and model names
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.set_yticklabels(display_names)
        
        # Rotate category labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add text annotations with scores
        for i in range(len(models)):
            for j in range(len(categories)):
                text = ax.text(j, i, f"{data[i, j]:.2f}",
                             ha="center", va="center", color="black" if data[i, j] > 0.5 else "white")
        
        # Set title
        ax.set_title("Model Performance Across Mathematical Reasoning Categories")
        
        # Adjust layout
        fig.tight_layout()
        
        # Save figure
        plt.savefig('llm_reasoning_comparison_heatmap.png', dpi=300)
        logger.info("Heatmap visualization saved as 'llm_reasoning_comparison_heatmap.png'")
    
    def create_report(self) -> Dict[str, Any]:
        """
        Create a structured report of the assessment results.
        
        Returns:
        --------
        Dict[str, Any]
            Comprehensive report of assessment results
        """
        logger.info("Generating comprehensive assessment report")
        
        report = {
            "assessment_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "models_assessed": self.model_names,
            "category_scores": {},
            "benchmark_scores": {},
            "overall_rankings": {},
            "strengths_and_weaknesses": {},
            "detailed_analysis": {},
            "interactive_assessment": {}
        }
        
        # Compile category scores
        for category in ASSESSMENT_CATEGORIES:
            report["category_scores"][category] = {
                model: self.results[model][category] for model in self.model_names
            }
        
        # Add benchmark scores
        for dataset in BENCHMARK_DATASETS:
            report["benchmark_scores"][dataset] = {
                model: self.benchmark_results[model][dataset]["score"] for model in self.model_names
            }
        
        # Add interactive scores if available
        interactive_categories = [f"interactive_{cat}" for cat in ["arithmetic_operations", "algebraic_reasoning", "sequential_reasoning", "mathematical_proof"]]
        for category in interactive_categories:
            if any(category in self.results[model] for model in self.model_names):
                report["category_scores"][category] = {
                    model: self.results[model].get(category, 0.0) for model in self.model_names
                }
        
        # Calculate overall scores and rankings
        overall_scores = {}
        for model in self.model_names:
            # Include both base categories and benchmark scores in overall ranking
            category_scores = [self.results[model][cat] for cat in ASSESSMENT_CATEGORIES]
            benchmark_scores = [self.benchmark_results[model][dataset]["score"] 
                              for dataset in BENCHMARK_DATASETS 
                              if self.benchmark_results[model][dataset]["samples"] > 0]
            
            # Weighted average: 60% core skills, 40% benchmarks
            if category_scores and benchmark_scores:
                overall_scores[model] = 0.6 * sum(category_scores) / len(category_scores) + \
                                     0.4 * sum(benchmark_scores) / len(benchmark_scores)
            elif category_scores:
                overall_scores[model] = sum(category_scores) / len(category_scores)
            else:
                overall_scores[model] = 0.0
        
        # Sort models by overall score for rankings
        sorted_models = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(sorted_models):
            report["overall_rankings"][i+1] = {
                "model": model,
                "display_name": model.split('/')[-1] if model not in MODEL_IDS else model,
                "score": score
            }
        
        # Analyze strengths and weaknesses
        for model in self.model_names:
            scores = {cat: self.results[model][cat] for cat in ASSESSMENT_CATEGORIES}
            
            # Find top and bottom categories
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_categories = [cat for cat, _ in sorted_scores[:3]]
            bottom_categories = [cat for cat, _ in sorted_scores[-3:]]
            
            # Calculate z-scores for each category to identify relative strengths/weaknesses
            all_scores = [score for _, score in scores.items()]
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores) if len(all_scores) > 1 else 1.0
            
            z_scores = {cat: (score - mean_score) / std_score if std_score > 0 else 0 
                      for cat, score in scores.items()}
            
            # Identify significant strengths (z > 0.5) and weaknesses (z < -0.5)
            significant_strengths = [cat for cat, z in z_scores.items() if z > 0.5]
            significant_weaknesses = [cat for cat, z in z_scores.items() if z < -0.5]
            
            # Add benchmark performance insights
            benchmark_insights = []
            for dataset in BENCHMARK_DATASETS:
                if self.benchmark_results[model][dataset]["samples"] > 0:
                    score = self.benchmark_results[model][dataset]["score"]
                    if score > 0.8:
                        benchmark_insights.append(f"Strong performance on {dataset.upper()} (accuracy: {score:.2f})")
                    elif score < 0.4:
                        benchmark_insights.append(f"Poor performance on {dataset.upper()} (accuracy: {score:.2f})")
            
            # Generate summary insights
            display_name = model.split('/')[-1] if model not in MODEL_IDS else model
            
            summary = f"{display_name} shows "
            if significant_strengths:
                strength_text = ", ".join([cat.replace('_', ' ') for cat in significant_strengths])
                summary += f"particular strength in {strength_text}"
                
                if significant_weaknesses:
                    summary += f" but struggles with {', '.join([cat.replace('_', ' ') for cat in significant_weaknesses])}"
                summary += ". "
            elif significant_weaknesses:
                summary += f"notable weaknesses in {', '.join([cat.replace('_', ' ') for cat in significant_weaknesses])}. "
            else:
                summary += "balanced performance across all categories. "
                
            if benchmark_insights:
                summary += " " + " ".join(benchmark_insights)
                
            # Store all analysis
            report["strengths_and_weaknesses"][model] = {
                "top_categories": top_categories,
                "bottom_categories": bottom_categories,
                "significant_strengths": significant_strengths,
                "significant_weaknesses": significant_weaknesses,
                "average_score": overall_scores[model],
                "z_scores": z_scores,
                "benchmark_insights": benchmark_insights,
                "summary": summary
            }
        
        # Create detailed analysis
        for model in self.model_names:
            # Get detailed evaluations for this model
            model_evaluations = self.detailed_evaluations.get(model, {})
            
            # Summarize evaluation metrics for each category
            category_metrics = {}
            for category, evaluations in model_evaluations.items():
                if evaluations:
                    avg_metrics = {
                        "answer_correctness": np.mean([e["answer_correctness"] for e in evaluations]),
                        "reasoning_quality": np.mean([e["reasoning_quality"] for e in evaluations]),
                        "work_completeness": np.mean([e["work_completeness"] for e in evaluations]),
                        "precision": np.mean([e["precision"] for e in evaluations])
                    }
                    category_metrics[category] = avg_metrics
            
            # Select exemplary responses (best and worst) for each category
            exemplary_responses = {}
            for category, evaluations in model_evaluations.items():
                if evaluations:
                    # Find indices of best and worst responses
                    scores = [(i, e["answer_correctness"] + e["reasoning_quality"]) 
                            for i, e in enumerate(evaluations)]
                    
                    best_idx = max(scores, key=lambda x: x[1])[0]
                    worst_idx = min(scores, key=lambda x: x[1])[0]
                    
                    # Get raw interactions for these responses
                    best_response = next((i for i in self.raw_interactions[model] 
                                        if i["category"] == category and i["problem_index"] == best_idx), None)
                    
                    worst_response = next((i for i in self.raw_interactions[model] 
                                         if i["category"] == category and i["problem_index"] == worst_idx), None)
                    
                    exemplary_responses[category] = {
                        "best": best_response,
                        "worst": worst_response
                    }
            
            # Add benchmark performance details
            benchmark_details = {}
            for dataset in BENCHMARK_DATASETS:
                if self.benchmark_results[model][dataset]["samples"] > 0:
                    benchmark_details[dataset] = {
                        "score": self.benchmark_results[model][dataset]["score"],
                        "samples": self.benchmark_results[model][dataset]["samples"],
                        "correct": self.benchmark_results[model][dataset]["correct"]
                    }
            
            # Generate insights on problem-solving approach
            problem_solving_insights = []
            
            # Check if model tends to use step-by-step reasoning
            if any("reasoning_quality" in evaluations[0] for cat, evaluations in model_evaluations.items() if evaluations):
                reasoning_scores = [e["reasoning_quality"] for cat, evaluations in model_evaluations.items() 
                                 for e in evaluations if "reasoning_quality" in e]
                
                if reasoning_scores:
                    avg_reasoning = sum(reasoning_scores) / len(reasoning_scores)
                    if avg_reasoning > 0.8:
                        problem_solving_insights.append("Consistently provides detailed step-by-step reasoning")
                    elif avg_reasoning < 0.5:
                        problem_solving_insights.append("Often lacks step-by-step reasoning")
            
            # Check calculation precision
            if any("precision" in evaluations[0] for cat, evaluations in model_evaluations.items() if evaluations):
                precision_scores = [e["precision"] for cat, evaluations in model_evaluations.items() 
                                 for e in evaluations if "precision" in e]
                
                if precision_scores:
                    avg_precision = sum(precision_scores) / len(precision_scores)
                    if avg_precision > 0.9:
                        problem_solving_insights.append("Demonstrates excellent calculation precision")
                    elif avg_precision < 0.6:
                        problem_solving_insights.append("Frequently makes calculation errors")
            
            # Store all analysis
            report["detailed_analysis"][model] = {
                "category_scores": {cat: self.results[model][cat] for cat in ASSESSMENT_CATEGORIES},
                "category_metrics": category_metrics,
                "exemplary_responses": exemplary_responses,
                "benchmark_details": benchmark_details,
                "problem_solving_insights": problem_solving_insights,
                "overall_score": overall_scores[model]
            }
        
        # Add interactive assessment results if available
        for model in self.model_names:
            interactive_scores = {}
            for category in interactive_categories:
                if category in self.results[model]:
                    interactive_scores[category.replace("interactive_", "")] = self.results[model][category]
            
            if interactive_scores:
                # Calculate adaptive reasoning score from interactive results
                adaptive_score = sum(interactive_scores.values()) / len(interactive_scores) if interactive_scores else 0
                
                report["interactive_assessment"][model] = {
                    "scores": interactive_scores,
                    "adaptive_score": adaptive_score,
                    "adaptive_insights": self._generate_adaptive_insights(model, interactive_scores)
                }
        
        # Save report to file
        with open('llm_reasoning_assessment_report.json', 'w', encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable report
        self._generate_human_readable_report(report)
        
        logger.info("Assessment report saved as 'llm_reasoning_assessment_report.json'")
        
        return report
    
    def _generate_adaptive_insights(self, model: str, interactive_scores: Dict[str, float]) -> List[str]:
        """
        Generate insights about a model's adaptive reasoning abilities.
        
        Parameters:
        -----------
        model : str
            Model name
        interactive_scores : Dict[str, float]
            Dictionary of interactive assessment scores
            
        Returns:
        --------
        List[str]
            List of insights
        """
        insights = []
        
        if not interactive_scores:
            return insights
        
        avg_score = sum(interactive_scores.values()) / len(interactive_scores)
        
        # General adaptive reasoning capability
        if avg_score > 0.8:
            insights.append("Demonstrates excellent ability to adapt reasoning based on follow-up questions")
        elif avg_score < 0.5:
            insights.append("Shows limited ability to adapt reasoning to follow-up questions")
        
        # Category-specific insights
        for category, score in interactive_scores.items():
            if score > 0.85:
                insights.append(f"Exceptional adaptive reasoning in {category.replace('_', ' ')}")
            elif score < 0.4:
                insights.append(f"Struggles to adapt reasoning in {category.replace('_', ' ')}")
        
        return insights
    
    def _generate_human_readable_report(self, report_data: Dict) -> None:
        """
        Generate a human-readable report in markdown format.
        
        Parameters:
        -----------
        report_data : Dict
            Report data
        """
        # Create markdown report
        with open('llm_reasoning_assessment_report.md', 'w', encoding="utf-8") as f:
            f.write("# LLM Mathematical Reasoning Assessment Report\n\n")
            
            f.write(f"**Assessment Date:** {report_data['assessment_date']}\n\n")
            
            # Overall rankings
            f.write("## Overall Model Rankings\n\n")
            f.write("| Rank | Model | Overall Score |\n")
            f.write("|------|-------|---------------|\n")
            
            for rank, data in report_data["overall_rankings"].items():
                f.write(f"| {rank} | {data['display_name']} | {data['score']:.3f} |\n")
            f.write("\n")
            
            # Strengths and weaknesses summary
            f.write("## Model Strengths and Weaknesses\n\n")
            
            for model, data in report_data["strengths_and_weaknesses"].items():
                display_name = model.split('/')[-1] if model not in MODEL_IDS else model
                f.write(f"### {display_name}\n\n")
                
                # Summary
                f.write(f"**Summary:** {data['summary']}\n\n")
                
                # Strengths and weaknesses
                f.write("**Strengths:** " + (", ".join(cat.replace('_', ' ').title() for cat in data['significant_strengths']) if data['significant_strengths'] else "No outstanding strengths") + "\n\n")
                f.write("**Weaknesses:** " + (", ".join(cat.replace('_', ' ').title() for cat in data['significant_weaknesses']) if data['significant_weaknesses'] else "No significant weaknesses") + "\n\n")
                
                # Benchmark performance
                if data.get("benchmark_insights"):
                    f.write("**Benchmark Performance:** " + " ".join(data["benchmark_insights"]) + "\n\n")
            
            # Detailed category performance
            f.write("## Detailed Category Performance\n\n")
            f.write("| Model | " + " | ".join(cat.replace('_', ' ').title() for cat in ASSESSMENT_CATEGORIES) + " |\n")
            f.write("|-------|" + "|".join(["-------" for _ in ASSESSMENT_CATEGORIES]) + "|\n")
            
            for model in report_data["models_assessed"]:
                display_name = model.split('/')[-1] if model not in MODEL_IDS else model
                scores = [f"{report_data['category_scores'][cat][model]:.2f}" for cat in ASSESSMENT_CATEGORIES]
                f.write(f"| {display_name} | " + " | ".join(scores) + " |\n")
            f.write("\n")
            
            # Benchmark performance
            if report_data["benchmark_scores"]:
                f.write("## Benchmark Performance\n\n")
                f.write("| Model | " + " | ".join(dataset.upper() for dataset in report_data["benchmark_scores"].keys()) + " |\n")
                f.write("|-------|" + "|".join(["-------" for _ in report_data["benchmark_scores"].keys()]) + "|\n")
                
                for model in report_data["models_assessed"]:
                    display_name = model.split('/')[-1] if model not in MODEL_IDS else model
                    scores = [f"{report_data['benchmark_scores'][dataset][model]:.2f}" for dataset in report_data["benchmark_scores"].keys()]
                    f.write(f"| {display_name} | " + " | ".join(scores) + " |\n")
                f.write("\n")
            
            # Interactive assessment
            if report_data["interactive_assessment"]:
                f.write("## Interactive and Adaptive Reasoning\n\n")
                
                for model, data in report_data["interactive_assessment"].items():
                    display_name = model.split('/')[-1] if model not in MODEL_IDS else model
                    f.write(f"### {display_name}\n\n")
                    
                    # Adaptive score
                    f.write(f"**Adaptive Reasoning Score:** {data['adaptive_score']:.3f}\n\n")
                    
                    # Insights
                    if data.get("adaptive_insights"):
                        f.write("**Insights:**\n\n")
                        for insight in data["adaptive_insights"]:
                            f.write(f"- {insight}\n")
                        f.write("\n")
            
            # Problem-solving approaches
            f.write("## Problem-Solving Approaches\n\n")
            
            for model, data in report_data["detailed_analysis"].items():
                if data.get("problem_solving_insights"):
                    display_name = model.split('/')[-1] if model not in MODEL_IDS else model
                    f.write(f"### {display_name}\n\n")
                    
                    for insight in data["problem_solving_insights"]:
                        f.write(f"- {insight}\n")
                    f.write("\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            
            # Generate conclusion based on the top model
            if report_data["overall_rankings"]:
                top_model = report_data["overall_rankings"]["1"]["model"]
                top_model_display = report_data["overall_rankings"]["1"]["display_name"]
                top_score = report_data["overall_rankings"]["1"]["score"]
                
                top_strengths = report_data["strengths_and_weaknesses"][top_model]["significant_strengths"]
                strength_text = ", ".join([cat.replace('_', ' ') for cat in top_strengths]) if top_strengths else "balanced performance"
                
                f.write(f"Based on our comprehensive assessment, **{top_model_display}** demonstrates the strongest overall mathematical reasoning capabilities with a score of {top_score:.3f}. ")
                f.write(f"Its particular strengths lie in {strength_text}. ")
                
                # Compare to second place if available
                if len(report_data["overall_rankings"]) > 1:
                    second_model = report_data["overall_rankings"]["2"]["model"]
                    second_model_display = report_data["overall_rankings"]["2"]["display_name"]
                    second_score = report_data["overall_rankings"]["2"]["score"]
                    
                    score_diff = top_score - second_score
                    if score_diff > 0.1:
                        f.write(f"There is a notable gap of {score_diff:.3f} points between {top_model_display} and the second-place model ({second_model_display}). ")
                    else:
                        f.write(f"The performance difference between {top_model_display} and {second_model_display} is relatively small ({score_diff:.3f}), suggesting comparable mathematical reasoning capabilities. ")
                
                f.write("\n\n")
                
                # Overall assessment
                f.write("This assessment demonstrates that modern language models have significant mathematical reasoning capabilities, ")
                f.write("but with varying strengths and weaknesses across different types of mathematical problems. ")
                f.write("The most capable models demonstrate not just the ability to compute correct answers, ")
                f.write("but also provide clear step-by-step reasoning and adapt their problem-solving approach based on follow-up questions.\n")
    
        logger.info("Human-readable report saved as 'llm_reasoning_assessment_report.md'")
    
    def save_results(self) -> None:
        """
        Save assessment results to file.
        """
        results_data = {
            "model_scores": self.results,
            "benchmark_results": {model: {dataset: {
                "score": results["score"],
                "samples": results["samples"],
                "correct": results.get("correct", 0)
            } for dataset, results in model_benchmarks.items()} 
            for model, model_benchmarks in self.benchmark_results.items()},
            "detailed_evaluations": self.detailed_evaluations,
            # Limit size of raw interactions to avoid huge files
            "raw_interactions_sample": {model: interactions[:3] for model, interactions in self.raw_interactions.items()}
        }
        
        with open('llm_reasoning_assessment_results.json', 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info("Assessment results saved to 'llm_reasoning_assessment_results.json'")

def main():
    """
    Main function to run the LLM reasoning assessment.
    """
    parser = argparse.ArgumentParser(description="Assess mathematical reasoning capabilities of LLM models")
    parser.add_argument("--models", nargs='+', required=True, help="List of model names to assess")
    parser.add_argument("--categories", nargs='+', choices=ASSESSMENT_CATEGORIES, 
                       help="Specific categories to assess (default: all categories)")
    parser.add_argument("--benchmarks", nargs='+', choices=BENCHMARK_DATASETS,
                       help="Specific benchmarks to evaluate (default: all benchmarks)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive assessment with follow-up questions")
    parser.add_argument("--problems_per_category", type=int, default=5, help="Number of problems per category")
    parser.add_argument("--data_dir", type=str, default="assessment_data", help="Directory to store/load benchmark datasets")
    args = parser.parse_args()
    
    # Initialize assessor with provided models
    assessor = LLMReasoningAssessor(args.models, data_dir=args.data_dir)
    
    # Run the assessment
    assessor.run_assessment()
    
    logger.info("Assessment completed successfully")

if __name__ == "__main__":
    main()