"""
Configuration for the embedding noise experiment.

All experiment parameters in one place for easy tuning.
"""

# ============================================================================
# Model Configuration
# ============================================================================

# Choose one mid-scale model (7B-13B parameters)
# Options:
#   - "Qwen/Qwen3-8B"
#   - "deepseek-ai/deepseek-coder-6.7b-base"
#   - "deepseek-ai/deepseek-coder-6.7b-instruct"
#   - "codellama/CodeLlama-7b-hf"
#   - "codellama/CodeLlama-13b-hf"
MODEL_NAME = "meta-llama/Llama-3.1-8B"

# Memory optimization
LOAD_IN_8BIT = False  # Set to True if VRAM limited
TORCH_DTYPE = "float16"  # "float16" or "bfloat16" or "float32"

# ============================================================================
# Experimental Conditions
# ============================================================================

# Number of outputs per condition (for B and C)
K_SAMPLES = 10

# Condition B: Temperature Sampling
TEMPERATURE = 0.8  # Tuned for fluent but varied outputs

# Condition C: Embedding Noise
SIGMA_SCALE = 0.01  # Noise magnitude as fraction of embedding norm
NOISE_SCOPE = "per_sequence"  # "per_sequence" or "per_token"

# Condition D: Temperature + Embedding Noise (combined)
# Uses TEMPERATURE and SIGMA_SCALE from above
# This tests whether combining both methods yields additive benefits

# ============================================================================
# Generation Settings
# ============================================================================

MAX_NEW_TOKENS = 1024  # Enough for full competitive programming problem
MIN_NEW_TOKENS = 64  # Prevent empty/newline-only generations
GENERATION_TIMEOUT = 60  # Seconds per generation (safety)

# ============================================================================
# Prompt Configuration
# ============================================================================

N_PROMPTS = 10  # Number of distinct prompts to use

# ============================================================================
# Output Configuration
# ============================================================================

OUTPUT_DIR = "core/outputs"
SAVE_INDIVIDUAL_FILES = True  # Save each output as separate .txt
SAVE_SUMMARY_JSON = True  # Save consolidated all_results.json

# ============================================================================
# Runtime Settings
# ============================================================================

SHOW_PROGRESS = True
VERBOSE = True
RANDOM_SEED = 42

# ============================================================================
# Hardware
# ============================================================================

# IMPORTANT: Use explicit device map to avoid silent CPU offload on Windows.
# `device_map="auto"` can cause extremely slow generation.
DEVICE_MAP = {"": 0}  # Force everything onto GPU 0

# ============================================================================
# Evaluation Settings
# ============================================================================

# Algorithm class choices for manual annotation
ALG_CLASS_CHOICES = {"greedy", "dp", "graph", "math", "brute", "ds", "other"}

# LLM Judge settings
GROQ_MODEL = "openai/gpt-oss-120b"
GEMINI_MODEL = "gemini-2.0-flash"
JUDGE_RATE_LIMIT_DELAY = 0.5  # Seconds between API calls
