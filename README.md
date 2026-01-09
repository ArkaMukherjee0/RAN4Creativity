# Embedding-Level Noise for Creative Generation

A research framework for injecting controlled Gaussian noise into language model embeddings to induce output diversity while maintaining deterministic decoding.

## Table of Contents

1. [The Big Picture (0→1 Explanation)](#the-big-picture-01-explanation)
2. [Deep Dive (1→100 Explanation)](#deep-dive-1100-explanation)
3. [Technical Workflow](#technical-workflow)
4. [Why This Should Work (Theory)](#why-this-should-work-theory)
5. [Usage Guide](#usage-guide)
6. [File Structure](#file-structure)

---

## The Big Picture (0→1 Explanation)

### What Problem Are We Solving?

When you ask a language model to generate something creative (like a programming problem), you have two common approaches:

1. **Greedy decoding** (temperature=0): Always picks the most likely next token. Produces consistent, high-quality output, but you get the *same* output every time. No diversity.

2. **Temperature sampling** (temperature>0): Randomly samples from the token distribution. Produces diverse outputs, but randomness is injected at *every single token*, which can lead to incoherent or low-quality outputs.

### The Core Insight

**What if we could get diversity WITHOUT sacrificing coherence?**

The key observation is that creativity happens at the *planning* level, not the word-by-word level. When a human writes a creative story, they don't randomly pick words—they decide on a creative *direction* first, then execute it coherently.

### Our Approach

We inject a small amount of random noise into the model's **embedding layer** exactly **once**, at the very beginning of generation. This:

- Nudges the model toward a slightly different "plan" or "direction"
- Then lets it execute that plan deterministically (greedy decoding)
- Results in diverse outputs that are each internally coherent

Think of it like this: instead of a drunk person stumbling randomly with each step, we gently push a sober person in a random direction at the start, then let them walk straight.

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   TEMPERATURE SAMPLING          vs.     EMBEDDING NOISE             │
│                                                                     │
│   "I want to... [random]                "I want to write about      │
│    write... [random]                     sorting algorithms"        │
│    about... [random]                           ↓                    │
│    sorting... [random]"                  [deterministic execution]  │
│                                                                     │
│   Randomness at EVERY step              Randomness at START only    │
│   → Can lose coherence                  → Maintains coherence       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Deep Dive (1→100 Explanation)

### How Language Models Generate Text

Let's trace through what happens when a language model generates text:

```
Input: "Write a programming problem"
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  1. TOKENIZATION                                                    │
│     "Write a programming problem" → [15043, 257, 8300, 1917]       │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  2. EMBEDDING LAYER                                                 │
│     Each token ID → dense vector in high-dimensional space          │
│     [15043] → [0.23, -0.11, 0.87, ..., 0.42]  (e.g., 4096 dims)    │
│     [257]   → [0.15, 0.33, -0.22, ..., 0.18]                       │
│     ...                                                             │
│                                                                     │
│     Result: (seq_len × embed_dim) tensor                           │
│     e.g., (4 × 4096) for 4 tokens                                  │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  3. TRANSFORMER LAYERS (×32 or more)                               │
│     - Self-attention: tokens "look at" each other                  │
│     - Feed-forward: non-linear transformations                     │
│     - Each layer refines the representations                       │
│                                                                     │
│     The model builds up an understanding of context and intent     │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  4. OUTPUT HEAD                                                     │
│     Final hidden states → probability distribution over vocabulary  │
│     e.g., P("Given") = 0.15, P("You") = 0.12, P("Problem") = 0.08  │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│  5. TOKEN SELECTION                                                 │
│     Greedy: Pick argmax(probabilities) → "Given"                   │
│     Sampling: Sample from distribution → might get "You"           │
└─────────────────────────────────────────────────────────────────────┘
         ↓
    Repeat steps 2-5 for each new token until done
```

### Where We Inject Noise

We inject noise at **step 2**, right after the embedding layer, but **only on the first forward pass**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER OUTPUT (before noise)                              │
│                                                                     │
│  Token embeddings E:                                                │
│  ┌─────────────────────────────────────────┐                       │
│  │ [0.23, -0.11, 0.87, ..., 0.42]  ← "Write"                       │
│  │ [0.15, 0.33, -0.22, ..., 0.18]  ← "a"                           │
│  │ [0.44, -0.28, 0.15, ..., 0.67]  ← "programming"                 │
│  │ [0.31, 0.19, -0.41, ..., 0.23]  ← "problem"                     │
│  └─────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │   ADD NOISE     │
                    │   E' = E + N    │
                    └─────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│  EMBEDDING LAYER OUTPUT (after noise)                               │
│                                                                     │
│  Perturbed embeddings E':                                          │
│  ┌─────────────────────────────────────────┐                       │
│  │ [0.24, -0.10, 0.88, ..., 0.43]  ← slightly shifted              │
│  │ [0.16, 0.34, -0.21, ..., 0.19]  ← slightly shifted              │
│  │ [0.45, -0.27, 0.16, ..., 0.68]  ← slightly shifted              │
│  │ [0.32, 0.20, -0.40, ..., 0.24]  ← slightly shifted              │
│  └─────────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────┘
```

### The Noise Calculation

We use **Gaussian noise** scaled relative to the embedding magnitudes:

```python
# Step 1: Compute the average embedding norm
norms = embeddings.norm(dim=-1)           # L2 norm of each token embedding
mean_norm = norms.mean()                  # Average across all tokens
# e.g., if embeddings have magnitude ~50, mean_norm ≈ 50

# Step 2: Scale the noise magnitude
sigma = sigma_scale * mean_norm
# sigma_scale = 0.01 (configurable)
# sigma = 0.01 * 50 = 0.5

# Step 3: Generate and apply noise
noise = torch.randn(embeddings.shape) * sigma
perturbed = embeddings + noise
```

**Why scale by embedding norm?**

Different models have different embedding scales. A noise of magnitude 1.0 might be:
- Imperceptible for a model with embeddings of magnitude 100
- Catastrophic for a model with embeddings of magnitude 0.1

By scaling relative to the mean norm, we ensure consistent perturbation strength across models.

### Per-Sequence vs. Per-Token Noise

We support two noise modes:

**Per-Token Noise:**
```
Token 1: [embed] + [noise_1]  ← independent random noise
Token 2: [embed] + [noise_2]  ← independent random noise
Token 3: [embed] + [noise_3]  ← independent random noise
```

**Per-Sequence Noise (default):**
```
Token 1: [embed] + [noise]    ← SAME noise vector
Token 2: [embed] + [noise]    ← SAME noise vector
Token 3: [embed] + [noise]    ← SAME noise vector
```

**Per-sequence is preferred** because:
- It applies a consistent "directional push" to the entire input
- Encourages global plan shifts rather than local token-level chaos
- More analogous to how humans approach creative tasks differently

### The Single-Shot Mechanism

**Critical design choice:** We inject noise **exactly once**, then stop.

```
Forward Pass 1 (prompt processing):
    Embeddings → ADD NOISE → Transformer → Next token prediction

Forward Pass 2 (generating token 1):
    Embeddings → NO NOISE (pass through) → Transformer → Next token

Forward Pass 3 (generating token 2):
    Embeddings → NO NOISE (pass through) → Transformer → Next token

... and so on
```

**Implementation via PyTorch hook:**

```python
class SingleShotEmbeddingNoise:
    def __init__(self):
        self._has_injected = False

    def _embedding_hook(self, module, input, output):
        if not self._is_active:
            return output

        if self._has_injected:  # Already injected? Pass through
            return output

        self._has_injected = True  # Mark as done
        return output + self.generate_noise(output)
```

The hook is registered on the embedding layer and intercepts every forward pass. After the first injection, it becomes a no-op.

### Why Single-Shot Matters

If we injected noise at every forward pass:

```
Pass 1: prompt + noise₁ → hidden states
Pass 2: token₁ + noise₂ → hidden states (inconsistent with pass 1!)
Pass 3: token₂ + noise₃ → hidden states (inconsistent with pass 1 and 2!)
```

This creates an **inconsistent context**—the model's "memory" of earlier tokens would be corrupted. By injecting once, we:

1. Establish a consistent perturbed representation of the prompt
2. Let the model generate coherently from that perturbed starting point
3. Maintain internal consistency throughout generation

---

## Technical Workflow

### The Four Experimental Conditions

We compare four generation strategies:

```
┌─────────────────────────────────────────────────────────────────────┐
│  CONDITION A: DETERMINISTIC BASELINE                                │
│  ─────────────────────────────────────                              │
│  Temperature: 0 (greedy)                                            │
│  Noise: OFF                                                         │
│  Samples per prompt: 1                                              │
│                                                                     │
│  Purpose: Establish the "default" output for each prompt            │
│  Expected: Same output every time (fully deterministic)             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONDITION B: TEMPERATURE SAMPLING                                  │
│  ─────────────────────────────────────                              │
│  Temperature: 0.8                                                   │
│  Noise: OFF                                                         │
│  Samples per prompt: 10                                             │
│                                                                     │
│  Purpose: Traditional diversity via token-level randomness          │
│  Expected: Diverse but potentially less coherent outputs            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONDITION C: EMBEDDING NOISE                                       │
│  ─────────────────────────────────────                              │
│  Temperature: 0 (greedy)                                            │
│  Noise: ON (sigma_scale=0.01, per_sequence)                        │
│  Samples per prompt: 10                                             │
│                                                                     │
│  Purpose: Diversity via representation-level perturbation           │
│  Expected: Diverse AND coherent outputs                             │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  CONDITION D: TEMPERATURE + EMBEDDING NOISE (combined)              │
│  ─────────────────────────────────────                              │
│  Temperature: 0.8                                                   │
│  Noise: ON (sigma_scale=0.01, per_sequence)                        │
│  Samples per prompt: 10                                             │
│                                                                     │
│  Purpose: Test if combining both methods yields additive benefits   │
│  Expected: Maximum diversity, potentially trading off coherence     │
└─────────────────────────────────────────────────────────────────────┘
```

### Complete Inference Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           EXPERIMENT PIPELINE                            │
└──────────────────────────────────────────────────────────────────────────┘

Step 1: SETUP
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────┐     ┌─────────────────────┐     ┌────────────────────────┐
│ Load Model  │ ──→ │ Attach Noise Hook   │ ──→ │ Load 10 Prompts        │
│ (DeepSeek)  │     │ to Embedding Layer  │     │ (CP problem requests)  │
└─────────────┘     └─────────────────────┘     └────────────────────────┘

Step 2: CONDITION A (Deterministic Baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each prompt (10 total):
    ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
    │ Deactivate   │ ──→ │ Generate with   │ ──→ │ Save result      │
    │ noise hook   │     │ temp=0, greedy  │     │ A_{prompt}_0.txt │
    └──────────────┘     └─────────────────┘     └──────────────────┘

Output: 10 files (1 per prompt)

Step 3: CONDITION B (Temperature Sampling)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each prompt (10 total):
    For each sample (10 per prompt):
        ┌──────────────┐     ┌─────────────────┐     ┌──────────────────┐
        │ Deactivate   │ ──→ │ Generate with   │ ──→ │ Save result      │
        │ noise hook   │     │ temp=0.8        │     │ B_{p}_{s}.txt    │
        └──────────────┘     └─────────────────┘     └──────────────────┘

Output: 100 files (10 prompts × 10 samples)

Step 4: CONDITION C (Embedding Noise)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each prompt (10 total):
    For each sample (10 per prompt):
        ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
        │ Set seed     │ ──→ │ Activate     │ ──→ │ Generate with   │
        │ (sample_idx) │     │ noise hook   │     │ temp=0, greedy  │
        └──────────────┘     └──────────────┘     └─────────────────┘
                                    │
                                    ↓
        ┌──────────────┐     ┌──────────────────┐
        │ Deactivate   │ ←── │ Save result      │
        │ noise hook   │     │ C_{p}_{s}.txt    │
        └──────────────┘     └──────────────────┘

Output: 100 files (10 prompts × 10 samples)

Step 5: CONDITION D (Temperature + Embedding Noise)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each prompt (10 total):
    For each sample (10 per prompt):
        ┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
        │ Set seed     │ ──→ │ Activate     │ ──→ │ Generate with   │
        │ (sample_idx) │     │ noise hook   │     │ temp=0.8        │
        └──────────────┘     └──────────────┘     └─────────────────┘
                                   │
                                   ↓
        ┌──────────────┐     ┌──────────────────┐
        │ Deactivate   │ ←── │ Save result      │
        │ noise hook   │     │ D_{p}_{s}.txt    │
        └──────────────┘     └──────────────────┘

Output: 100 files (10 prompts × 10 samples)

Step 6: CONSOLIDATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
┌─────────────────────────────────────────────────────────────────────────┐
│ all_results.json                                                        │
│ ├── metadata (model, params, etc.)                                      │
│ └── results[]                                                           │
│     ├── {condition: "A", prompt_idx: 0, sample_idx: 0, text: "..."}    │
│     ├── {condition: "B", prompt_idx: 0, sample_idx: 0, text: "..."}    │
│     ├── {condition: "C", prompt_idx: 0, sample_idx: 0, text: "..."}    │
│     ├── ...                                                             │
│     └── {condition: "D", prompt_idx: 9, sample_idx: 9, text: "..."}    │
└─────────────────────────────────────────────────────────────────────────┘

Total outputs: 10 (A) + 100 (B) + 100 (C) + 100 (D) = 310
```

### Evaluation Pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PIPELINE                             │
└──────────────────────────────────────────────────────────────────────────┘

Option 1: BASIC METRICS (requires manual annotation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

all_results.json
      │
      ↓
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────────────┐
│ Create template │ ──→ │ Human annotates │ ──→ │ Compute metrics         │
│ annotations.csv │     │ valid/alg_class │     │ - Validity@k            │
└─────────────────┘     └─────────────────┘     │ - AlgDiversity@k        │
                                                │ - SigDiversity@k        │
                                                └─────────────────────────┘

Option 2: LLM JUDGE (automated)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

all_results.json
      │
      ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ For each output:                                                        │
│   ┌───────────────┐     ┌─────────────────┐     ┌───────────────────┐  │
│   │ Send to LLM   │ ──→ │ Parse JSON      │ ──→ │ Aggregate scores  │  │
│   │ (Groq/Gemini) │     │ response        │     │ by condition      │  │
│   └───────────────┘     └─────────────────┘     └───────────────────┘  │
│                                                                         │
│ Scores per output:                                                      │
│   - Creativity (1-10)                                                   │
│   - Validity (1-10 + PASS/FAIL)                                        │
└─────────────────────────────────────────────────────────────────────────┘
      │
      ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ Summary comparison:                                                     │
│                                                                         │
│ Condition B (temp=0.8):                                                │
│   Creativity: mean=5.2, Validity: mean=6.8, Pass rate: 72%             │
│                                                                         │
│ Condition C (noise):                                                    │
│   Creativity: mean=5.8, Validity: mean=7.4, Pass rate: 85%             │
│                                                                         │
│ Condition D (temp+noise):                                               │
│   Creativity: mean=6.1, Validity: mean=7.1, Pass rate: 78%             │
│                                                                         │
│ Pairwise comparisons:                                                   │
│   B vs C: Δ Creativity +0.6, Δ Validity +0.6, Δ Pass rate +13%         │
│   B vs D: Δ Creativity +0.9, Δ Validity +0.3, Δ Pass rate +6%          │
│   C vs D: Δ Creativity +0.3, Δ Validity -0.3, Δ Pass rate -7%          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Why This Should Work (Theory)

### 1. The Embedding Space is Smooth

Neural network embedding spaces are generally **smooth**—nearby points represent semantically similar concepts. A small perturbation shouldn't teleport you to a completely unrelated region.

```
                    ┌─────────────────────────────────────┐
                    │     EMBEDDING SPACE (simplified)    │
                    │                                     │
                    │    "sorting"    "searching"         │
                    │         ●───────────●               │
                    │        /             \              │
                    │   "arrays"         "graphs"         │
                    │       ●               ●             │
                    │        \             /              │
                    │         ●───────────●               │
                    │    "efficient"  "optimal"           │
                    │                                     │
                    │  Small noise: move within cluster   │
                    │  Large noise: jump between clusters │
                    └─────────────────────────────────────┘
```

With `sigma_scale=0.01` (1% of embedding magnitude), we stay in the "local neighborhood" but explore different facets of the same general concept.

### 2. Initial Context Shapes Everything

Language models are **autoregressive**—each token depends on all previous tokens. The initial hidden state (shaped by the prompt embedding) acts as a "seed" that influences the entire generation trajectory.

```
Original embedding:           Perturbed embedding:
       │                             │
       ↓                             ↓
   ┌───────┐                    ┌───────┐
   │ Sort  │                    │ Graph │
   │ array │                    │ color │
   └───┬───┘                    └───┬───┘
       │                             │
       ↓                             ↓
   ┌───────┐                    ┌───────┐
   │ Quick │                    │ BFS   │
   │ sort  │                    │ search│
   └───┬───┘                    └───┬───┘
       │                             │
       ↓                             ↓
     ...                           ...
```

A tiny initial difference can lead to a qualitatively different output—this is related to the concept of **sensitive dependence on initial conditions** (chaos theory, but in embedding space).

### 3. Diversity Without Incoherence

Temperature sampling introduces randomness at every token:

```
Token 1: P("The") = 0.3 → sample → "A"       (unexpected)
Token 2: P("cat") = 0.2 → sample → "dog"     (unexpected)
Token 3: P("sat") = 0.4 → sample → "ran"     (unexpected)

Result: "A dog ran..." (each token is locally reasonable but globally less coherent)
```

Embedding noise introduces randomness once:

```
Perturbed prompt → Model "decides" on a direction
Token 1: P("The") = 0.5 → greedy → "The"     (most likely given perturbed prompt)
Token 2: P("algorithm") = 0.4 → greedy → "algorithm"
Token 3: P("uses") = 0.6 → greedy → "uses"

Result: "The algorithm uses..." (coherent execution of a different plan)
```

### 4. Theoretical Intuition: The Lottery Ticket Hypothesis

One way to think about this: a well-trained language model has learned many "good solutions" to a given prompt. Temperature sampling explores this space by randomly wandering. Embedding noise explores it by nudging the model toward a different starting point, then letting it find the locally optimal path from there.

```
                    SOLUTION LANDSCAPE

           ╱╲      ╱╲      ╱╲
          ╱  ╲    ╱  ╲    ╱  ╲
         ╱    ╲  ╱    ╲  ╱    ╲
        ╱      ╲╱      ╲╱      ╲
       A        B        C        D
       ↑        ↑        ↑        ↑
    solution  solution solution solution
       1        2        3        4

Temperature: Random walk across landscape (might fall into valleys)
Noise: Jump to different peak, then greedy descent (reaches local optimum)
```

### 5. Why Competitive Programming Problems?

We test on **competitive programming problem generation** because:

1. **Verifiable quality**: A problem is either valid or not (clear success criteria)
2. **Structured output**: Problems have consistent format (statement, constraints, solution)
3. **Diversity is measurable**: Different algorithm classes, different constraint structures
4. **Real-world relevance**: Generating training data, educational content, benchmarks

---

## Usage Guide

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd <repo-name>

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install torch transformers
pip install openai  # for Groq judge
pip install google-genai  # for Gemini judge
```

### Running the Experiment

```bash
# Run the full experiment (A, B, C, D conditions)
python core/experiment.py

# This will:
# 1. Load the model (default: Llama-3.1-8B)
# 2. Run all four conditions
# 3. Save outputs to core/outputs/
```

### Configuration

Edit `core/config.py` to customize:

```python
# Model
MODEL_NAME = "deepseek-ai/deepseek-coder-6.7b-base"

# Experiment parameters
K_SAMPLES = 10          # Samples per prompt for B, C, and D
TEMPERATURE = 0.8       # For conditions B and D
SIGMA_SCALE = 0.01      # Noise magnitude for conditions C and D
NOISE_SCOPE = "per_sequence"  # or "per_token"

# Generation
MAX_NEW_TOKENS = 1024
MIN_NEW_TOKENS = 64
```

### Running Evaluation

```bash
# Using LLM judge (Groq)
export GROQ_API_KEY="your-api-key"
python core/evaluate.py judge groq

# Using LLM judge (Gemini)
export GEMINI_API_KEY="your-api-key"
python core/evaluate.py judge gemini

# Limit samples per condition
python core/evaluate.py judge groq --n-per-condition 5

# Compute metrics from manual annotations
python core/evaluate.py metrics
```

---

## File Structure

```
.
├── README.md              # This file
├── core/
│   ├── __init__.py        # Package marker
│   ├── config.py          # All experiment configuration
│   ├── prompts.py         # 10 CP problem generation prompts
│   ├── experiment.py      # Main experiment (model loading, noise, inference)
│   └── evaluate.py        # Evaluation (metrics, LLM judges)
└── core/outputs/          # Generated outputs (created at runtime)
    ├── A_0_0.txt          # Condition A outputs
    ├── B_0_0.txt          # Condition B outputs
    ├── C_0_0.txt          # Condition C outputs
    ├── D_0_0.txt          # Condition D outputs
    └── all_results.json   # Consolidated results
```

---

## Expected Results

Based on the theory, we expect:

| Metric | Condition B (temp) | Condition C (noise) | Condition D (temp+noise) | Interpretation |
|--------|-------------------|---------------------|-------------------------|----------------|
| Validity | Lower | Higher | Medium | D trades some coherence for diversity |
| Diversity | High | High | Highest | Combined methods maximize exploration |
| Creativity | Moderate | Moderate-High | Highest | Dual perturbation explores more space |

**Key hypotheses**:
1. Condition C should achieve comparable diversity to Condition B while maintaining higher validity/coherence.
2. Condition D tests whether combining temperature sampling AND embedding noise yields additive creativity benefits.
3. Condition D may show increased diversity at the cost of some validity compared to C alone.

---

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{embedding_noise_2024,
  title={Embedding-Level Noise for Creative Generation},
  year={2024},
  url={<your-repo-url>}
}
```

---

## License

MIT License - see LICENSE file for details.
