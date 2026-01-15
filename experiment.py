#!/usr/bin/env python3
# File experiment.py
"""
Main experiment script for embedding noise experiments.

Consolidates: model loading, noise injection, text generation, and inference orchestration.

Usage:
    python core/experiment.py
    python core/experiment.py --num_prompts 10 --num_generations 100
    python core/experiment.py --num_prompts 35 --num_generations 50
    python core/experiment.py --num_generations 100 --num_prompts 10 --model_name 'meta-llama/Llama-3.1-8B'
    python core/experiment.py --gpu 0 --model_name 'deepseek-ai/deepseek-coder-6.7b-base'
    python core/experiment.py --gpu 1 --num_generations 50 --model_name 'meta-llama/Llama-3.1-8B'
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================================
# Model Loading
# ============================================================================

def load_model(
    model_name: str,
    device_map: dict | str = "auto",
    torch_dtype: torch.dtype = torch.float16,
    load_in_8bit: bool = False,
):
    """
    Load a causal LM with optional quantization.

    Args:
        model_name: HuggingFace model identifier
        device_map: Device placement strategy
        torch_dtype: Model dtype (float16 for inference)
        load_in_8bit: Whether to use int8 quantization

    Returns:
        (model, tokenizer) tuple
    """
    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "device_map": device_map,# None, # # TODO: Local system logic, swap to device_map before pushing
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    #model = model.to("cuda:0") # TODO: Local system logic, comment out before pushing
    model.eval()

    # Check if this is a chat model
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None
    model_type = "chat model (using chat template)" if has_chat_template else "base model (direct text)"

    print(f"Model loaded. Device: {next(model.parameters()).device}")
    print(f"Model type: {model_type}")
    return model, tokenizer


# ============================================================================
# Single-Shot Embedding Noise
# ============================================================================

class SingleShotEmbeddingNoise:
    """
    Injects Gaussian noise into embedding layer ONCE per generation.

    Unlike continuous noise injection, this version:
    1. Injects noise only on the first forward pass (prompt embeddings)
    2. Passes through without modification on subsequent forward passes
    3. Must be reset between generations via activate()

    This isolates the effect of representation-level stochasticity
    to the initial planning phase, not token-by-token decoding.
    """

    def __init__(
        self,
        sigma_scale: float = 0.01,
        noise_scope: Literal["per_token", "per_sequence"] = "per_sequence",
        seed: Optional[int] = None,
    ):
        """
        Args:
            sigma_scale: Noise std as fraction of mean embedding norm
            noise_scope: "per_token" or "per_sequence" (shared noise vector)
            seed: Random seed for reproducibility
        """
        self.sigma_scale = sigma_scale
        self.noise_scope = noise_scope
        self.seed = seed
        self._generator: Optional[torch.Generator] = None
        self._hook_handle = None
        self._is_active = False
        self._has_injected = False

    def _get_generator(self, device: torch.device) -> torch.Generator:
        """Get or create random generator for reproducibility."""
        if self._generator is None or self._generator.device != device:
            self._generator = torch.Generator(device=device)
            if self.seed is not None:
                self._generator.manual_seed(self.seed)
        return self._generator

    def compute_sigma(self, embeddings: torch.Tensor) -> float:
        """Compute σ = sigma_scale × E[||E(x_i)||_2]"""
        norms = embeddings.norm(dim=-1)
        mean_norm = norms.mean().item()
        return self.sigma_scale * mean_norm

    def inject_noise(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to embeddings (ONLY on first call per generation)."""
        if not self._is_active:
            return embeddings

        if self._has_injected:
            return embeddings

        self._has_injected = True
        sigma = self.compute_sigma(embeddings)
        generator = self._get_generator(embeddings.device)

        if self.noise_scope == "per_token":
            noise = torch.randn(
                embeddings.shape,
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            ) * sigma
        else:  # per_sequence
            batch_size, seq_len, embed_dim = embeddings.shape
            noise_vec = torch.randn(
                (batch_size, 1, embed_dim),
                device=embeddings.device,
                dtype=embeddings.dtype,
                generator=generator,
            ) * sigma
            noise = noise_vec.expand(-1, seq_len, -1)

        return embeddings + noise

    def _embedding_hook(self, module, input, output):
        """Forward hook to inject noise after embedding layer."""
        return self.inject_noise(output)

    def attach_to_model(self, model) -> None:
        """Attach noise injection hook to model's embedding layer."""
        embed_layer = None

        # Try common patterns
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            embed_layer = model.model.embed_tokens
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            embed_layer = model.transformer.wte
        elif hasattr(model, 'model') and hasattr(model.model, 'embeddings'):
            embed_layer = model.model.embeddings
        else:
            for name, module in model.named_modules():
                if 'embed' in name.lower() and 'token' in name.lower():
                    embed_layer = module
                    break

        if embed_layer is None:
            raise ValueError("Could not find embedding layer in model")

        self._hook_handle = embed_layer.register_forward_hook(self._embedding_hook)
        print(f"Attached single-shot noise hook to: {type(embed_layer).__name__}")

    def detach(self) -> None:
        """Remove the hook from the model."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def activate(self) -> None:
        """Enable noise injection and reset injection state."""
        self._is_active = True
        self._has_injected = False

    def deactivate(self) -> None:
        """Disable noise injection."""
        self._is_active = False

    def set_seed(self, seed: int) -> None:
        """Update seed for reproducible noise."""
        self.seed = seed
        self._generator = None


# ============================================================================
# Text Generation
# ============================================================================

def _seed_everything(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_text(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    do_sample: bool,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate text from a prompt with consistent decoding settings.

    Supports both base models (direct text) and chat models (with chat template).

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        max_new_tokens: Maximum tokens to generate
        min_new_tokens: Minimum tokens (prevents empty outputs)
        do_sample: Whether to use sampling
        temperature: Sampling temperature (if do_sample=True)
        top_p: Nucleus sampling threshold (if do_sample=True)
        seed: Random seed for reproducibility

    Returns:
        Generated text (excluding the prompt)
    """
    if seed is not None:
        _seed_everything(seed)

    # Check if tokenizer has chat template support
    has_chat_template = hasattr(tokenizer, 'chat_template') and tokenizer.chat_template is not None

    if has_chat_template:
        # Use chat template format for chat models (e.g., Falcon-H1R-7B)
        messages = [{"role": "user", "content": prompt}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt"
            )
            # apply_chat_template returns input_ids directly, wrap it
            inputs = {"input_ids": inputs.to(model.device)}
        except Exception as e:
            # Fallback to direct tokenization if chat template fails
            print(f"Warning: Chat template failed ({e}), using direct tokenization")
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    else:
        # Direct tokenization for base models
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.pad_token_id,
    )

    if do_sample:
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
    else:
        # Override model's default generation_config to suppress warnings
        gen_kwargs["temperature"] = None
        gen_kwargs["top_p"] = None
        gen_kwargs["top_k"] = None

    outputs = model.generate(**gen_kwargs)

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ============================================================================
# Generation Result
# ============================================================================

@dataclass
class GenerationResult:
    """Stores a single generation result with metadata."""
    condition: str  # "A", "B", or "C"
    prompt_idx: int
    sample_idx: int
    prompt_text: str
    generated_text: str
    timestamp: str
    seed: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    def save_to_file(self, filepath: Path) -> None:
        """Save individual result to text file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"# Condition {self.condition}\n")
            f.write(f"# Prompt {self.prompt_idx}, Sample {self.sample_idx}\n")
            f.write(f"# Timestamp: {self.timestamp}\n")
            if self.seed is not None:
                f.write(f"# Seed: {self.seed}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"PROMPT:\n{self.prompt_text}\n")
            f.write(f"\n{'='*80}\n")
            f.write(f"GENERATED OUTPUT:\n{self.generated_text}\n")


# ============================================================================
# Inference Orchestrator
# ============================================================================

class SanityCheckInference:
    """
    Orchestrates the three experimental conditions:
    - Condition A: Deterministic baseline (temp=0, no noise)
    - Condition B: Temperature sampling (temp>0, no noise)
    - Condition C: Embedding noise (temp=0, with noise)
    """

    def __init__(self, config, prompts_list: List[str], checkpoint_path: Optional[Path] = None):
        """
        Args:
            config: Configuration module with settings
            prompts_list: List of prompt strings to use
            checkpoint_path: Path to checkpoint file for resume functionality
        """
        self.config = config
        self.prompts = prompts_list
        self.model = None
        self.tokenizer = None
        self.noise_injector = None
        self.results: List[GenerationResult] = []
        self.checkpoint_path = checkpoint_path
        self.checkpoint_state = {
            'condition_a_complete': False,
            'condition_b_progress': 0,  # Number of generations completed
            'condition_c_progress': 0,
            'condition_d_progress': 0,
            'results': [],
        }
        self.checkpoint_interval = 10  # Save every 10 generations

    def save_checkpoint(self) -> None:
        """Save current progress to checkpoint file."""
        if self.checkpoint_path is None:
            return

        self.checkpoint_state['results'] = [r.to_dict() for r in self.results]

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.checkpoint_path.with_suffix('.tmp')

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(self.checkpoint_state, f, indent=2, ensure_ascii=False)

        # Atomic rename
        temp_path.replace(self.checkpoint_path)

    def load_checkpoint(self) -> bool:
        """
        Load checkpoint from file.

        Returns:
            True if checkpoint was loaded, False if no checkpoint exists
        """
        if self.checkpoint_path is None or not self.checkpoint_path.exists():
            return False

        with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
            self.checkpoint_state = json.load(f)

        # Restore results
        self.results = [
            GenerationResult(**r) for r in self.checkpoint_state['results']
        ]

        return True

    def print_checkpoint_diagnostics(self) -> None:
        """Print diagnostic information about checkpoint progress."""
        print("\n" + "=" * 80)
        print("CHECKPOINT DIAGNOSTICS")
        print("=" * 80)

        total_a = len(self.prompts)
        total_bcd = len(self.prompts) * self.config.K_SAMPLES

        print(f"\nCondition A: {'COMPLETE' if self.checkpoint_state['condition_a_complete'] else 'NOT STARTED'}")
        if self.checkpoint_state['condition_a_complete']:
            print(f"  Generated: {total_a}/{total_a} outputs")

        print(f"\nCondition B: {self.checkpoint_state['condition_b_progress']}/{total_bcd} generations")
        if total_bcd > 0:
            pct_b = (self.checkpoint_state['condition_b_progress'] / total_bcd) * 100
            print(f"  Progress: {pct_b:.1f}%")

        print(f"\nCondition C: {self.checkpoint_state['condition_c_progress']}/{total_bcd} generations")
        if total_bcd > 0:
            pct_c = (self.checkpoint_state['condition_c_progress'] / total_bcd) * 100
            print(f"  Progress: {pct_c:.1f}%")

        print(f"\nCondition D: {self.checkpoint_state['condition_d_progress']}/{total_bcd} generations")
        if total_bcd > 0:
            pct_d = (self.checkpoint_state['condition_d_progress'] / total_bcd) * 100
            print(f"  Progress: {pct_d:.1f}%")

        total_progress = (
            (total_a if self.checkpoint_state['condition_a_complete'] else 0) +
            self.checkpoint_state['condition_b_progress'] +
            self.checkpoint_state['condition_c_progress'] +
            self.checkpoint_state['condition_d_progress']
        )
        total_needed = total_a + (3 * total_bcd)

        print(f"\nOverall Progress: {total_progress}/{total_needed} ({(total_progress/total_needed)*100:.1f}%)")
        print(f"Results collected: {len(self.results)}")
        print("=" * 80 + "\n")

    def setup(self) -> None:
        """Load model and prepare noise injector."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SETUP")
        print("=" * 80)

        torch_dtype = getattr(torch, self.config.TORCH_DTYPE)
        self.model, self.tokenizer = load_model(
            model_name=self.config.MODEL_NAME,
            device_map=self.config.DEVICE_MAP,
            torch_dtype=torch_dtype,
            load_in_8bit=self.config.LOAD_IN_8BIT,
        )

        print(f"\nSetting up single-shot embedding noise injector")
        print(f"  Sigma scale: {self.config.SIGMA_SCALE}")
        print(f"  Noise scope: {self.config.NOISE_SCOPE}")

        self.noise_injector = SingleShotEmbeddingNoise(
            sigma_scale=self.config.SIGMA_SCALE,
            noise_scope=self.config.NOISE_SCOPE,
        )
        self.noise_injector.attach_to_model(self.model)

        print("\nSetup complete")

    def run_condition_a(self) -> List[GenerationResult]:
        """Condition A: Deterministic Baseline (temp=0, no noise, one sample per prompt)."""
        # Skip if already complete
        if self.checkpoint_state['condition_a_complete']:
            print("\n" + "=" * 80)
            print("CONDITION A: ALREADY COMPLETE (skipping)")
            print("=" * 80)
            return []

        print("\n" + "=" * 80)
        print("CONDITION A: DETERMINISTIC BASELINE")
        print("=" * 80)
        print(f"Settings: greedy decoding (temp=0), 1 sample/prompt (deterministic), seed=prompt_idx")
        print(f"Note: Generating only {len(self.prompts)} outputs (one per prompt) since greedy is deterministic")

        results = []

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)} (seed={prompt_idx})")

            self.noise_injector.deactivate()

            # Generate only ONE sample per prompt (greedy is deterministic)
            if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                print(f"  Generating...", end=" ")

            generated = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                do_sample=False,
                temperature=None,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                min_new_tokens=self.config.MIN_NEW_TOKENS,
                seed=prompt_idx,
            )

            result = GenerationResult(
                condition="A",
                prompt_idx=prompt_idx,
                sample_idx=0,  # Always 0 since we only generate one
                prompt_text=prompt_text,
                generated_text=generated,
                timestamp=datetime.now().isoformat(),
                seed=prompt_idx,
            )
            results.append(result)

            if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                print(f"{len(generated)} chars")

        # Mark as complete and save checkpoint
        self.checkpoint_state['condition_a_complete'] = True
        self.save_checkpoint()

        print(f"\nCondition A complete: {len(results)} outputs")
        return results

    def run_condition_b(self) -> List[GenerationResult]:
        """Condition B: Temperature Sampling (temp>0, no noise, k samples)."""
        print("\n" + "=" * 80)
        print("CONDITION B: TEMPERATURE SAMPLING")
        print("=" * 80)
        print(f"Settings: temperature={self.config.TEMPERATURE}, {self.config.K_SAMPLES} samples/prompt")

        total_needed = len(self.prompts) * self.config.K_SAMPLES
        start_from = self.checkpoint_state['condition_b_progress']

        if start_from > 0:
            print(f"Resuming from generation {start_from}/{total_needed}")

        results = []
        generation_count = start_from

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")

            self.noise_injector.deactivate()

            for sample_idx in range(self.config.K_SAMPLES):
                # Skip if already generated
                if generation_count < start_from:
                    generation_count += 1
                    continue

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"  Sample {sample_idx + 1}/{self.config.K_SAMPLES}...", end=" ")

                generated = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_text,
                    do_sample=True,
                    temperature=self.config.TEMPERATURE,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    min_new_tokens=self.config.MIN_NEW_TOKENS,
                    seed=sample_idx,
                )

                result = GenerationResult(
                    condition="B",
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    prompt_text=prompt_text,
                    generated_text=generated,
                    timestamp=datetime.now().isoformat(),
                )
                results.append(result)
                generation_count += 1

                # Update checkpoint every 10 generations
                self.checkpoint_state['condition_b_progress'] = generation_count
                if generation_count % self.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")

        # Final checkpoint save
        self.checkpoint_state['condition_b_progress'] = generation_count
        self.save_checkpoint()

        print(f"\nCondition B complete: {len(results)} new outputs (total: {generation_count})")
        return results

    def run_condition_c(self) -> List[GenerationResult]:
        """Condition C: Embedding Noise (temp=0, noise, k samples)."""
        print("\n" + "=" * 80)
        print("CONDITION C: EMBEDDING NOISE")
        print("=" * 80)
        print(f"Settings: sigma={self.config.SIGMA_SCALE}, {self.config.K_SAMPLES} samples/prompt")

        total_needed = len(self.prompts) * self.config.K_SAMPLES
        start_from = self.checkpoint_state['condition_c_progress']

        if start_from > 0:
            print(f"Resuming from generation {start_from}/{total_needed}")

        results = []
        generation_count = start_from

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")

            for sample_idx in range(self.config.K_SAMPLES):
                # Skip if already generated
                if generation_count < start_from:
                    generation_count += 1
                    continue

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"  Sample {sample_idx + 1}/{self.config.K_SAMPLES} (seed={sample_idx})...", end=" ")

                self.noise_injector.set_seed(sample_idx)
                self.noise_injector.activate()

                generated = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_text,
                    do_sample=False,
                    temperature=None,
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    min_new_tokens=self.config.MIN_NEW_TOKENS,
                )

                self.noise_injector.deactivate()

                result = GenerationResult(
                    condition="C",
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    prompt_text=prompt_text,
                    generated_text=generated,
                    timestamp=datetime.now().isoformat(),
                    seed=sample_idx,
                )
                results.append(result)
                generation_count += 1

                # Update checkpoint every 10 generations
                self.checkpoint_state['condition_c_progress'] = generation_count
                if generation_count % self.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")

        # Final checkpoint save
        self.checkpoint_state['condition_c_progress'] = generation_count
        self.save_checkpoint()

        print(f"\nCondition C complete: {len(results)} new outputs (total: {generation_count})")
        return results

    def run_condition_d(self) -> List[GenerationResult]:
        """Condition D: Temperature + Embedding Noise (combined)."""
        print("\n" + "=" * 80)
        print("CONDITION D: TEMPERATURE + EMBEDDING NOISE")
        print("=" * 80)
        print(f"Settings: temp={self.config.TEMPERATURE}, sigma={self.config.SIGMA_SCALE}, {self.config.K_SAMPLES} samples/prompt")

        total_needed = len(self.prompts) * self.config.K_SAMPLES
        start_from = self.checkpoint_state['condition_d_progress']

        if start_from > 0:
            print(f"Resuming from generation {start_from}/{total_needed}")

        results = []
        generation_count = start_from

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")

            for sample_idx in range(self.config.K_SAMPLES):
                # Skip if already generated
                if generation_count < start_from:
                    generation_count += 1
                    continue

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"  Sample {sample_idx + 1}/{self.config.K_SAMPLES} (seed={sample_idx})...", end=" ")

                # Activate embedding noise
                self.noise_injector.set_seed(sample_idx)
                self.noise_injector.activate()

                # Generate with BOTH temperature sampling AND embedding noise
                generated = generate_text(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=prompt_text,
                    do_sample=True,  # Enable sampling
                    temperature=self.config.TEMPERATURE,  # Use temperature
                    max_new_tokens=self.config.MAX_NEW_TOKENS,
                    min_new_tokens=self.config.MIN_NEW_TOKENS,
                    seed=sample_idx,  # For reproducibility of sampling
                )

                self.noise_injector.deactivate()

                result = GenerationResult(
                    condition="D",
                    prompt_idx=prompt_idx,
                    sample_idx=sample_idx,
                    prompt_text=prompt_text,
                    generated_text=generated,
                    timestamp=datetime.now().isoformat(),
                    seed=sample_idx,
                )
                results.append(result)
                generation_count += 1

                # Update checkpoint every 10 generations
                self.checkpoint_state['condition_d_progress'] = generation_count
                if generation_count % self.checkpoint_interval == 0:
                    self.save_checkpoint()

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")

        # Final checkpoint save
        self.checkpoint_state['condition_d_progress'] = generation_count
        self.save_checkpoint()

        print(f"\nCondition D complete: {len(results)} new outputs (total: {generation_count})")
        return results

    def run_all_conditions(self) -> List[GenerationResult]:
        """Run all four conditions in sequence."""
        print("\n" + "=" * 80)
        print("RUNNING ALL CONDITIONS")
        print("=" * 80)
        print(f"Total prompts: {len(self.prompts)}")
        print(f"Expected outputs:")
        print(f"  Condition A: {len(self.prompts)} (deterministic baseline - 1 per prompt)")
        print(f"  Condition B: {len(self.prompts) * self.config.K_SAMPLES} (temperature only - {self.config.K_SAMPLES} per prompt)")
        print(f"  Condition C: {len(self.prompts) * self.config.K_SAMPLES} (noise only - {self.config.K_SAMPLES} per prompt)")
        print(f"  Condition D: {len(self.prompts) * self.config.K_SAMPLES} (temperature + noise - {self.config.K_SAMPLES} per prompt)")
        total = len(self.prompts) * (1 + 3 * self.config.K_SAMPLES)
        print(f"  Total: {total}")

        all_results = []
        all_results.extend(self.run_condition_a())
        all_results.extend(self.run_condition_b())
        all_results.extend(self.run_condition_c())
        all_results.extend(self.run_condition_d())

        self.results = all_results
        return all_results

    def save_results(self, output_dir: Path) -> None:
        """Save all results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        print(f"Output directory: {output_dir}")

        if self.config.SAVE_INDIVIDUAL_FILES:
            print("\nSaving individual output files...")
            for result in self.results:
                filename = f"{result.condition}_{result.prompt_idx}_{result.sample_idx}.txt"
                filepath = output_dir / filename
                result.save_to_file(filepath)
            print(f"  Saved {len(self.results)} individual files")

        if self.config.SAVE_SUMMARY_JSON:
            print("\nSaving consolidated JSON...")
            json_path = output_dir / "all_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {
                        'metadata': {
                            'model': self.config.MODEL_NAME,
                            'n_prompts': len(self.prompts),
                            'k_samples': self.config.K_SAMPLES,
                            'temperature': self.config.TEMPERATURE,
                            'sigma_scale': self.config.SIGMA_SCALE,
                            'noise_scope': self.config.NOISE_SCOPE,
                            'total_outputs': len(self.results),
                        },
                        'results': [r.to_dict() for r in self.results],
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            print(f"  Saved {json_path}")

        print("\nAll results saved")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run the complete experiment."""
    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import config
    from prompts import get_all_prompts

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run embedding noise experiments")
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=config.N_PROMPTS,
        help=f"Number of prompts to use (default: {config.N_PROMPTS}, max: 35 for Codeforces, 20 for LeetCode, 10 for Riddle/Math)"
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=config.K_SAMPLES,
        help=f"Number of generations per prompt per condition (default: {config.K_SAMPLES})"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=config.MODEL_NAME,
        help=f"HuggingFace model name (default: {config.MODEL_NAME})"
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device ID to use (e.g., '0', '1', '0,1' for multiple GPUs). If not specified, uses all available GPUs."
    )
    parser.add_argument(
        "--start_from_beginning",
        action="store_true",
        help="Overwrite existing checkpoint and start from beginning"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (shows progress diagnostics)"
    )

    # Platform selection (mutually exclusive)
    platform_group = parser.add_mutually_exclusive_group()
    platform_group.add_argument(
        "--codeforces",
        action="store_true",
        help="Use Codeforces Div-2 B prompts (default)"
    )
    platform_group.add_argument(
        "--leetcode",
        action="store_true",
        help="Use LeetCode Medium-Hard prompts"
    )
    platform_group.add_argument(
        "--riddle",
        action="store_true",
        help="Use TED-Ed style riddle prompts (CF-aligned philosophy)"
    )
    platform_group.add_argument(
        "--math",
        action="store_true",
        help="Use grade school math word problem prompts (CF-aligned philosophy)"
    )
    args = parser.parse_args()

    # Validate mutually exclusive arguments
    if args.start_from_beginning and args.resume:
        parser.error("--start_from_beginning and --resume are mutually exclusive")

    # Determine platform (default to codeforces)
    if args.leetcode:
        platform = "leetcode"
    elif args.riddle:
        platform = "riddle"
    elif args.math:
        platform = "math"
    else:
        platform = "codeforces"

    # Set GPU visibility if specified
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Setting CUDA_VISIBLE_DEVICES={args.gpu}")

    # Override config with CLI arguments
    num_prompts = args.num_prompts
    num_generations = args.num_generations
    model_name = args.model_name

    # Update config
    config.K_SAMPLES = num_generations
    config.MODEL_NAME = model_name

    # Create output directory with model name, platform, and generation count
    model_short_name = model_name.split('/')[-1]  # e.g., "Llama-3.1-8B" from "meta-llama/Llama-3.1-8B"
    output_dir = Path(config.OUTPUT_DIR) / model_short_name / platform / f"k{num_generations}"

    # Calculate total inferences
    # Condition A: num_prompts (1 per prompt)
    # Conditions B, C, D: num_prompts * num_generations each
    total_inferences = num_prompts + (num_prompts * num_generations * 3)

    # Get platform display name
    platform_display = {
        "codeforces": "Codeforces Div-2 B",
        "leetcode": "LeetCode Medium-Hard",
        "riddle": "TED-Ed Style Riddles",
        "math": "Grade School Math",
    }[platform]

    print("""
================================================================================
                        EMBEDDING NOISE EXPERIMENT
================================================================================
""")

    print("Configuration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Platform: {platform_display}")
    print(f"  GPU: {args.gpu if args.gpu else 'all available'}")
    print(f"  Prompts: {num_prompts}")
    print(f"  Samples per condition:")
    print(f"    Condition A (greedy): 1 per prompt = {num_prompts} total")
    print(f"    Conditions B/C/D: {num_generations} per prompt = {num_prompts * num_generations} each")
    print(f"  Total inferences: {total_inferences}")
    print(f"  Temperature (B, D): {config.TEMPERATURE}")
    print(f"  Sigma scale (C, D): {config.SIGMA_SCALE}")
    print(f"  Output directory: {output_dir}")
    print(f"\nConditions:")
    print(f"  A: Deterministic baseline (temp=0, no noise, 1 sample/prompt)")
    print(f"  B: Temperature sampling only (temp={config.TEMPERATURE}, {num_generations} samples/prompt)")
    print(f"  C: Embedding noise only (sigma={config.SIGMA_SCALE}, {num_generations} samples/prompt)")
    print(f"  D: Temperature + Noise combined (temp={config.TEMPERATURE}, sigma={config.SIGMA_SCALE}, {num_generations} samples/prompt)")

    print("\n" + "=" * 80)
    response = input("Proceed with experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return

    prompts = get_all_prompts(platform)[:num_prompts]

    # Setup checkpoint path
    checkpoint_path = output_dir / "checkpoint.json"

    # Handle --start_from_beginning flag
    if args.start_from_beginning:
        if checkpoint_path.exists():
            print(f"\nDeleting existing checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()
        print("Starting from beginning (no checkpoint)")

    # Create inference with checkpoint support
    inference = SanityCheckInference(config, prompts, checkpoint_path=checkpoint_path)

    # Handle --resume flag
    if args.resume:
        if inference.load_checkpoint():
            print(f"\nLoaded checkpoint from: {checkpoint_path}")
            inference.print_checkpoint_diagnostics()
        else:
            print(f"\nNo checkpoint found at: {checkpoint_path}")
            print("Starting from beginning")

    inference.setup()

    results = inference.run_all_conditions()
    inference.save_results(output_dir)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(results)} total outputs:")
    print(f"  Condition A (deterministic):    {sum(1 for r in results if r.condition == 'A')}")
    print(f"  Condition B (temperature):      {sum(1 for r in results if r.condition == 'B')}")
    print(f"  Condition C (noise):            {sum(1 for r in results if r.condition == 'C')}")
    print(f"  Condition D (temp + noise):     {sum(1 for r in results if r.condition == 'D')}")
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()


""" 
Usage examples:

uv run experiment.py --gpu 0 --model_name 'deepseek-ai/deepseek-coder-6.7b-base' --num_generations 50 --num_prompts 10
"""