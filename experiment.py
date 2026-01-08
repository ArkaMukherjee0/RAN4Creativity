#!/usr/bin/env python3
"""
Main experiment script for embedding noise experiments.

Consolidates: model loading, noise injection, text generation, and inference orchestration.

Usage:
    python core/experiment.py
"""

from __future__ import annotations

import json
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
        "device_map": device_map,
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }

    if load_in_8bit:
        model_kwargs["load_in_8bit"] = True
        model_kwargs.pop("torch_dtype", None)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    print(f"Model loaded. Device: {next(model.parameters()).device}")
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

    def __init__(self, config, prompts_list: List[str]):
        """
        Args:
            config: Configuration module with settings
            prompts_list: List of prompt strings to use
        """
        self.config = config
        self.prompts = prompts_list
        self.model = None
        self.tokenizer = None
        self.noise_injector = None
        self.results: List[GenerationResult] = []

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
        """Condition A: Deterministic Baseline (temp=0, no noise, 1 sample)."""
        print("\n" + "=" * 80)
        print("CONDITION A: DETERMINISTIC BASELINE")
        print("=" * 80)

        results = []

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")

            self.noise_injector.deactivate()

            generated = generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt_text,
                do_sample=False,
                temperature=None,
                max_new_tokens=self.config.MAX_NEW_TOKENS,
                min_new_tokens=self.config.MIN_NEW_TOKENS,
            )

            result = GenerationResult(
                condition="A",
                prompt_idx=prompt_idx,
                sample_idx=0,
                prompt_text=prompt_text,
                generated_text=generated,
                timestamp=datetime.now().isoformat(),
            )
            results.append(result)

            if self.config.VERBOSE:
                print(f"  Generated {len(generated)} characters")

        print(f"\nCondition A complete: {len(results)} outputs")
        return results

    def run_condition_b(self) -> List[GenerationResult]:
        """Condition B: Temperature Sampling (temp>0, no noise, k samples)."""
        print("\n" + "=" * 80)
        print("CONDITION B: TEMPERATURE SAMPLING")
        print("=" * 80)
        print(f"Settings: temperature={self.config.TEMPERATURE}, {self.config.K_SAMPLES} samples/prompt")

        results = []

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")

            self.noise_injector.deactivate()

            for sample_idx in range(self.config.K_SAMPLES):
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

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")

        print(f"\nCondition B complete: {len(results)} outputs")
        return results

    def run_condition_c(self) -> List[GenerationResult]:
        """Condition C: Embedding Noise (temp=0, noise, k samples)."""
        print("\n" + "=" * 80)
        print("CONDITION C: EMBEDDING NOISE")
        print("=" * 80)
        print(f"Settings: sigma={self.config.SIGMA_SCALE}, {self.config.K_SAMPLES} samples/prompt")

        results = []

        for prompt_idx, prompt_text in enumerate(self.prompts):
            if self.config.SHOW_PROGRESS:
                print(f"\nPrompt {prompt_idx + 1}/{len(self.prompts)}")

            for sample_idx in range(self.config.K_SAMPLES):
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

                if self.config.VERBOSE and self.config.SHOW_PROGRESS:
                    print(f"{len(generated)} chars")

        print(f"\nCondition C complete: {len(results)} outputs")
        return results

    def run_all_conditions(self) -> List[GenerationResult]:
        """Run all three conditions in sequence."""
        print("\n" + "=" * 80)
        print("RUNNING ALL CONDITIONS")
        print("=" * 80)
        print(f"Total prompts: {len(self.prompts)}")
        print(f"Expected outputs:")
        print(f"  Condition A: {len(self.prompts)}")
        print(f"  Condition B: {len(self.prompts) * self.config.K_SAMPLES}")
        print(f"  Condition C: {len(self.prompts) * self.config.K_SAMPLES}")

        all_results = []
        all_results.extend(self.run_condition_a())
        all_results.extend(self.run_condition_b())
        all_results.extend(self.run_condition_c())

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
    import prompts

    print("""
================================================================================
                        EMBEDDING NOISE EXPERIMENT
================================================================================
""")

    print("Configuration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Prompts: {config.N_PROMPTS}")
    print(f"  Samples per condition: 1 (A), {config.K_SAMPLES} (B), {config.K_SAMPLES} (C)")
    print(f"  Temperature (B): {config.TEMPERATURE}")
    print(f"  Sigma scale (C): {config.SIGMA_SCALE}")
    print(f"  Output directory: {config.OUTPUT_DIR}")

    print("\n" + "=" * 80)
    response = input("Proceed with experiment? [y/N]: ")
    if response.lower() != 'y':
        print("Experiment cancelled.")
        return

    prompts = get_all_prompts()[:config.N_PROMPTS]

    inference = SanityCheckInference(config, prompts)
    inference.setup()

    results = inference.run_all_conditions()
    inference.save_results(Path(config.OUTPUT_DIR))

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)
    print(f"\nGenerated {len(results)} total outputs:")
    print(f"  Condition A: {sum(1 for r in results if r.condition == 'A')}")
    print(f"  Condition B: {sum(1 for r in results if r.condition == 'B')}")
    print(f"  Condition C: {sum(1 for r in results if r.condition == 'C')}")
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
