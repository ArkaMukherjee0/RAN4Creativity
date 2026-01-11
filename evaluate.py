#!/usr/bin/env python3
"""
Evaluation module for embedding noise experiments.

Consolidates: signature extraction, basic metrics, and LLM judges (Groq/Gemini).

Usage:
    python core/evaluate.py metrics                    # Compute basic metrics
    python core/evaluate.py judge groq                 # Run Groq LLM judge
    python core/evaluate.py judge gemini               # Run Gemini LLM judge
    python core/evaluate.py judge groq --n-per-condition 5  # Limit samples
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Literal


# ============================================================================
# Constraint Signature Extraction
# ============================================================================

_RE_O_COMPLEXITY = re.compile(r"O\s*\(([^\)]{1,40})\)", re.IGNORECASE)
_RE_TESTCASES = re.compile(r"\btest\s*cases\b|\beach\s*test\s*case\b", re.IGNORECASE)
_RE_N_BOUND = re.compile(
    r"\b(?:n|m|q|k)\b\s*(?:<=|<|≤)\s*([0-9][0-9_.,]*\s*(?:e\s*\d{1,2}|\^\s*\d{1,2})?)",
    re.IGNORECASE,
)


def _normalize_text(s: str) -> str:
    return (s or "").lower()


def detect_input_type(text: str) -> str:
    """Detect the input structure type from problem text."""
    t = _normalize_text(text)

    if any(k in t for k in ["grid", "matrix", "n rows", "m columns", "cell (i, j)", "2d"]):
        return "grid"
    if any(k in t for k in ["tree", "rooted tree", "parent", "subtree", "lca"]):
        return "tree"
    if any(k in t for k in ["graph", "vertices", "edges", "adjacency", "shortest path", "bfs", "dfs"]):
        return "graph"
    if any(k in t for k in ["string", "substring", "palindrome", "characters"]):
        return "string"
    if any(k in t for k in ["permutation", "permute"]):
        return "permutation"
    if any(k in t for k in ["array", "sequence", "list of integers", "a1", "a_i", "ai", "elements"]):
        return "array"
    if "integer" in t or "numbers" in t:
        return "numeric"
    return "unknown"


def _parse_sci_like(token: str) -> Optional[int]:
    """Parse common CP constraint formats: 2e5, 10^5, 200000, etc."""
    if not token:
        return None
    tok = token.strip().lower().replace(",", "").replace("_", "")

    # 10^5
    m = re.fullmatch(r"10\s*\^\s*(\d{1,2})", tok)
    if m:
        return 10 ** int(m.group(1))

    # 2e5, 1e6
    m = re.fullmatch(r"(\d{1,3})\s*e\s*(\d{1,2})", tok)
    if m:
        return int(m.group(1)) * (10 ** int(m.group(2)))

    # plain integer
    m = re.fullmatch(r"\d{1,10}", tok)
    if m:
        return int(tok)

    return None


def extract_n_max(text: str) -> Optional[int]:
    """Extract upper bound for N/M/Q/K from problem text."""
    vals: list[int] = []
    for m in _RE_N_BOUND.finditer(text or ""):
        v = _parse_sci_like(m.group(1))
        if v is not None:
            vals.append(v)
    return max(vals) if vals else None


def bucket_n(n_max: Optional[int]) -> str:
    """Bucket N into coarse CP-style ranges."""
    if n_max is None:
        return "unknown"
    if n_max <= 200:
        return "<=200"
    if n_max <= 2000:
        return "<=2e3"
    if n_max <= 20000:
        return "<=2e4"
    if n_max <= 200000:
        return "<=2e5"
    if n_max <= 1000000:
        return "<=1e6"
    return ">1e6"


def detect_multi_testcase(text: str) -> bool:
    """Detect if problem has multiple test cases."""
    t = text or ""
    if _RE_TESTCASES.search(t):
        return True
    if re.search(r"first\s+line\s+contains\s+an?\s+integer\s+t\b", t, re.IGNORECASE):
        return True
    return False


def extract_complexity_hint(text: str) -> Optional[str]:
    """Extract O(...) complexity hint if present."""
    m = _RE_O_COMPLEXITY.search(text or "")
    if not m:
        return None
    return m.group(1).strip()[:40]


@dataclass(frozen=True)
class ConstraintSignature:
    """Heuristic constraint signature for diversity measurement."""
    input_type: str
    n_bucket: str
    multi_test: bool
    complexity_hint: Optional[str]

    def to_compact(self) -> str:
        base = f"{self.input_type}|{self.n_bucket}|{'T' if self.multi_test else 'F'}"
        return f"{base}|O({self.complexity_hint})" if self.complexity_hint else base


def build_signature(text: str) -> ConstraintSignature:
    """Build constraint signature from problem text."""
    return ConstraintSignature(
        input_type=detect_input_type(text),
        n_bucket=bucket_n(extract_n_max(text)),
        multi_test=detect_multi_testcase(text),
        complexity_hint=extract_complexity_hint(text),
    )


# ============================================================================
# Basic Metrics Computation
# ============================================================================

ALG_CLASS_CHOICES = {"greedy", "dp", "graph", "math", "brute", "ds", "other"}


def _norm_bool(v: str) -> Optional[bool]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in {"y", "yes", "true", "1"}:
        return True
    if s in {"n", "no", "false", "0"}:
        return False
    return None


def _norm_alg(v: str) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s == "":
        return None
    return s if s in ALG_CLASS_CHOICES else "other"


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compute_metrics(
    results_path: Path,
    annotations_path: Path,
    output_path: Path,
    include_a: bool = False,
) -> dict:
    """
    Compute basic metrics from results and annotations.

    Metrics:
    - Validity@k: fraction of valid outputs
    - AlgDiversity@k: distinct algorithm classes among valid
    - SigDiversity@k: distinct constraint signatures among valid
    """
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = data.get("results", [])

    # Load annotations
    anns = {}
    with open(annotations_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["condition"], int(row["prompt_idx"]), int(row["sample_idx"]))
            anns[key] = row

    keep_conditions = {"A", "B", "C"} if include_a else {"B", "C"}

    # Group by prompt and condition
    per_prompt: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    missing_ann = 0

    for r in results:
        cond = r.get("condition")
        if cond not in keep_conditions:
            continue

        key = (cond, r.get("prompt_idx"), r.get("sample_idx"))
        ann = anns.get(key)
        if ann is None:
            missing_ann += 1
            continue

        valid = _norm_bool(ann.get("valid", ""))
        alg = _norm_alg(ann.get("alg_class", ""))
        sig_override = (ann.get("sig_override") or "").strip()
        sig_auto = (ann.get("sig_auto") or "").strip()
        sig = sig_override if sig_override else sig_auto

        per_prompt[int(r.get("prompt_idx"))][cond].append((valid, alg, sig))

    # Compute per-prompt metrics
    prompt_rows = []
    for p_idx in sorted(per_prompt.keys()):
        for cond in sorted(per_prompt[p_idx].keys()):
            triples = per_prompt[p_idx][cond]
            labeled = [t for t in triples if t[0] is not None]
            valid_triples = [t for t in labeled if t[0] is True]

            validity_k = len(valid_triples) / len(labeled) if labeled else 0.0
            alg_set = {t[1] for t in valid_triples if t[1]}
            sig_set = {t[2] for t in valid_triples if t[2]}

            prompt_rows.append({
                "prompt_idx": p_idx,
                "condition": cond,
                "n_total": len(triples),
                "n_labeled": len(labeled),
                "n_valid": len(valid_triples),
                "validity_at_k": validity_k,
                "alg_diversity_at_k": len(alg_set),
                "sig_diversity_at_k": len(sig_set),
                "effective_alg_diversity": len(alg_set) * validity_k,
            })

    # Aggregate by condition
    by_cond = defaultdict(list)
    for row in prompt_rows:
        by_cond[row["condition"]].append(row)

    summary = {}
    for cond, rows in by_cond.items():
        summary[cond] = {
            "prompts": len({r["prompt_idx"] for r in rows}),
            "avg_validity_at_k": _mean([r["validity_at_k"] for r in rows]),
            "avg_alg_diversity_at_k": _mean([r["alg_diversity_at_k"] for r in rows]),
            "avg_sig_diversity_at_k": _mean([r["sig_diversity_at_k"] for r in rows]),
            "avg_effective_alg_diversity": _mean([r["effective_alg_diversity"] for r in rows]),
        }

    out = {
        "metadata": data.get("metadata", {}),
        "missing_annotations": missing_ann,
        "per_prompt": prompt_rows,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out


# ============================================================================
# LLM Judge
# ============================================================================

JUDGE_PROMPT = """\
You are an expert judge for competitive programming problems generated by smaller models.

Rate this problem on TWO dimensions:

1. CREATIVITY (1-10): How novel/original is the problem concept?
   1-3 = Completely generic (basic array max, simple iteration)
   4-6 = Familiar pattern with some variation
   7-8 = Interesting twist or combination of ideas
   9-10 = Genuinely novel or creative concept

2. VALIDITY (1-10): Is the core idea coherent and potentially solvable?
   Focus on: Does the problem make logical sense? Is there a clear task?
   Be lenient on: Minor wording issues, small ambiguities, missing edge cases
   
   Mark PASS if:
   - The core problem is understandable
   - A reasonable solution approach exists (even if not fully specified)
   - Constraints are present (even if imperfect)
   
   Mark FAIL only if:
   - Problem is completely incoherent
   - Contradictory requirements make it impossible
   - No clear task is defined

Return ONLY valid JSON (no markdown, no explanation):
{{"creativity_score": 5, "creativity_reason": "brief reason", "validity_score": 7, "validity_pass": true, "validity_reason": "brief reason"}}

PROBLEM TO JUDGE:
{problem_text}
"""


@dataclass
class Judgment:
    """LLM judge result for a single output."""
    condition: str
    prompt_idx: int
    sample_idx: int
    creativity_score: int
    creativity_reason: str
    validity_score: int
    validity_pass: bool
    validity_reason: str
    raw_response: str
    error: Optional[str] = None


def extract_json(text: str) -> dict:
    """Robustly extract JSON from LLM response."""
    text = text.strip()

    # Remove markdown code fences
    if "```" in text:
        text = re.sub(r'```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```', '', text)
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON by matching braces
    start = text.find('{')
    if start == -1:
        raise ValueError(f"No JSON object found in: {text[:200]}")

    depth = 0
    end = -1
    for i, c in enumerate(text[start:], start):
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError(f"Unclosed JSON object in: {text[:200]}")

    return json.loads(text[start:end + 1])


class LLMJudge:
    """Unified LLM judge supporting Groq and Gemini."""

    def __init__(
        self,
        provider: Literal["groq", "gemini"],
        model: Optional[str] = None,
        rate_limit_delay: float = 0.5,
    ):
        self.provider = provider
        self.rate_limit_delay = rate_limit_delay
        self.client = None
        self.model = model

        if provider == "groq":
            self._init_groq(model)
        elif provider == "gemini":
            self._init_gemini(model)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def _init_groq(self, model: Optional[str]):
        try:
            from groq import Groq
        except ImportError:
            raise SystemExit("Missing groq package. Install with: pip install groq")

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit("Set GROQ_API_KEY environment variable")

        self.client = Groq()
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

    def _init_gemini(self, model: Optional[str]):
        try:
            from google import genai
            self._use_new_api = True
            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise SystemExit("Set GEMINI_API_KEY environment variable")
            self.client = genai.Client(api_key=api_key)
        except ImportError:
            try:
                import google.generativeai as genai_old
                self._use_new_api = False
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise SystemExit("Set GEMINI_API_KEY environment variable")
                genai_old.configure(api_key=api_key)
                self.client = genai_old
            except ImportError:
                raise SystemExit("Missing google-genai. Install with: pip install google-genai")

        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    def call(self, problem_text: str) -> str:
        """Call the LLM with the judge prompt."""
        prompt = JUDGE_PROMPT.format(problem_text=problem_text[:6000])

        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                    ],
                reasoning_effort="high",
                # include_reasoning=True,
                reasoning_format="hidden",
            )
            return response.choices[0].message.content

        elif self.provider == "gemini":
            if self._use_new_api:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                return response.text
            else:
                model = self.client.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                return response.text

    def judge(self, result: dict, debug: bool = False) -> Judgment:
        """Judge a single result."""
        cond = result["condition"]
        prompt_idx = result["prompt_idx"]
        sample_idx = result["sample_idx"]
        text = result.get("generated_text") or ""

        if not text.strip():
            return Judgment(
                condition=cond,
                prompt_idx=prompt_idx,
                sample_idx=sample_idx,
                creativity_score=0,
                creativity_reason="Empty output",
                validity_score=0,
                validity_pass=False,
                validity_reason="Empty output",
                raw_response="",
                error="empty_output",
            )

        raw = ""
        try:
            raw = self.call(text)
            parsed = extract_json(raw)

            return Judgment(
                condition=cond,
                prompt_idx=prompt_idx,
                sample_idx=sample_idx,
                creativity_score=int(parsed.get("creativity_score", 0)),
                creativity_reason=str(parsed.get("creativity_reason", "")),
                validity_score=int(parsed.get("validity_score", 0)),
                validity_pass=bool(parsed.get("validity_pass", False)),
                validity_reason=str(parsed.get("validity_reason", "")),
                raw_response=raw,
            )
        except Exception as e:
            if debug:
                print(f"\nDEBUG raw response:\n{raw[:500]}\n")
            return Judgment(
                condition=cond,
                prompt_idx=prompt_idx,
                sample_idx=sample_idx,
                creativity_score=0,
                creativity_reason="",
                validity_score=0,
                validity_pass=False,
                validity_reason="",
                raw_response=raw,
                error=str(e),
            )


def save_judge_checkpoint(checkpoint_path: Path, progress: int, judgments: list[Judgment]) -> None:
    """Save judge checkpoint to file atomically."""
    checkpoint_state = {
        'progress': progress,
        'judgments': [asdict(j) for j in judgments],
    }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_suffix('.tmp')

    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_state, f, indent=2, ensure_ascii=False)

    # Atomic rename
    temp_path.replace(checkpoint_path)


def load_judge_checkpoint(checkpoint_path: Path) -> tuple[int, list[Judgment]]:
    """
    Load judge checkpoint from file.

    Returns:
        Tuple of (progress, judgments list). Returns (0, []) if no checkpoint exists.
    """
    if not checkpoint_path.exists():
        return 0, []

    with open(checkpoint_path, 'r', encoding='utf-8') as f:
        state = json.load(f)

    judgments = [Judgment(**j) for j in state.get('judgments', [])]
    progress = state.get('progress', 0)

    return progress, judgments


def print_judge_diagnostics(progress: int, total: int) -> None:
    """Print diagnostic information about judge checkpoint progress."""
    print("\n" + "=" * 80)
    print("CHECKPOINT DIAGNOSTICS")
    print("=" * 80)
    print(f"\nProgress: {progress}/{total} judgments ({(progress/total)*100:.1f}%)")
    print("=" * 80 + "\n")


def run_judge(
    provider: str,
    results_path: Path,
    output_jsonl: Path,
    output_summary: Path,
    n_per_condition: Optional[int] = None,
    include_a: bool = False,
    model: Optional[str] = None,
    debug: bool = False,
    start_from_beginning: bool = False,
    resume: bool = False,
):
    """Run LLM judge on results."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_results = data.get("results", [])

    keep_conditions = {"A", "B", "C", "D"} if include_a else {"B", "C", "D"}
    filtered = [r for r in all_results if r.get("condition") in keep_conditions]

    # Implement weighted sampling: replicate Condition A to match other conditions
    if include_a and "A" in keep_conditions:
        # Count samples per condition
        counts = {c: sum(1 for r in filtered if r["condition"] == c) for c in keep_conditions}

        # Find max count (from B, C, or D)
        non_a_conditions = {c for c in keep_conditions if c != "A"}
        if non_a_conditions:
            max_count = max(counts.get(c, 0) for c in non_a_conditions)
            a_count = counts.get("A", 0)

            if a_count > 0 and max_count > a_count:
                # Calculate replication factor
                replication_factor = max_count // a_count

                print(f"\nWeighted sampling: Replicating Condition A samples")
                print(f"  Condition A: {a_count} samples")
                print(f"  Other conditions: {max_count} samples each")
                print(f"  Replication factor: {replication_factor}x")

                # Replicate Condition A samples
                a_samples = [r for r in filtered if r["condition"] == "A"]
                replicated_a = []
                for _ in range(replication_factor):
                    replicated_a.extend(a_samples)

                # Replace Condition A samples with replicated versions
                filtered = [r for r in filtered if r["condition"] != "A"] + replicated_a

                print(f"  After replication: Condition A has {len(replicated_a)} samples\n")

    if n_per_condition:
        limited = []
        counts = {c: 0 for c in keep_conditions}
        for r in filtered:
            c = r["condition"]
            if counts[c] < n_per_condition:
                limited.append(r)
                counts[c] += 1
        filtered = limited

    judge = LLMJudge(provider=provider, model=model)

    print(f"Judging {len(filtered)} outputs with {provider} ({judge.model})...")
    print(f"Conditions: {keep_conditions}")

    # Setup checkpoint
    checkpoint_path = output_jsonl.parent / f"{provider}_judge_checkpoint.json"
    checkpoint_interval = 10

    # Handle --start_from_beginning flag
    if start_from_beginning:
        if checkpoint_path.exists():
            print(f"\nDeleting existing checkpoint: {checkpoint_path}")
            checkpoint_path.unlink()
        print("Starting from beginning (no checkpoint)")

    # Load checkpoint if resuming
    start_from = 0
    judgments: list[Judgment] = []

    if resume:
        start_from, judgments = load_judge_checkpoint(checkpoint_path)
        if start_from > 0:
            print(f"\nLoaded checkpoint from: {checkpoint_path}")
            print_judge_diagnostics(start_from, len(filtered))
        else:
            print(f"\nNo checkpoint found at: {checkpoint_path}")
            print("Starting from beginning")

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Open file in append mode if resuming, write mode otherwise
    file_mode = "a" if (resume and start_from > 0) else "w"

    with open(output_jsonl, file_mode, encoding="utf-8") as f_out:
        for i, r in enumerate(filtered):
            # Skip already processed judgments
            if i < start_from:
                continue

            cond = r["condition"]
            print(f"  [{i + 1}/{len(filtered)}] {cond} p{r['prompt_idx']} s{r['sample_idx']}...", end=" ", flush=True)

            j = judge.judge(r, debug=debug)
            judgments.append(j)
            f_out.write(json.dumps(asdict(j)) + "\n")
            f_out.flush()

            if j.error:
                print(f"ERROR: {j.error[:60]}")
            else:
                print(f"C={j.creativity_score} V={j.validity_score} {'PASS' if j.validity_pass else 'FAIL'}")

            time.sleep(judge.rate_limit_delay)

            # Save checkpoint every 10 judgments
            current_progress = i + 1
            if current_progress % checkpoint_interval == 0:
                save_judge_checkpoint(checkpoint_path, current_progress, judgments)

    # Save final checkpoint
    save_judge_checkpoint(checkpoint_path, len(filtered), judgments)

    # Aggregate summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = {}
    for cond in sorted(keep_conditions):
        cond_js = [j for j in judgments if j.condition == cond and j.error is None]
        n_errors = sum(1 for j in judgments if j.condition == cond and j.error)

        if not cond_js:
            summary[cond] = {"n": 0, "n_errors": n_errors}
            continue

        creativity_scores = [j.creativity_score for j in cond_js]
        validity_scores = [j.validity_score for j in cond_js]
        validity_passes = [j.validity_pass for j in cond_js]

        stats = {
            "n": len(cond_js),
            "n_errors": n_errors,
            "creativity_mean": sum(creativity_scores) / len(creativity_scores),
            "creativity_min": min(creativity_scores),
            "creativity_max": max(creativity_scores),
            "validity_mean": sum(validity_scores) / len(validity_scores),
            "validity_min": min(validity_scores),
            "validity_max": max(validity_scores),
            "validity_pass_rate": sum(validity_passes) / len(validity_passes),
        }
        summary[cond] = stats

        print(f"\nCondition {cond} (n={stats['n']}, errors={stats['n_errors']}):")
        print(f"  Creativity:  mean={stats['creativity_mean']:.2f}  range=[{stats['creativity_min']}, {stats['creativity_max']}]")
        print(f"  Validity:    mean={stats['validity_mean']:.2f}  range=[{stats['validity_min']}, {stats['validity_max']}]")
        print(f"  Pass rate:   {stats['validity_pass_rate'] * 100:.1f}%")

    # Pairwise comparisons
    def compare_conditions(cond1: str, cond2: str, label: str):
        if cond1 in summary and cond2 in summary and summary[cond1].get("n", 0) > 0 and summary[cond2].get("n", 0) > 0:
            print(f"\n{label}:")
            print("-" * 60)
            s1, s2 = summary[cond1], summary[cond2]
            print(f"  Creativity:  {cond1}={s1['creativity_mean']:.2f}  {cond2}={s2['creativity_mean']:.2f}  Δ={s2['creativity_mean'] - s1['creativity_mean']:+.2f}")
            print(f"  Validity:    {cond1}={s1['validity_mean']:.2f}  {cond2}={s2['validity_mean']:.2f}  Δ={s2['validity_mean'] - s1['validity_mean']:+.2f}")
            print(f"  Pass rate:   {cond1}={s1['validity_pass_rate'] * 100:.1f}%  {cond2}={s2['validity_pass_rate'] * 100:.1f}%")

    compare_conditions("A", "C", "A (greedy) vs C (greedy+noise)")
    compare_conditions("B", "C", "B (temp) vs C (noise)")
    compare_conditions("B", "D", "B (temp) vs D (temp+noise)")
    compare_conditions("C", "D", "C (noise) vs D (temp+noise)")

    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nWrote {output_jsonl}")
    print(f"Wrote {output_summary}")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import config

    parser = argparse.ArgumentParser(description="Evaluation tools for embedding noise experiments")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Metrics subcommand
    metrics_parser = subparsers.add_parser("metrics", help="Compute basic metrics from annotations")
    metrics_parser.add_argument("--results", default=f"{config.OUTPUT_DIR}/all_results.json")
    metrics_parser.add_argument("--annotations", default=f"{config.OUTPUT_DIR}/annotations.csv")
    metrics_parser.add_argument("--out", default=f"{config.OUTPUT_DIR}/metrics_summary.json")
    metrics_parser.add_argument("--include-a", action="store_true")

    # Judge subcommand
    judge_parser = subparsers.add_parser("judge", help="Run LLM judge on outputs")
    judge_parser.add_argument("provider", choices=["groq", "gemini"], help="LLM provider")
    judge_parser.add_argument("--results", default=f"{config.OUTPUT_DIR}/all_results.json")
    judge_parser.add_argument("--n-per-condition", type=int, default=None)
    judge_parser.add_argument("--include-a", action="store_true")
    judge_parser.add_argument("--model", default=None)
    judge_parser.add_argument("--debug", action="store_true")
    judge_parser.add_argument(
        "--start_from_beginning",
        action="store_true",
        help="Overwrite existing checkpoint and start from beginning"
    )
    judge_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint (shows progress diagnostics)"
    )

    args = parser.parse_args()

    # Validate mutually exclusive arguments for judge command
    if args.command == "judge":
        if args.start_from_beginning and args.resume:
            parser.error("--start_from_beginning and --resume are mutually exclusive")

    if args.command == "metrics":
        results_path = Path(args.results)
        ann_path = Path(args.annotations)
        out_path = Path(args.out)

        if not results_path.exists():
            raise SystemExit(f"Missing results JSON: {results_path}")
        if not ann_path.exists():
            raise SystemExit(f"Missing annotations CSV: {ann_path}")

        out = compute_metrics(results_path, ann_path, out_path, args.include_a)

        print("\n=== BASIC METRICS SUMMARY ===")
        for cond in sorted(out["summary"].keys()):
            s = out["summary"][cond]
            print(f"\nCondition {cond}:")
            print(f"  avg Validity@k: {s['avg_validity_at_k']:.3f}")
            print(f"  avg AlgDiversity@k: {s['avg_alg_diversity_at_k']:.3f}")
            print(f"  avg SigDiversity@k: {s['avg_sig_diversity_at_k']:.3f}")

        if out["missing_annotations"]:
            print(f"\nWARNING: {out['missing_annotations']} results had no annotation")

        print(f"\nWrote {out_path}")

    elif args.command == "judge":
        results_path = Path(args.results)
        if not results_path.exists():
            raise SystemExit(f"Missing results: {results_path}")

        output_jsonl = Path(f"{config.OUTPUT_DIR}/{args.provider}_d2__judgments.jsonl")
        output_summary = Path(f"{config.OUTPUT_DIR}/{args.provider}__{config.MODEL_NAME}_d2_judgments_summary.json")

        run_judge(
            provider=args.provider,
            results_path=results_path,
            output_jsonl=output_jsonl,
            output_summary=output_summary,
            n_per_condition=args.n_per_condition,
            include_a=args.include_a,
            model=args.model,
            debug=args.debug,
            start_from_beginning=args.start_from_beginning,
            resume=args.resume,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
