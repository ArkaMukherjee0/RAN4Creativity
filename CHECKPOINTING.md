# Checkpointing Guide

Both `experiment.py` and `evaluate.py` now support robust checkpointing to prevent data loss during long-running operations.

## Features

- **Automatic checkpointing**: Saves progress every 10 generations/judgments
- **Atomic file operations**: Uses temp file + rename to prevent corruption
- **Resume capability**: Continue from where you left off after interruption
- **Progress diagnostics**: See detailed progress when resuming

## Usage

### experiment.py

Run the experiment with automatic checkpointing:
```bash
python core/experiment.py
```

**Start fresh** (deletes existing checkpoint):
```bash
python core/experiment.py --start_from_beginning
```

**Resume** from last checkpoint (shows progress):
```bash
python core/experiment.py --resume
```

#### Checkpoint Location
- Checkpoint file: `{output_dir}/checkpoint.json`
- Example: `core/outputs/Llama-3.1-8B/checkpoint.json`

#### What Gets Saved
- Condition A completion status
- Condition B/C/D progress (number of generations completed)
- All generated results so far

#### Resume Diagnostics Example
```
================================================================================
CHECKPOINT DIAGNOSTICS
================================================================================

Condition A: COMPLETE
  Generated: 10/10 outputs

Condition B: 47/100 generations
  Progress: 47.0%

Condition C: 0/100 generations
  Progress: 0.0%

Condition D: 0/100 generations
  Progress: 0.0%

Overall Progress: 57/310 (18.4%)
Results collected: 57
================================================================================
```

### evaluate.py

Run LLM judge with automatic checkpointing:
```bash
python core/evaluate.py judge groq
```

**Start fresh** (deletes existing checkpoint):
```bash
python core/evaluate.py judge groq --start_from_beginning
```

**Resume** from last checkpoint (shows progress):
```bash
python core/evaluate.py judge groq --resume
```

#### Checkpoint Location
- Checkpoint file: `{output_dir}/{provider}_judge_checkpoint.json`
- Example: `core/outputs/groq_judge_checkpoint.json`

#### What Gets Saved
- Number of judgments completed
- All judgment results so far

#### Resume Diagnostics Example
```
================================================================================
CHECKPOINT DIAGNOSTICS
================================================================================

Progress: 47/100 judgments (47.0%)
================================================================================
```

## How It Works

### Saving Checkpoints
- Saves every 10 operations to minimize I/O overhead
- Uses atomic write (temp file + rename) to prevent corruption
- Final save after completion

### Resuming
1. Loads checkpoint state
2. Skips already-completed work
3. Continues from where it left off
4. Appends new results to existing output files

### Starting Fresh
- Deletes checkpoint file if it exists
- Overwrites output files
- Starts from the beginning

## Important Notes

1. **Mutually exclusive flags**: Cannot use both `--start_from_beginning` and `--resume` together
2. **Automatic resume**: By default (no flags), the experiment will start fresh each time. Use `--resume` to continue.
3. **Checkpoint cleanup**: Delete checkpoint files manually if you want to ensure a clean start without using `--start_from_beginning`
4. **Output files**: When resuming judge evaluation, results are appended to existing JSONL files

## Example Workflows

### Long experiment with interruptions
```bash
# Start the experiment
python core/experiment.py --num_prompts 10 --num_generations 100

# ... gets interrupted after 50 generations ...

# Resume from checkpoint
python core/experiment.py --num_prompts 10 --num_generations 100 --resume
```

### Testing then full run
```bash
# Test with small sample
python core/experiment.py --num_prompts 2 --num_generations 5

# Start full experiment fresh
python core/experiment.py --num_prompts 10 --num_generations 100 --start_from_beginning
```

### Judge evaluation with retry
```bash
# Start judging
python core/evaluate.py judge groq

# ... API fails after 25 judgments ...

# Resume from checkpoint
python core/evaluate.py judge groq --resume
```
