---
name: ml-training
description: Train ML/DL models on GPU — PyTorch, TensorFlow, JAX, distributed training
  (DDP, FSDP, DeepSpeed), mixed precision, multi-GPU, checkpointing, hyperparameter tuning
allowed_tools:
  - submit_slurm_job
  - slurm_monitor_job
  - check_user_slurm_access
  - write_text_file
  - edit_file
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - query_slurm_cluster
model: null
max_iterations: 30
guardrails:
- ALWAYS check GPU availability before submission (query_slurm_cluster for free GPUs)
- ALWAYS verify user has GPU partition access (check_user_slurm_access)
- ALWAYS recommend checkpointing for training jobs > 2 hours
- Set memory high enough for data loading (not just model — 64G+ typical)
- Set CPUs proportional to GPUs (8 per GPU for data loading)
- For multi-node, verify NCCL connectivity is possible (same partition)
---

# ML Training

Train machine learning and deep learning models on GPU clusters. Handles
single-GPU, multi-GPU (DDP), and multi-node distributed training with
proper resource allocation and checkpointing.

## When to Use This Skill

**Triggers:**
- "Train my model" / "Fine-tune" / "Run training"
- "Use 4 GPUs" / "Distributed training"
- "Multi-GPU" / "DDP" / "FSDP" / "DeepSpeed"
- "Mixed precision" / "fp16 training"
- "Checkpoint my model" / "Resume training"
- "Hyperparameter sweep"

**NOT for (route elsewhere):**
- "Run inference" (single quick prediction) → code-execution
- "Install PyTorch" → software-management
- "Submit a job" (generic) → hpc-submit-job
- "How many free GPUs?" → hpc-query

## Complete Workflow

### Step 1: Context & Discovery

1. **Check software:**
   → get_environment_info('category:ml') — find PyTorch/TensorFlow/JAX
   → get_environment_info('package:pytorch') — check version, CUDA compatibility
   → read_memory(project) — previous training configs, env paths

2. **Check GPU availability:**
   ```
   query_slurm_cluster(
       query="Available GPUs by type",
       commands=["sinfo -p gpu -N -o '%N|%G|%T' -h | grep idle"]
   )
   ```

3. **Verify access:**
   ```
   check_user_slurm_access(username="{user}", partition="gpu")
   ```

### Step 2: Choose Resources

| Training type | GPUs | Memory | CPUs | Time |
|--------------|------|--------|------|------|
| Small model, quick test | 1 | 32G | 8 | 01:00:00 |
| Standard training | 1-2 | 64G | 16 | 04:00:00 - 24:00:00 |
| Large model (DDP) | 4 | 128G | 32 | 12:00:00 - 7d |
| Very large (multi-node) | 8+ | 256G+ | 64+ | 24:00:00 - 7d |
| Hyperparameter sweep | 1/trial | 32G | 8 | varies |

**GPU selection by model size:**
| Model size | Minimum GPU | Recommended |
|-----------|-------------|-------------|
| <1B params | V100 (16GB) | A100 (40GB) |
| 1-7B params | A100 (80GB) | H100 (80GB) |
| 7-13B params | 2× A100 (80GB) | 4× H100 or H200 |
| >13B params | 4-8× H100/H200 | Multi-node |

### Step 3: Write Training Script

**Single GPU:**
```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_path}

python3 train.py \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 100 \
    --checkpoint_dir {work_dir}/checkpoints \
    --data_dir {data_path}
```

**Multi-GPU (DDP):**
```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_path}

export MASTER_ADDR=localhost
export MASTER_PORT=29500

torchrun --nproc_per_node=$SLURM_GPUS_ON_NODE \
    train.py \
    --batch_size 32 \
    --lr 1e-4 \
    --epochs 100 \
    --checkpoint_dir {work_dir}/checkpoints
```

**Multi-Node:**
```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_path}

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export RANK=$SLURM_PROCID

torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py --distributed
```

### Step 4: Submit

```
submit_slurm_job(
    job_name="train_{model_name}",
    script_content="{script}",
    work_dir="{work_dir}",
    time_limit="{time}",
    memory="{mem}",
    cpus={cpus},
    gres="gpu:{type}:{count}"
)
```

For long training (>4h), recommend preemptable with checkpointing.

### Step 5: Monitor & Report

- Report job_id immediately
- For short training: slurm_monitor_job(wait=True)
- For long training: tell user to check back
- After completion: read training logs for final metrics

## Key Recipes

### Recipe: Single GPU Training

```
submit_slurm_job(
    job_name="train_classifier",
    script_content="...",
    gres="gpu:1",
    memory="64G",
    cpus=8,
    time_limit="04:00:00"
)
```

### Recipe: 4-GPU DDP Training

```
submit_slurm_job(
    job_name="train_ddp",
    script_content="...",  # torchrun --nproc_per_node=4
    gres="gpu:a100:4",
    memory="256G",
    cpus=32,
    time_limit="24:00:00"
)
```

### Recipe: Resume from Checkpoint

Check for existing checkpoints, pass --resume flag:
```bash
CKPT_DIR="{work_dir}/checkpoints"
if [ -f "$CKPT_DIR/latest.pt" ]; then
    RESUME_FLAG="--resume $CKPT_DIR/latest.pt"
else
    RESUME_FLAG=""
fi
python3 train.py $RESUME_FLAG ...
```

## Checkpointing Best Practices

- Save every N epochs or every N hours (whichever comes first)
- Save: model state, optimizer state, epoch, best metric, RNG state
- Keep last 3 checkpoints + best model
- For preemptable: checkpoint every 30 minutes minimum
- Use torch.save with full training state (not just model weights)

## Best Practices & Pitfalls

### Always:
- Checkpoint for any training >2 hours
- Set CPUs = 8× number of GPUs (for DataLoader workers)
- Request enough memory for data loading (not just model)
- Use mixed precision (fp16/bf16) to maximize throughput
- Pin memory in DataLoader (pin_memory=True)

### Never:
- Don't train on login node (via execute_dynamic_task)
- Don't forget to set MASTER_ADDR/PORT for multi-GPU
- Don't use bare CUDA device IDs (use LOCAL_RANK from environment)
- Don't skip gradient accumulation if batch doesn't fit in GPU memory

## Tools

- `submit_slurm_job` — Submit training jobs (primary)
- `slurm_monitor_job` — Check training progress
- `check_user_slurm_access` — Verify GPU partition access
- `query_slurm_cluster` — Check GPU availability
- `write_text_file` — Write training scripts
- `edit_file` — Modify training configs
- `read_text_file` — Read training logs
- `find_files` — Locate checkpoints, data

## References

- `references/distributed-strategies.md` — Load WHEN multi-GPU/multi-node training needed
- `references/gpu-memory-planning.md` — Load WHEN model doesn't fit in GPU memory
- `../shared/references/cluster-config.md` — Load WHEN choosing GPU type/partition
