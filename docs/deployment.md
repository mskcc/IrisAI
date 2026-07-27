# IrisAI Deployment Guide

This guide covers deploying IrisAI at your institution.

## Infrastructure Requirements

| Component | Version | Notes |
|-----------|---------|-------|
| Slurm | Any recent | Workload manager |
| Singularity/Apptainer | ≥1.5 | Container runtime with fakeroot support |
| Open OnDemand | ≥2.0 | Web portal for interactive app launch |
| LiteLLM Proxy | ≥1.35 | LLM routing layer |
| PostgreSQL | ≥13 | Required by LiteLLM for virtual key management |
| Python | 3.11+ | For LiteLLM and local tooling |

## Step 1: Set Up LiteLLM Proxy

LiteLLM acts as a unified API layer between IrisAI and your LLM provider (AWS Bedrock, Azure OpenAI, etc.).

### Install LiteLLM

```bash
pip install 'litellm[proxy]'
```

### Configure LiteLLM

Create a `litellm_config.yaml`:

```yaml
model_list:
  - model_name: claude-sonnet
    litellm_params:
      model: bedrock/anthropic.claude-sonnet-4-20250514-v1:0
      aws_region_name: us-east-1
  - model_name: claude-opus
    litellm_params:
      model: bedrock/anthropic.claude-opus-4-20250514-v1:0
      aws_region_name: us-east-1
  - model_name: claude-haiku
    litellm_params:
      model: bedrock/anthropic.claude-haiku-4-5-20251001-v1:0
      aws_region_name: us-east-1

general_settings:
  master_key: sk-your-master-key-here
  database_url: postgresql://user:pass@localhost:5432/litellm
```

### Start LiteLLM Proxy

```bash
litellm --config litellm_config.yaml --port 8080
```

## Step 2: Build Singularity Containers

See [containers/README.md](../containers/README.md) for full build instructions.

```bash
export SINGULARITY=/path/to/singularity

# Build MCP servers container (used for all Slurm jobs)
$SINGULARITY build --fakeroot mcp_servers_v2.sif containers/mcp_servers_v3.def

# Build Chainlit UI container
$SINGULARITY build --fakeroot chainlit.sif containers/chainlit.def
```

## Step 3: Configure Open OnDemand

1. Copy the IrisAI directory to your OOD apps location:
   ```bash
   cp -r IrisAI /var/www/ood/apps/sys/irisai
   ```

2. Edit `template/before.sh.erb` to set your environment:
   - `LITELLM_API_BASE` — your LiteLLM proxy URL (already set to use env var override; set `LITELLM_API_BASE` in your OOD environment or directly in this file)
   - Add virtual key generation logic (see [docs/ARCHITECTURE.md](ARCHITECTURE.md#litellm-virtual-key-authentication))
   - Update container paths to match your deployment

3. Edit `submit.yml.erb` to match your Slurm cluster's partitions and resources.

4. Edit `form.yml.erb` if you want to customize the OOD launch form.

## Step 4: Set Environment Variables

All required variables are documented in [.env.example](../.env.example).

The critical ones to configure in `template/before.sh.erb`:

```bash
# Your LiteLLM proxy URL
export LITELLM_API_BASE="http://<your-litellm-host>:8080"
export LITELLM_URL="${LITELLM_API_BASE}"

# Generate a per-user virtual key at session start
LITELLM_VIRTUAL_KEY=$(curl -s -X POST "${LITELLM_API_BASE}/key/generate" \
  -H "Authorization: Bearer ${LITELLM_MASTER_KEY}" \
  -H "Content-Type: application/json" \
  -d "{\"duration\": \"8h\", \"metadata\": {\"user\": \"${USER}\"}}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['key'])")
export LITELLM_VIRTUAL_KEY
```

## Step 5: Test the Deployment

Launch IrisAI from OOD and verify:
1. Chainlit UI loads in browser
2. MCP servers start (check logs in session directory)
3. A simple query ("What's my username?") works end-to-end
4. A Slurm query ("Show me my running jobs") works

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| `LITELLM_VIRTUAL_KEY not set` | before.sh.erb not generating key | Add virtual key generation to before.sh.erb |
| `LITELLM_URL not set` | Env var missing | Set LITELLM_API_BASE in before.sh.erb |
| MCP tools not available | MCP servers not running | Check container launch in script.sh.erb |
| Container not found | Wrong SIF path | Verify container paths in script.sh.erb |
| Slurm jobs not submitting | Partition doesn't exist | Check partition name in submit.yml.erb |
