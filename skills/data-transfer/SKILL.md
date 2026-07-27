---
name: data-transfer
description: Download datasets and transfer files — wget/curl from public repos (GEO,
  SRA, TCGA), rsync between systems, datatransfer partition, checksum verification
allowed_tools:
  - submit_slurm_job
  - slurm_monitor_job
  - execute_dynamic_task
  - batch
  - write_text_file
  - read_text_file
  - find_files
  - list_directory
  - check_user_slurm_access
model: null
max_iterations: 30
guardrails:
- Large downloads (>1GB) MUST use submit_slurm_job on datatransfer partition (not login node)
- ALWAYS verify downloads with checksum (md5sum/sha256sum) when available
- NEVER download to /home (limited space) — use work_dir on /data1
- For public databases, use their official download tools (prefetch for SRA, gdc-client for TCGA)
---

# Data Transfer

Download datasets from public repositories and transfer files between systems.
Handles downloads from GEO, SRA, TCGA, and other public databases, plus
rsync transfers and checksum verification.

## When to Use This Skill

**Triggers:**
- "Download this dataset" / "Get data from GEO/SRA/TCGA"
- "wget this URL" / "Download from..."
- "Transfer files from..." / "rsync"
- "Download the reference genome"

**NOT for:**
- "Upload my file" → file-operations
- "Find a file on disk" → file-operations
- "How much space do I have?" → storage-management

## Key Recipes

### Download from URL (Small, <1GB)
```
execute_dynamic_task(
    commands=["wget -P {work_dir}/downloads '{url}'"]
)
```

### Download from URL (Large, >1GB)
Must use datatransfer partition:
```
submit_slurm_job(
    job_name="download_dataset",
    script_content="#!/bin/bash\nwget -P {work_dir}/downloads '{url}'",
    partition="datatransfer",
    time_limit="04:00:00",
    memory="4G",
    cpus=1
)
```

### SRA Download (Sequencing Data)
```bash
#!/bin/bash
# Use prefetch + fasterq-dump for SRA downloads
prefetch {SRR_accession} --output-directory {work_dir}/sra/
fasterq-dump {work_dir}/sra/{SRR_accession} --outdir {work_dir}/fastq/ --threads 4
```

### TCGA/GDC Download
```bash
#!/bin/bash
gdc-client download -m {manifest_file} -d {work_dir}/gdc_data/
```

### Checksum Verification
```bash
# After download, verify integrity
md5sum -c {checksum_file}
# OR
sha256sum {downloaded_file} | grep {expected_hash}
```

## Datatransfer Partition

- Dedicated partition for I/O-heavy transfers
- QOS limit: 1 CPU max
- Use for any download >1GB or long-running transfers
- Do NOT use for computation — only data movement

## Tools

- `submit_slurm_job` — Large downloads on datatransfer partition
- `slurm_monitor_job` — Wait for download completion
- `execute_dynamic_task` — Small downloads, checksum verification
- `write_text_file` — Write download scripts
- `check_user_slurm_access` — Verify datatransfer partition access
