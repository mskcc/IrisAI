---
name: sequence-processing
description: Upstream bioinformatics — FASTQ QC (FastQC, fastp), alignment (STAR, BWA-MEM2,
  minimap2, HISAT2), quantification (Cell Ranger, STARsolo, featureCounts), variant
  calling (GATK, Mutect2, DeepVariant, Strelka2), count matrix generation
allowed_tools:
  - submit_slurm_job
  - slurm_monitor_job
  - execute_dynamic_task
  - batch
  - write_text_file
  - edit_file
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - analyze_files
  - run_pipeline_script
  - check_user_slurm_access
  - register_software
  - query_software
model: null
max_iterations: 45
guardrails:
- BEFORE pipeline execution: call query_software to check if tools (STAR, BWA, samtools, etc.) are already installed
- AFTER any software install (via escalation or direct): ALWAYS call register_software
- ALWAYS check available software with get_environment_info before writing pipeline scripts
- ALWAYS use submit_slurm_job for alignment/variant calling (never execute_dynamic_task — too slow)
- ALWAYS verify input FASTQ integrity before starting (file exists, non-empty, gzipped)
- NEVER hardcode reference genome paths — discover via get_environment_info or ask user
- For multi-sample processing, use job arrays or dependency chains (not sequential submission)
- Set memory based on tool requirements (STAR needs 30-40GB, BWA needs 8-16GB)
---

# Sequence Processing

Run upstream bioinformatics pipelines: raw data QC, read alignment, quantification,
and variant calling. All heavy computation runs as Slurm jobs.

## When to Use This Skill

**Triggers:**
- "Align my FASTQ files" / "Run STAR" / "Map reads to hg38"
- "Run FastQC" / "QC my sequencing data"
- "Run Cell Ranger" / "Process my 10X data"
- "Call variants" / "Run GATK" / "Somatic mutations"
- "Generate count matrix" / "featureCounts"
- "Process my sequencing data" / "Run my NGS pipeline"
- "Trim adapters" / "Run fastp"

**NOT for (route elsewhere):**
- "Analyze my scRNA-seq clusters" → bioinformatics-analysis (downstream)
- "GSEA on my DEGs" → pathway-analysis
- "Make a UMAP" → visualization
- "Install STAR" → software-management
- "Submit a job" (generic) → hpc-submit-job

## Defaults & Conventions

- All alignment/variant calling: submit_slurm_job (never execute_dynamic_task)
- Default reference genome: ask user (hg38, mm10, etc.) — never assume
- Default output location: {work_dir}/results/{step_name}/
- Default threads for alignment: 8-16 CPUs
- Default memory: STAR=40G, BWA=16G, GATK=32G, Cell Ranger=64G
- QC can use execute_dynamic_task if single-sample and quick

## Complete Workflow

### Step 1: Context & Discovery

1. **Check project memory:**
   → read_memory(project)
   → Look for: reference genome paths, index locations, conda envs with bioinformatics tools

2. **Discover available tools:**
   → get_environment_info('category:bioinformatics') — find alignment/variant tools
   → get_environment_info('package:star') (or bwa, gatk, etc.) — check if installed
   → If tools not available → escalate to software-management first

3. **Understand the data:**
   → What sequencing type? (RNA-seq, WGS, WES, scRNA-seq, spatial)
   → Paired-end or single-end?
   → How many samples?
   → What organism/reference genome?
   → Where are the FASTQ files?

4. **Verify inputs exist:**
   → find_files(pattern="*.fastq.gz", directory=input_dir)
   → Confirm files are non-empty and properly paired

### Step 2: Plan the Pipeline

Standard workflows:

```
Bulk RNA-seq:
  FastQC/fastp → STAR align → featureCounts → count matrix

Single-cell RNA-seq:
  Cell Ranger / STARsolo → filtered matrix → (downstream in bioinformatics-analysis)

Whole Genome/Exome:
  FastQC/fastp → BWA-MEM2 align → Mark Duplicates → BQSR → Variant Calling

Somatic Mutations:
  Tumor + Normal FASTQs → align both → Mutect2 → filter → annotate
```

### Step 3: Write Pipeline Scripts

Each step is a self-contained Slurm script. For multi-sample, use one script
with sample name as parameter (run as array job or submit per sample).

### Step 4: Execute via Slurm

All heavy steps go through submit_slurm_job. Chain steps with dependencies.

### Step 5: Verify & Report

After each step:
- Check output file exists and is non-empty
- For BAMs: verify with samtools flagstat
- For VCFs: quick variant count check
- Report summary statistics to user

## Key Recipes

### Recipe: FASTQ Quality Control

**Trigger:** "Run FastQC" / "Check quality of my reads"

**Resources:** 4 CPUs, 8G, 30min per sample

```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_with_fastqc}

OUTDIR="{work_dir}/results/fastqc"
mkdir -p "$OUTDIR"

fastqc -t 4 -o "$OUTDIR" {input_fastqs}

echo "FastQC complete. Results in $OUTDIR"
ls "$OUTDIR"/*.html
```

For trimming (fastp):
```bash
fastp \
    -i {r1.fastq.gz} -I {r2.fastq.gz} \
    -o {trimmed_r1.fastq.gz} -O {trimmed_r2.fastq.gz} \
    --thread 8 \
    --html {outdir}/fastp_report.html \
    --json {outdir}/fastp_report.json
```

### Recipe: RNA-seq Alignment (STAR)

**Trigger:** "Align to hg38 with STAR" / "Map my RNA-seq reads"

**Resources:** 16 CPUs, 40G (STAR loads genome into memory), 2-4h per sample

```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_with_star}

GENOME_DIR="{path_to_star_index}"  # Get from get_environment_info or user
OUTDIR="{work_dir}/results/star/{sample_name}"
mkdir -p "$OUTDIR"

STAR \
    --runThreadN 16 \
    --genomeDir "$GENOME_DIR" \
    --readFilesIn {r1.fastq.gz} {r2.fastq.gz} \
    --readFilesCommand zcat \
    --outSAMtype BAM SortedByCoordinate \
    --outFileNamePrefix "${OUTDIR}/" \
    --quantMode GeneCounts \
    --outSAMattributes NH HI AS NM MD

# Index the BAM
samtools index "${OUTDIR}/Aligned.sortedByCoord.out.bam"

# Quick stats
samtools flagstat "${OUTDIR}/Aligned.sortedByCoord.out.bam"
echo "STAR alignment complete for {sample_name}"
```

### Recipe: DNA Alignment (BWA-MEM2)

**Trigger:** "Align WGS to reference" / "Run BWA"

**Resources:** 16 CPUs, 16G, 2-8h per sample (depends on coverage)

```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_with_bwa}

REF="{path_to_bwa_index}"
OUTDIR="{work_dir}/results/bwa/{sample_name}"
mkdir -p "$OUTDIR"

# Align
bwa-mem2 mem -t 16 -R "@RG\tID:{sample}\tSM:{sample}\tPL:ILLUMINA" \
    "$REF" {r1.fastq.gz} {r2.fastq.gz} | \
    samtools sort -@ 4 -o "${OUTDIR}/{sample}.sorted.bam"

# Index
samtools index "${OUTDIR}/{sample}.sorted.bam"

# Stats
samtools flagstat "${OUTDIR}/{sample}.sorted.bam"
echo "BWA alignment complete for {sample_name}"
```

### Recipe: Cell Ranger (10X scRNA-seq)

**Trigger:** "Run Cell Ranger" / "Process 10X data"

**Resources:** 16 CPUs, 64G, 4-12h per sample

```bash
#!/bin/bash
set -euo pipefail

OUTDIR="{work_dir}/results/cellranger"
mkdir -p "$OUTDIR"
cd "$OUTDIR"

cellranger count \
    --id={sample_name} \
    --transcriptome={reference_path} \
    --fastqs={fastq_directory} \
    --sample={sample_name} \
    --localcores=16 \
    --localmem=60

echo "Cell Ranger complete. Matrix at:"
ls "$OUTDIR/{sample_name}/outs/filtered_feature_bc_matrix/"
```

### Recipe: Variant Calling (GATK HaplotypeCaller)

**Trigger:** "Call variants" / "Run GATK"

**Resources:** 8 CPUs, 32G, 2-8h per sample

```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_with_gatk}

REF="{reference_fasta}"
INPUT="{sorted_markdup_bam}"
OUTDIR="{work_dir}/results/variants/{sample_name}"
mkdir -p "$OUTDIR"

# Mark duplicates
gatk MarkDuplicates \
    -I "$INPUT" \
    -O "${OUTDIR}/{sample}.markdup.bam" \
    -M "${OUTDIR}/{sample}.metrics.txt"

samtools index "${OUTDIR}/{sample}.markdup.bam"

# Base Quality Score Recalibration (if known sites available)
gatk BaseRecalibrator \
    -R "$REF" \
    -I "${OUTDIR}/{sample}.markdup.bam" \
    --known-sites {known_sites_vcf} \
    -O "${OUTDIR}/{sample}.recal.table"

gatk ApplyBQSR \
    -R "$REF" \
    -I "${OUTDIR}/{sample}.markdup.bam" \
    --bqsr-recal-file "${OUTDIR}/{sample}.recal.table" \
    -O "${OUTDIR}/{sample}.recal.bam"

# Call variants
gatk HaplotypeCaller \
    -R "$REF" \
    -I "${OUTDIR}/{sample}.recal.bam" \
    -O "${OUTDIR}/{sample}.g.vcf.gz" \
    -ERC GVCF

echo "Variant calling complete. GVCF at ${OUTDIR}/{sample}.g.vcf.gz"
```

### Recipe: Somatic Variant Calling (Mutect2)

**Trigger:** "Somatic mutations" / "Tumor vs normal"

**Resources:** 8 CPUs, 32G, 4-12h

```bash
#!/bin/bash
set -euo pipefail
eval "$(conda shell.bash hook)"
conda activate {env_with_gatk}

gatk Mutect2 \
    -R {reference} \
    -I {tumor_bam} \
    -I {normal_bam} \
    -normal {normal_sample_name} \
    --germline-resource {germline_resource} \
    --panel-of-normals {pon_vcf} \
    -O {outdir}/{sample}.mutect2.vcf.gz

gatk FilterMutectCalls \
    -R {reference} \
    -V {outdir}/{sample}.mutect2.vcf.gz \
    -O {outdir}/{sample}.filtered.vcf.gz

echo "Somatic calling complete. Filtered VCF at {outdir}/{sample}.filtered.vcf.gz"
```

### Recipe: Multi-Sample Processing

**Trigger:** "Process all 20 samples" / "Run pipeline on all FASTQs"

**Approach:** Submit per-sample jobs with shared dependency for merge step:

```
# Submit alignment for each sample
for sample in sample1 sample2 ... sample20; do
    submit_slurm_job(job_name="align_${sample}", ...)
done

# Submit merge/aggregate step depending on all alignments
submit_slurm_job(
    job_name="merge_counts",
    dependency="afterok:{id1}:{id2}:...:{id20}",
    ...
)
```

## Resource Requirements

| Tool | CPUs | Memory | Time/sample | Notes |
|------|------|--------|-------------|-------|
| FastQC | 4 | 4G | 10-30 min | Quick, can use execute_dynamic_task for single file |
| fastp | 8 | 4G | 15-30 min | |
| STAR | 16 | 40G | 30-60 min | Genome index needs 30GB RAM to load |
| BWA-MEM2 | 16 | 16G | 1-4 hours | Depends on coverage |
| Cell Ranger | 16 | 64G | 4-12 hours | |
| STARsolo | 16 | 40G | 1-4 hours | |
| GATK HaplotypeCaller | 8 | 32G | 2-8 hours | |
| Mutect2 | 8 | 32G | 4-12 hours | |
| DeepVariant | 16 | 32G | 2-6 hours | GPU optional |
| samtools sort | 8 | 16G | 30-60 min | |
| featureCounts | 4 | 8G | 10-30 min | Quick |

## Best Practices & Pitfalls

### Always:
- Verify FASTQ files exist and are non-empty before starting
- Set read group (@RG) information during alignment
- Index BAM files after sorting (samtools index)
- Use job dependencies for multi-step pipelines
- Check alignment stats (samtools flagstat) after mapping
- Save reference genome paths to memory for future sessions

### Never:
- Don't run alignment via execute_dynamic_task (will timeout)
- Don't hardcode reference paths — discover or ask user
- Don't skip QC (bad input → bad output, hard to debug later)
- Don't forget to sort BAMs before variant calling
- Don't mix up paired-end read order (R1 before R2)

### Common Pitfalls:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| STAR: "genome loading" then OOM | Insufficient memory | Request ≥40G |
| BWA: "not a valid reference" | Wrong index format | Rebuild index for BWA-MEM2 |
| Cell Ranger: "no barcodes found" | Wrong --sample name | Match FASTQ filename prefix exactly |
| Low mapping rate | Wrong reference or adapters present | Run fastp first, verify reference |
| GATK: "bad CIGAR" | Unsorted or unindexed BAM | Sort + index before GATK |
| Mutect2 timeout | WGS without interval list | Split into chromosomes, run parallel |

## Tools

- `submit_slurm_job` — Run alignment/calling as Slurm jobs (primary)
- `slurm_monitor_job` — Wait for pipeline steps
- `execute_dynamic_task` — Quick checks (file existence, samtools stats)
- `write_text_file` — Write pipeline scripts
- `edit_file` — Modify scripts
- `read_text_file` — Read output logs and stats
- `grep_file` — Search logs for errors
- `find_files` — Locate FASTQ/BAM/VCF files
- `list_directory` — Browse output directories
- `analyze_files` — Multi-file analysis
- `run_pipeline_script` — Multi-step pipelines
- `check_user_slurm_access` — Verify partition access

## References

- `references/aligner-selection.md` — Load WHEN user unsure which aligner to use
- `references/resource-requirements.md` — Load WHEN estimating resources for novel workload
- `../shared/references/conda-in-containers.md` — Load WHEN setting up bioinformatics env
