---
name: bioinformatics-analysis
description: Downstream bioinformatics analysis — scRNA-seq (Scanpy, h5ad, clustering,
  DEG, cell typing, trajectory), variant annotation (VEP, ClinVar), differential
  expression, immune deconvolution
allowed_tools:
  - run_pipeline_script
  - execute_dynamic_task
  - batch
  - submit_slurm_job
  - slurm_monitor_job
  - read_text_file
  - write_text_file
  - edit_file
  - grep_file
  - find_files
  - list_directory
  - extract_coding_variants
  - extract_h5ad_summary
  - get_top_n_categories
  - get_unique_values
  - query_software
  - register_software
  - inspect_vcf_summary
  - list_obs_columns
  - summarize_cell_types
  - analyze_files
  - summarize_command_output
  - render_image_inline
  - save_image
model: null
max_iterations: 30
guardrails:
- BEFORE analysis: call query_software to confirm tool/env paths — do NOT guess paths
- AFTER installing any new software (via escalation or direct): ALWAYS call register_software
- ALWAYS use dedicated bio tools (extract_h5ad_summary, inspect_vcf_summary) BEFORE custom scripts
- NEVER read large files (h5ad, BAM, VCF) with read_text_file — use dedicated tools
- For multi-step analysis, use ONE run_pipeline_script call (not multiple tool calls)
- ALWAYS call get_environment_info before writing analysis scripts to confirm packages exist
- After analysis, render_image_inline for any generated plots — user cannot see disk files
---

# Bioinformatics Analysis (Downstream)

Analyze processed bioinformatics data: scRNA-seq clustering and annotation,
differential expression, variant annotation, cell composition analysis, and
trajectory inference. Works with count matrices, h5ad files, and VCF files.

## When to Use This Skill

**Triggers:**
- "Analyze my scRNA-seq data" / "Cluster this h5ad"
- "What cell types are in this dataset?"
- "Differential expression" / "DEGs between groups"
- "Annotate my variants" / "What's in this VCF?"
- "Cell type composition" / "Proportions by condition"
- "Trajectory analysis" / "Pseudotime"
- "Explore this AnnData file"

**NOT for (route elsewhere):**
- "Align my FASTQ files" / "Run STAR" → sequence-processing (upstream)
- "GSEA" / "GO enrichment" / "Pathway analysis" → pathway-analysis
- "Make a UMAP plot" (from existing coordinates) → visualization
- "Install scanpy" → software-management
- "Predict protein structure" → alphafold

## Complete Workflow

### Step 1: Context & Discovery

1. **Check project memory:**
   → read_memory(project) — existing analyses, conda envs, data locations

2. **Discover software:**
   → get_environment_info('category:bioinformatics') — check scanpy, seurat, etc.
   → get_environment_info('package:scanpy') — read API patterns

3. **Understand the data:**
   → What format? (h5ad, CSV counts, VCF, Seurat .rds)
   → What organism?
   → What's already been done? (raw counts vs normalized vs clustered)

### Step 2: Explore the Data

**For h5ad/AnnData:**
```
extract_h5ad_summary(file_path="{path}")
list_obs_columns(file_path="{path}")
get_unique_values(file_path="{path}", column="cell_type")
summarize_cell_types(file_path="{path}")
```

**For VCF:**
```
inspect_vcf_summary(file_path="{path}")
extract_coding_variants(file_path="{path}")
```

### Step 3: Run Analysis

For complex analyses, use ONE run_pipeline_script call:

```python
result = iris.run_python('''
import scanpy as sc
import numpy as np

adata = sc.read_h5ad("{path}")
# ... analysis steps ...
print(results)
''')
print(result["stdout"])
```

For memory-intensive work, use submit_slurm_job instead.

### Step 4: Display Results

- render_image_inline for any plots generated
- Present numerical results in tables
- Highlight notable findings

## Key Recipes

### Recipe: scRNA-seq Exploration

**Trigger:** "What's in this h5ad file?"

1. extract_h5ad_summary → overview (cells, genes, layers)
2. list_obs_columns → available metadata
3. summarize_cell_types → cell composition (if annotated)
4. Report summary to user, suggest next steps

### Recipe: Differential Expression

**Trigger:** "DEGs between condition A and B"

```python
import scanpy as sc
adata = sc.read_h5ad("{path}")
sc.tl.rank_genes_groups(adata, groupby='{condition_column}',
                        groups=['{group_A}'], reference='{group_B}',
                        method='wilcoxon')
result = sc.get.rank_genes_groups_df(adata, group='{group_A}')
result_sig = result[result['pvals_adj'] < 0.05]
print(f"Significant DEGs: {len(result_sig)}")
print(result_sig.head(20).to_string())
result_sig.to_csv("{outdir}/degs_{group_A}_vs_{group_B}.csv", index=False)
```

### Recipe: Full scRNA-seq Pipeline (QC → Clustering → Annotation)

**Trigger:** "Cluster my scRNA-seq data" / "Full analysis from counts"

```python
import scanpy as sc

adata = sc.read_h5ad("{path}")

# QC filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 20].copy()

# Normalize and find variable genes
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Dimensionality reduction
sc.pp.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)

# Save
adata.write_h5ad("{outdir}/clustered.h5ad")
print(f"Cells: {adata.n_obs}, Clusters: {adata.obs['leiden'].nunique()}")
print(adata.obs['leiden'].value_counts())

# Plot
sc.pl.umap(adata, color='leiden', save='_clusters.png', show=False)
```

### Recipe: Variant Annotation

**Trigger:** "Annotate variants" / "What mutations are significant?"

1. inspect_vcf_summary → overview
2. extract_coding_variants → get coding mutations
3. For annotation (VEP/ANNOVAR): submit_slurm_job with annotation script
4. Report: coding variants, missense/nonsense counts, known pathogenic

### Recipe: Cell Composition Comparison

**Trigger:** "Compare cell types between conditions"

```python
import scanpy as sc
import pandas as pd

adata = sc.read_h5ad("{path}")
composition = adata.obs.groupby(['{condition}', '{celltype_col}']).size().unstack(fill_value=0)
proportions = composition.div(composition.sum(axis=1), axis=0)
print(proportions.to_string())
proportions.to_csv("{outdir}/cell_composition.csv")
```

## Bulk Tool Decision Rules

| Situation | Use This | NOT This |
|-----------|----------|----------|
| Exploring h5ad metadata | extract_h5ad_summary + list_obs_columns | Custom script |
| Multi-column h5ad exploration | ONE run_pipeline_script with scanpy | get_unique_values ×N |
| Complex multi-step analysis | ONE run_pipeline_script | Multiple tool calls |
| Need LLM reasoning about results | analyze_files | Raw data dump |
| Memory-intensive (>50GB) | submit_slurm_job | execute_dynamic_task |

## Best Practices & Pitfalls

### Always:
- Use dedicated bio tools first (they're faster and more reliable)
- Check data dimensions before choosing execution method
- ONE pipeline script for multi-step analysis
- Save intermediate results (h5ad after each major step)
- Report cell/gene counts at each filtering step

### Never:
- Don't read h5ad with read_text_file (it's binary)
- Don't submit 5 separate execute_dynamic_task calls for one analysis
- Don't skip QC filtering on raw count data
- Don't assume cell type annotations exist — check first

## Tools

- `extract_h5ad_summary` — Overview of AnnData file (cells, genes, layers)
- `list_obs_columns` — List metadata columns in h5ad
- `get_unique_values` — Unique values in a metadata column
- `get_top_n_categories` — Top N categories by count
- `summarize_cell_types` — Cell type composition
- `inspect_vcf_summary` — VCF file overview
- `extract_coding_variants` — Filter coding variants
- `run_pipeline_script` — Multi-step analysis scripts
- `execute_dynamic_task` — Quick single commands
- `submit_slurm_job` — Memory-intensive analysis
- `render_image_inline` — Display generated plots
- `save_image` — Save figures to disk

## References

- `references/scrna-workflows.md` — Load WHEN complex scRNA-seq analysis needed
- `references/variant-annotation.md` — Load WHEN annotating variants with VEP/ANNOVAR
