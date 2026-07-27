---
name: pathway-analysis
description: Gene set enrichment and pathway analysis — GSEA, GO enrichment, KEGG,
  Reactome, MSigDB, over-representation analysis, enrichment visualization
allowed_tools:
  - execute_dynamic_task
  - batch
  - submit_slurm_job
  - slurm_monitor_job
  - write_text_file
  - edit_file
  - read_text_file
  - grep_file
  - find_files
  - list_directory
  - run_pipeline_script
  - render_image_inline
  - save_image
model: null
max_iterations: 30
guardrails:
- ALWAYS check which enrichment tools are available via get_environment_info before writing scripts
- ALWAYS specify organism/database version when running enrichment
- Present results sorted by adjusted p-value (FDR), not raw p-value
- Include number of genes in overlap and gene set size in results
- For GSEA, use pre-ranked list (log2FC × -log10(p)) not just fold change
---

# Pathway Analysis

Perform gene set enrichment and pathway analysis on gene lists or ranked gene
lists. Covers GO (Biological Process, Molecular Function, Cellular Component),
KEGG, Reactome, MSigDB Hallmarks, and custom gene sets.

## When to Use This Skill

**Triggers:**
- "GSEA" / "Gene set enrichment"
- "GO enrichment" / "What pathways are enriched?"
- "KEGG analysis" / "Reactome pathways"
- "What are these genes doing?" (given a gene list)
- "Enrichment analysis on my DEGs"
- "MSigDB" / "Hallmark gene sets"
- "Over-representation analysis"

**NOT for (route elsewhere):**
- "Find differentially expressed genes" → bioinformatics-analysis
- "Cluster my cells" → bioinformatics-analysis
- "Make a plot of enrichment results" (if already have results) → visualization
- "Align my reads" → sequence-processing

## Complete Workflow

### Step 1: Context & Discovery

1. **Check available tools:**
   → get_environment_info('category:bioinformatics')
   → Look for: gseapy, gprofiler, clusterProfiler (R), enrichr, scanpy
   → get_environment_info('package:gseapy') — read API if available

2. **Understand the input:**
   → Ranked gene list? (for GSEA — needs ranking metric)
   → Unranked gene list? (for ORA/hypergeometric)
   → Full DEG table? (can extract both)
   → What organism? (human, mouse — affects database)

3. **Choose method:**

| Input | Method | Tool |
|-------|--------|------|
| Full DEG table with stats | GSEA (pre-ranked) | gseapy.prerank |
| Significant gene list only | Over-Representation Analysis (ORA) | gseapy.enrich, gprofiler |
| Ranked list (custom metric) | GSEA (pre-ranked) | gseapy.prerank |
| Expression matrix + groups | GSVA/ssGSEA | R GSVA package |

### Step 2: Prepare Input

**For GSEA (pre-ranked):**
```python
import pandas as pd
degs = pd.read_csv("{deg_file}")
# Ranking metric: sign(log2FC) × -log10(pvalue)
degs['rank_metric'] = degs['log2FoldChange'] * (-np.log10(degs['pvalue'].clip(1e-300)))
ranked = degs[['gene_name', 'rank_metric']].dropna().sort_values('rank_metric', ascending=False)
ranked.to_csv("{outdir}/ranked_genes.rnk", sep='\t', header=False, index=False)
```

**For ORA (gene list):**
```python
sig_genes = degs[degs['padj'] < 0.05]['gene_name'].tolist()
# Optionally: up-regulated only
up_genes = degs[(degs['padj'] < 0.05) & (degs['log2FoldChange'] > 1)]['gene_name'].tolist()
```

### Step 3: Run Enrichment

**Using gseapy (Python — preferred if available):**

```python
import gseapy as gp

# GSEA pre-ranked
results = gp.prerank(
    rnk="{ranked_file}",
    gene_sets='MSigDB_Hallmark_2020',  # or GO_Biological_Process_2023, KEGG_2021_Human
    outdir="{outdir}/gsea_results",
    min_size=15,
    max_size=500,
    permutation_num=1000,
    seed=42
)
# Top results
top = results.res2d.sort_values('FDR q-val').head(20)
print(top[['Term', 'NES', 'FDR q-val', 'Lead_genes']].to_string())
```

**Over-representation analysis:**

```python
results = gp.enrich(
    gene_list=sig_genes,
    gene_sets='GO_Biological_Process_2023',
    organism='human',
    outdir="{outdir}/ora_results"
)
top = results.results.sort_values('Adjusted P-value').head(20)
print(top[['Term', 'Overlap', 'Adjusted P-value', 'Genes']].to_string())
```

**Using gprofiler (web-based, no install needed):**

```python
from gprofiler import GProfiler
gp = GProfiler(return_dataframe=True)
results = gp.profile(
    organism='hsapiens',
    query=sig_genes,
    sources=['GO:BP', 'KEGG', 'REAC']
)
print(results[['source', 'native', 'name', 'p_value', 'intersection_size']].head(20).to_string())
```

### Step 4: Visualize Results

Standard enrichment visualizations:

**Dot plot (most common):**
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5))
scatter = ax.scatter(
    top['Overlap_ratio'], range(len(top)),
    s=top['Gene_count'] * 5,
    c=-np.log10(top['Adjusted P-value']),
    cmap='RdYlBu_r'
)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['Term'])
ax.set_xlabel('Gene Ratio')
plt.colorbar(scatter, label='-log10(FDR)')
fig.savefig("{outdir}/enrichment_dotplot.png", dpi=300, bbox_inches='tight')
```

**Bar plot:**
```python
fig, ax = plt.subplots(figsize=(5, 4))
ax.barh(range(len(top)), -np.log10(top['Adjusted P-value']))
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top['Term'])
ax.set_xlabel('-log10(FDR)')
```

Always: render_image_inline after generating plot.

### Step 5: Report & Persist

- Report top 10-20 enriched pathways
- Include: Term name, NES (for GSEA), FDR, gene count
- Save full results to CSV
- If pattern emerges → update_memory with key findings

## Gene Set Databases

| Database | Coverage | Best For |
|----------|----------|----------|
| GO:BP | Biological processes | General function |
| GO:MF | Molecular functions | Specific activities |
| GO:CC | Cellular compartments | Localization |
| KEGG | Metabolic/signaling pathways | Pathway-centric view |
| Reactome | Detailed pathway hierarchy | Mechanistic detail |
| MSigDB Hallmarks | 50 curated gene sets | Quick biological themes |
| MSigDB C2:CP | Canonical pathways | Comprehensive pathways |
| MSigDB C5 | GO gene sets | GO via MSigDB format |
| MSigDB C7 | Immunologic signatures | Immune-focused |
| WikiPathways | Community-curated | Niche pathways |

## Method Selection Guide

| Question | Method | Why |
|----------|--------|-----|
| "What pathways are affected?" (have full DEG stats) | GSEA pre-ranked | Uses all genes, detects subtle shifts |
| "What are these genes enriched for?" (list only) | ORA (hypergeometric) | Simple, no ranking needed |
| "Per-sample pathway scores" | GSVA/ssGSEA | Sample-level enrichment scores |
| "Quick functional overview" | gprofiler (web API) | No install needed, multiple databases |

## Best Practices & Pitfalls

### Always:
- Use FDR-corrected p-values for reporting (not raw p-values)
- Report gene set size and overlap count alongside p-values
- For GSEA: include NES (Normalized Enrichment Score) — direction matters
- Specify organism and database version
- Consider multiple testing: hundreds of gene sets tested

### Never:
- Don't use raw p-values to claim significance
- Don't use only fold change for GSEA ranking (combine with p-value)
- Don't ignore the direction of enrichment (up vs down in GSEA)
- Don't report enrichment without specifying background (ORA needs it)
- Don't mix organisms (human gene list against mouse pathways)

### Common Pitfalls:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| No significant results | Background too broad or gene list too small | Check gene list size (≥10 needed), use appropriate background |
| Everything significant | Gene list very large or wrong background | Use smaller, focused gene list or restrict background |
| Redundant terms | GO hierarchy overlap | Use GO slim or filter redundant terms (revigo) |
| Wrong gene IDs | Mixed Ensembl/Symbol/Entrez | Convert all to same ID type before analysis |
| GSEA: all positive NES | Ranking metric wrong direction | Check sign of ranking metric |

## Tools

- `execute_dynamic_task` — Quick enrichment analysis (<5 min)
- `submit_slurm_job` — Large-scale enrichment or GSVA
- `write_text_file` — Write analysis scripts
- `read_text_file` — Read gene lists, DEG tables
- `run_pipeline_script` — Multi-step enrichment workflow
- `render_image_inline` — Display enrichment plots
- `save_image` — Save figure files
- `find_files` — Locate input gene lists
- `grep_file` — Search for gene names in results

## References

- `references/gene-set-databases.md` — Load WHEN user needs help choosing database
- `references/enrichment-methods.md` — Load WHEN comparing ORA vs GSEA vs GSVA approaches
