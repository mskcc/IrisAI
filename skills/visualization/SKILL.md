---
name: visualization
description: Create publication-quality figures — matplotlib, seaborn, plotly, multi-panel
  assembly, Nature/Science/Cell style formatting, heatmaps, UMAP, volcano plots,
  survival curves, bar charts, line plots
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
  - render_image_inline
  - save_image
  - run_pipeline_script
model: null
max_iterations: 45
guardrails:
- ALWAYS call get_environment_info('category:visualization') BEFORE writing any plotting code
- ALWAYS render_image_inline after generating a figure — user cannot see files on disk
- NEVER assume matplotlib/seaborn/plotly are available — CHECK with get_environment_info first
- For large datasets (>1M points), submit_slurm_job instead of execute_dynamic_task
- Save final figures as both PNG (display) and PDF/SVG (publication)
- Default DPI — publication 300, screen display 150
---

# Visualization

Create publication-quality figures for scientific papers, presentations, and data
exploration. Handles everything from quick exploratory plots to multi-panel figures
meeting Nature/Science/Cell submission guidelines.

## When to Use This Skill

**Triggers:**
- "Make a plot" / "Create a figure" / "Visualize this data"
- "Generate a heatmap/UMAP/volcano/bar chart/..."
- "Publication figure" / "Nature style" / "figure for my paper"
- "Multi-panel figure" / "Combine these plots"
- "Show me the distribution" / "Plot the results"
- "Kaplan-Meier curve" / "Survival plot"
- Any request that primarily produces a visual output

**NOT for (route elsewhere):**
- "Run this analysis script" (that happens to make a plot as side effect) → code-execution
- "Process this data" (no visualization requested) → code-execution
- "Install matplotlib" → software-management
- "What plotting libraries exist?" (pure discovery, no figure) → conversational

## Defaults & Conventions

- Default format: PNG at 300 DPI (publication-ready)
- Default style: Clean, minimal, publication-appropriate
- Default font: Arial/Helvetica (universal journal acceptance)
- Default figure size: Single column (3.5" wide) or double column (7" wide)
- Default color palette: colorblind-friendly (viridis, Set2, or paired)
- Output location: work_dir/figures/
- Always save BOTH raster (PNG) and vector (PDF or SVG) versions

## Complete Workflow

### Step 1: Context & Discovery (MANDATORY)

Before writing ANY plotting code:

1. **Check project memory:**
   → read_memory(project)
   → Look for: previously created figures, established style, preferred libraries
   → If a specific style was used before, maintain consistency

2. **Discover available software:**
   → get_environment_info('category:visualization') — find plotting packages
   → get_environment_info('packages') — full registry if category returns nothing
   → get_environment_info('package:<name>') — read API patterns for chosen library
   → Key packages to look for: matplotlib, seaborn, plotly, bokeh, altair, ggplot

3. **Understand the data:**
   → What format? (CSV, h5ad, DataFrame, raw numbers)
   → How many data points? (<10K: quick plot, >1M: needs Slurm)
   → What story should the figure tell?

4. **Choose library and approach:**
   → matplotlib/seaborn: publication figures, full control, static
   → plotly: interactive exploration, HTML output
   → Scanpy/Seurat built-in: scRNA-seq specific (UMAP, violin, dot plot)
   → R ggplot2: if user's data is in R or prefers R style

### Step 2: Plan the Figure

Before writing code, decide:
- **Layout:** Single panel or multi-panel? Grid arrangement?
- **Type:** What plot type best represents this data?
- **Style:** Publication journal requirements? Color scheme?
- **Size:** Single column (3.5") or double column (7") or full page?
- **Labels:** Title, axis labels, legend, annotations?

### Step 3: Write the Plotting Script

Write a COMPLETE, self-contained script:

```python
#!/usr/bin/env python3
"""Generate [description] figure."""
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (REQUIRED in containers)
import matplotlib.pyplot as plt
import numpy as np

# === STYLE SETUP ===
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# === DATA LOADING ===
# ... load from file or generate

# === PLOTTING ===
fig, ax = plt.subplots(figsize=(3.5, 3.0))
# ... create visualization

# === SAVE ===
fig.savefig('{work_dir}/figures/output.png', dpi=300, bbox_inches='tight')
fig.savefig('{work_dir}/figures/output.pdf', bbox_inches='tight')
plt.close(fig)
print(f"Saved: {work_dir}/figures/output.png")
```

### Step 4: Execute

```
execute_dynamic_task(
    task_description="Generate figure",
    commands=["python3 {work_dir}/plot_script.py"],
    work_dir="{work_dir}"
)
```

For large data or GPU-accelerated rendering:
```
submit_slurm_job(
    job_name="generate_figure",
    script_content="#!/bin/bash\npython3 {work_dir}/plot_script.py",
    work_dir="{work_dir}",
    time_limit="00:15:00",
    memory="16G"
)
```

### Step 5: Display to User

**ALWAYS call render_image_inline after generating the figure:**
```
render_image_inline(image_path="{work_dir}/figures/output.png")
```

The user CANNOT see files on disk — if you don't render, they see nothing.

### Step 6: Iterate if Needed

If the user wants changes:
1. Read the existing script: read_text_file
2. Modify specific parameters: edit_file
3. Re-run and re-render
4. Typical iterations: colors, labels, layout, font sizes

### Step 7: Persist Knowledge

- If you established a figure style → update_memory(knowledge)
- If user has journal preferences → update_memory(knowledge)
- Save the final script path for reuse

## Key Recipes

### Recipe: Simple Bar/Line/Scatter Plot

**Trigger:** "Plot X vs Y" / "Bar chart of these values"

**Workflow:**
1. get_environment_info('package:matplotlib') → confirm available
2. Write plotting script with data loading + style + save
3. execute_dynamic_task → run it
4. render_image_inline → show to user

### Recipe: Heatmap (Gene Expression, Correlation, etc.)

**Trigger:** "Heatmap of gene expression" / "Correlation matrix"

**Workflow:**
1. get_environment_info → check for seaborn (preferred) or matplotlib
2. Write script:
   ```python
   import seaborn as sns
   fig, ax = plt.subplots(figsize=(7, 6))
   sns.heatmap(data, cmap='RdBu_r', center=0, 
               xticklabels=True, yticklabels=True,
               cbar_kws={'label': 'Expression (z-score)'})
   ```
3. Execute and render
4. Common adjustments: cluster rows/cols, adjust color range, resize labels

### Recipe: UMAP/t-SNE (Single-Cell)

**Trigger:** "UMAP colored by cell type" / "t-SNE plot"

**Workflow:**
1. get_environment_info → check for scanpy
2. If scanpy available:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad('{path}')
   sc.pl.umap(adata, color='cell_type', save='_celltype.png')
   ```
3. If not, use matplotlib scatter with pre-computed coordinates
4. render_image_inline → show result
5. Common: multiple panels for different colorings

### Recipe: Volcano Plot (Differential Expression)

**Trigger:** "Volcano plot" / "Show significantly changed genes"

**Workflow:**
1. get_environment_info → check matplotlib/seaborn
2. Write script:
   ```python
   # Standard thresholds
   log2fc_thresh = 1.0
   pval_thresh = 0.05
   
   colors = np.where(
       (np.abs(log2fc) > log2fc_thresh) & (pval < pval_thresh),
       'red', np.where(pval < pval_thresh, 'blue', 'grey')
   )
   plt.scatter(log2fc, -np.log10(pval), c=colors, s=5, alpha=0.5)
   plt.axhline(-np.log10(pval_thresh), ls='--', c='grey', lw=0.5)
   plt.axvline(-log2fc_thresh, ls='--', c='grey', lw=0.5)
   plt.axvline(log2fc_thresh, ls='--', c='grey', lw=0.5)
   plt.xlabel('log₂ Fold Change')
   plt.ylabel('-log₁₀ p-value')
   ```
3. Execute, render
4. Optional: label top N significant genes

### Recipe: Kaplan-Meier Survival Curve

**Trigger:** "Survival curve" / "KM plot" / "Time to event"

**Workflow:**
1. get_environment_info → check lifelines or survival libraries
2. Write script:
   ```python
   from lifelines import KaplanMeierFitter
   kmf = KaplanMeierFitter()
   fig, ax = plt.subplots(figsize=(4, 3.5))
   for group in groups:
       mask = data['group'] == group
       kmf.fit(data.loc[mask, 'time'], data.loc[mask, 'event'], label=group)
       kmf.plot_survival_function(ax=ax, ci_show=True)
   ax.set_xlabel('Time (months)')
   ax.set_ylabel('Survival probability')
   ```
3. Execute, render
4. Common additions: risk table, p-value annotation, confidence intervals

### Recipe: Multi-Panel Figure (Publication)

**Trigger:** "Multi-panel figure" / "Figure 1 with panels A, B, C"

**Workflow:**
1. Plan layout: how many panels, grid arrangement
2. Write script using gridspec or subplot_mosaic:
   ```python
   fig = plt.figure(figsize=(7, 5))
   gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.4)
   
   ax_a = fig.add_subplot(gs[0, 0:2])  # Top-left, spans 2 columns
   ax_b = fig.add_subplot(gs[0, 2])    # Top-right
   ax_c = fig.add_subplot(gs[1, :])    # Bottom, full width
   
   # Add panel labels
   for ax, label in zip([ax_a, ax_b, ax_c], 'ABC'):
       ax.text(-0.1, 1.1, label, transform=ax.transAxes,
               fontsize=12, fontweight='bold', va='top')
   ```
3. Execute, render
4. Iterate on spacing, sizes, alignment

### Recipe: Frequency/Distribution Chart

**Trigger:** "Amino acid frequency" / "Distribution of values" / "Histogram"

**Workflow:**
1. get_environment_info('packages') → discover what's available
2. Read or compute the data
3. Write script:
   ```python
   fig, ax = plt.subplots(figsize=(5, 3))
   ax.bar(categories, frequencies, color='steelblue', edgecolor='white', lw=0.5)
   ax.set_xlabel('Category')
   ax.set_ylabel('Frequency')
   ax.set_title('Distribution of X')
   ```
4. Execute, render

## Publication Style Guidelines

### Nature/Science/Cell Formatting

| Property | Value |
|----------|-------|
| Font | Arial or Helvetica (required by most journals) |
| Font size | 5-7pt minimum for any text in figure |
| Line width | 0.5-1.0 pt for axes, 1.0-2.0 pt for data lines |
| Figure width | Single column: 89mm (3.5"), double: 183mm (7.2"), full page: 247mm |
| Resolution | 300 DPI minimum for raster; vector (PDF/EPS) preferred |
| Color | Must be distinguishable in grayscale; colorblind-safe |
| File format | PDF or EPS for vector; TIFF or PNG for raster |
| Panel labels | Bold uppercase letters (A, B, C...) top-left corner |

### Color Palettes (Colorblind-Safe)

| Purpose | Palette | Notes |
|---------|---------|-------|
| Sequential | viridis, plasma, inferno | For continuous data |
| Diverging | RdBu_r, coolwarm, PiYG | For data with a meaningful center |
| Categorical (≤8) | Set2, Dark2, Paired | For discrete groups |
| Categorical (>8) | tab20, custom | May need to use shapes too |
| Heatmap | RdBu_r (diverging), YlOrRd (sequential) | Match data type |

### Common Style Template

```python
PUBLICATION_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.linewidth': 0.5,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'legend.fontsize': 6,
    'legend.frameon': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.5,
}
```

## Best Practices & Pitfalls

### Always:
- Use `matplotlib.use('Agg')` FIRST (before importing pyplot) — required in headless containers
- Save both PNG (for display) and PDF/SVG (for publication)
- Call render_image_inline after generating — user can't see disk files
- Use colorblind-friendly palettes by default
- Set explicit figure size in inches (not rely on defaults)
- Close figures after saving: `plt.close(fig)` — prevents memory leaks in batch
- Create output directory: `os.makedirs('{work_dir}/figures', exist_ok=True)`

### Never:
- Don't use `plt.show()` — it hangs in headless environments
- Don't assume any plotting library is installed — ALWAYS check first
- Don't use `jet` colormap (terrible for colorblind users and perceptual uniformity)
- Don't use tiny fonts (<5pt) — journals will reject
- Don't hardcode colors as hex without documenting meaning
- Don't create figures without axis labels (always label X and Y)
- Don't rely on screen rendering — always use explicit DPI

### Common Pitfalls:

| Symptom | Root Cause | Fix |
|---------|-----------|-----|
| "Display not found" / Tkinter error | Missing `matplotlib.use('Agg')` | Add before any matplotlib import |
| Figure is blank/empty | Data not loaded or wrong column names | Print data shape before plotting |
| Labels cut off | Tight layout not applied | Use `bbox_inches='tight'` in savefig |
| Colors look wrong in PDF | CMYK conversion issue | Use RGB-safe colors, test in PDF viewer |
| Multi-panel spacing ugly | Default spacing too tight | Use `fig.subplots_adjust()` or gridspec |
| Fonts not embedded in PDF | System font not found | Use standard Arial/DejaVu Sans |
| File too large (>10MB) | Too many data points as vector | Rasterize dense scatter: `ax.scatter(..., rasterized=True)` |
| Legend overlaps data | Auto-placement failed | Use `bbox_to_anchor` for external legend |

## Tools

- `execute_dynamic_task` — Run plotting scripts (<5 min)
- `submit_slurm_job` — Run plotting with large data or GPU rendering
- `slurm_monitor_job` — Wait for Slurm plotting job
- `write_text_file` — Write the plotting script
- `edit_file` — Modify existing plotting script (iterate on style)
- `read_text_file` — Read data files or existing scripts
- `render_image_inline` — Display generated figure in chat (MANDATORY after every plot)
- `save_image` — Save image to a specific path
- `find_files` — Locate data files or previous figures
- `list_directory` — Browse for input data
- `grep_file` — Search for specific data patterns
- `run_pipeline_script` — Multi-step figure generation

## References

- `references/publication-guidelines.md` — Load WHEN user mentions "Nature", "Science", "Cell", "journal", or "publication"
- `references/plot-type-selection.md` — Load WHEN user is unsure which plot type to use
- `../shared/references/conda-in-containers.md` — Load WHEN plotting needs a conda env with specific libraries
