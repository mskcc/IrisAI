---
name: clinical-biostatistics
description: Clinical data analysis — survival analysis (Kaplan-Meier, Cox regression),
  biomarker validation (ROC, calibration), power calculations, clinical trial
  statistics, competing risks
allowed_tools:
  - execute_dynamic_task
  - batch
  - submit_slurm_job
  - slurm_monitor_job
  - write_text_file
  - edit_file
  - read_text_file
  - find_files
  - run_pipeline_script
  - render_image_inline
  - save_image
model: null
max_iterations: 30
guardrails:
- ALWAYS check for censoring in survival data — never treat censored as events
- ALWAYS report confidence intervals with any survival estimate
- For Cox regression, verify proportional hazards assumption before reporting results
- NEVER report p-values without specifying the test used
- Use FDR correction when testing multiple biomarkers
---

# Clinical Biostatistics

Analyze clinical and translational research data: survival analysis, biomarker
validation, power calculations, and standard clinical trial statistics.

## When to Use This Skill

**Triggers:**
- "Survival analysis" / "Kaplan-Meier" / "Cox regression"
- "Biomarker validation" / "ROC curve" / "AUC"
- "Power calculation" / "Sample size"
- "Clinical outcome" / "Time to event"
- "Competing risks" / "Fine-Gray"
- "Hazard ratio"

**NOT for:**
- "Differential expression" → bioinformatics-analysis
- "GSEA/pathway" → pathway-analysis
- "Make a KM plot" (from existing results) → visualization

## Key Recipes

### Kaplan-Meier Survival Analysis

```python
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

kmf = KaplanMeierFitter()
fig, ax = plt.subplots(figsize=(5, 4))

for group_name, group_data in data.groupby('{group_column}'):
    kmf.fit(group_data['{time_col}'], group_data['{event_col}'], label=group_name)
    kmf.plot_survival_function(ax=ax, ci_show=True)

# Log-rank test
results = logrank_test(
    group1['{time_col}'], group2['{time_col}'],
    group1['{event_col}'], group2['{event_col}']
)
ax.set_title(f'Log-rank p = {results.p_value:.4f}')
ax.set_xlabel('Time (months)')
ax.set_ylabel('Survival probability')
fig.savefig('{outdir}/km_curve.png', dpi=300, bbox_inches='tight')
```

### Cox Proportional Hazards

```python
from lifelines import CoxPHFitter

cph = CoxPHFitter()
cph.fit(data, duration_col='{time_col}', event_col='{event_col}',
        formula='{covariates}')
cph.print_summary()

# Check proportional hazards assumption
cph.check_assumptions(data, show_plots=True)
```

### ROC Curve / Biomarker Validation

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.3f}')
ax.plot([0, 1], [0, 1], 'k--', lw=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
fig.savefig('{outdir}/roc_curve.png', dpi=300, bbox_inches='tight')
```

## Statistical Considerations

- Always report: effect size + CI + p-value (not just p-value)
- For survival: median survival with 95% CI
- For Cox: hazard ratio with 95% CI
- Multiple testing: Bonferroni or FDR as appropriate
- Missing data: report and handle explicitly (not silent drops)

## Tools

- `execute_dynamic_task` — Quick statistical calculations
- `submit_slurm_job` — Large datasets or permutation tests
- `write_text_file` — Write analysis scripts
- `run_pipeline_script` — Multi-step analysis
- `render_image_inline` — Display survival curves, ROC plots
- `save_image` — Save publication figures

## References

- `references/survival-methods.md` — Load WHEN complex survival (competing risks, time-varying)
- `references/biomarker-validation.md` — Load WHEN validating clinical biomarker
