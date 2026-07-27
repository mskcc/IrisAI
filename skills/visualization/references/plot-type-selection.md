# Plot Type Selection Guide

Choose the right visualization based on what you're showing:

## By Data Relationship

| What you're showing | Best plot type | Alternative |
|--------------------|--------------:|-------------|
| Distribution (1 variable) | Histogram, KDE, violin | Box plot, swarm |
| Comparison (categories) | Bar chart, dot plot | Lollipop, Cleveland dot |
| Relationship (2 continuous) | Scatter plot | Hexbin (large N), contour |
| Trend over time | Line plot | Area plot, step plot |
| Composition | Stacked bar, pie (≤5 slices) | Treemap, waffle |
| Ranking | Horizontal bar | Bump chart |
| Correlation matrix | Heatmap | Clustermap |
| Flow/network | Sankey, network graph | Chord diagram |

## By Bioinformatics Domain

| Analysis type | Standard plot | Library |
|--------------|--------------|---------|
| Single-cell clusters | UMAP/t-SNE | scanpy, seaborn |
| Differential expression | Volcano plot | matplotlib |
| Gene expression comparison | Violin/dot plot | scanpy, seaborn |
| Expression matrix | Heatmap + dendrograms | seaborn clustermap |
| Survival analysis | Kaplan-Meier | lifelines |
| Enrichment results | Dot plot, bar chart | matplotlib |
| Genomic coverage | IGV-style track | matplotlib, pyGenomeTracks |
| Copy number | Genome-wide scatter | matplotlib |
| Variant allele frequency | Histogram, density | matplotlib |
| Cell composition | Stacked bar | matplotlib |
| Trajectory/pseudotime | Stream plot, PAGA | scanpy |
| Spatial transcriptomics | Spatial scatter | scanpy, squidpy |

## Decision Tree

```
What is the primary goal?
│
├─ Show distribution of values?
│  ├─ One group → Histogram or KDE
│  ├─ Compare groups → Violin or box plot
│  └─ Many groups → Ridge plot
│
├─ Compare quantities across categories?
│  ├─ ≤10 categories → Bar chart (vertical)
│  ├─ >10 categories → Horizontal bar (sorted)
│  └─ Show individual points → Strip/swarm + box
│
├─ Show relationship between variables?
│  ├─ 2 continuous → Scatter (+ regression line if needed)
│  ├─ N < 10000 → Regular scatter
│  ├─ N > 10000 → Hexbin or 2D density
│  └─ 3+ variables → Pair plot or parallel coordinates
│
├─ Show change over time?
│  ├─ Single series → Line plot
│  ├─ Multiple series → Multi-line (≤7) or small multiples
│  └─ Discrete time points → Connected scatter
│
├─ Show proportions/composition?
│  ├─ ≤5 categories → Pie chart (controversial but acceptable)
│  ├─ >5 categories → Stacked bar
│  └─ Over time → Stacked area
│
└─ Show matrix/grid of values?
   ├─ Continuous values → Heatmap
   ├─ With clustering → Clustermap
   └─ Sparse binary → Dot plot (sized)
```

## Figure Size Guidelines

| Context | Width | Height | Aspect |
|---------|-------|--------|--------|
| Single column (Nature/Science) | 89mm / 3.5" | varies | ~ 1:1 to 3:4 |
| 1.5 column | 120mm / 4.7" | varies | varies |
| Double column | 183mm / 7.2" | varies | varies |
| Full page | 183mm × 247mm | 7.2" × 9.7" | ~ 3:4 |
| Presentation slide | 10" × 5.6" | — | 16:9 |
| Poster panel | 8" × 6" | — | 4:3 |

## When NOT to Plot

- ≤5 data points → table is clearer
- Single value comparison → state it in text
- Data that's better as a formatted table → don't force a chart
