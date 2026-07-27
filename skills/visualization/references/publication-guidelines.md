# Publication Figure Guidelines

## Journal-Specific Requirements

### Nature / Nature family
- Width: single column 89mm, double 183mm
- Font: Arial, Helvetica, or Times New Roman ONLY
- Minimum font size: 5pt (7pt recommended)
- Resolution: 300 DPI minimum
- Format: PDF, EPS, or TIFF (NOT JPEG for line art)
- Color: RGB for online, CMYK not required
- Panel labels: lowercase bold (a, b, c)

### Science / Science family
- Width: single column 85mm, double 180mm
- Font: Arial or Helvetica preferred
- Minimum font size: 6pt
- Resolution: 300 DPI for photos, 600 DPI for line art
- Format: PDF or EPS preferred
- Panel labels: uppercase bold (A, B, C)

### Cell / Cell Press
- Width: single column 85mm, 1.5 column 114mm, double 174mm
- Font: Arial (Helvetica acceptable)
- Minimum font size: 5pt
- Resolution: 300 DPI
- Format: PDF, EPS, TIFF
- Panel labels: uppercase bold, 8pt minimum

### PNAS
- Width: single column 8.7cm, double 17.8cm
- Font: sans-serif preferred
- Resolution: 300-600 DPI
- Format: TIFF, EPS, PDF

### General Best Practices (Any Journal)

1. **Always use vector format** (PDF/EPS) when possible — raster only for photographs or >100K data points
2. **Embed all fonts** — don't rely on system fonts in the PDF
3. **White background** — no grey figure backgrounds
4. **Remove unnecessary elements** — no grid lines unless essential, minimal ticks
5. **Consistent style** across all figures in the paper
6. **Colorblind accessible** — use patterns/shapes in addition to color when possible
7. **Scale bars** instead of axis labels for images/microscopy
8. **Error bars** must be defined in the legend (SEM, SD, or CI)
9. **No 3D effects** — never use 3D bar charts or pie charts
10. **Legend inside figure** when possible (saves space)

## Matplotlib rcParams for Publication

```python
# Nature-style
NATURE_STYLE = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
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
    'pdf.fonttype': 42,  # TrueType fonts in PDF (required by many journals)
    'ps.fonttype': 42,
}
```

## Panel Labels

```python
# Add panel labels to axes
def add_panel_labels(axes, labels=None, fontsize=10, fontweight='bold',
                     x=-0.1, y=1.1, uppercase=True):
    """Add A, B, C... labels to figure panels."""
    if labels is None:
        labels = 'abcdefghijklmnopqrstuvwxyz' if not uppercase else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for ax, label in zip(axes, labels):
        ax.text(x, y, label, transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, va='top', ha='right')
```

## Statistical Annotations

```python
# Add significance brackets
def add_significance(ax, x1, x2, y, p_value, height=0.02):
    """Add significance bracket between two positions."""
    stars = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], 'k-', lw=0.5)
    ax.text((x1+x2)/2, y+height, stars, ha='center', va='bottom', fontsize=7)
```

## Color Accessibility

For any figure where color encodes information:
1. Test with a colorblind simulator (e.g., coblis.myndex.com)
2. Use redundant encoding: color + shape, color + pattern
3. Avoid red-green distinctions as the sole differentiator
4. Consider printing in grayscale — does the figure still work?

Recommended colorblind-safe palettes:
- **Wong palette** (8 colors): #000000, #E69F00, #56B4E9, #009E73, #F0E442, #0072B2, #D55E00, #CC79A7
- **Tol palette** (12 colors): designed specifically for colorblind users
- **Viridis/Plasma/Inferno**: perceptually uniform, colorblind-safe sequential colormaps
