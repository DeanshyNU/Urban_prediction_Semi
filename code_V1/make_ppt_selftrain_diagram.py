"""Generate a clean self-train pipeline diagram for PPT.

Simple boxes + arrows, no clutter.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = '/home/hhz6461/Urban_prediction_Semi/data/figures/ppt_selftrain_pipeline.png'

fig, ax = plt.subplots(figsize=(11, 7.5), dpi=150)
ax.set_xlim(0, 11); ax.set_ylim(0, 7.5)
ax.axis('off')

# colors
BOX_FACE = '#f5f5f7'
BOX_EDGE = '#333333'
LOOP_EDGE = '#1f77b4'
ACCENT   = '#d62728'

def box(x, y, w, h, text, fontsize=10, fc=BOX_FACE, ec=BOX_EDGE, bold=False, color='black'):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle="round,pad=0.05,rounding_size=0.1",
                          facecolor=fc, edgecolor=ec, linewidth=1.4)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, weight=weight, color=color, wrap=True)

def arrow(x1, y1, x2, y2, color='black', lw=1.4):
    a = FancyArrowPatch((x1, y1), (x2, y2),
                        arrowstyle='-|>', mutation_scale=14,
                        color=color, linewidth=lw)
    ax.add_patch(a)

# === Title ===
ax.text(5.5, 7.15, 'Self-Train Pipeline (5 rounds, +40 pseudo each)',
        ha='center', fontsize=14, weight='bold')

# === Input box ===
box(0.3, 5.7, 2.7, 0.9,
    'R0 baseline ckpt\n(50 train, 400 unlabeled,\nbest_valid_RMSE = 0.0453)',
    fontsize=10)

# Arrow into loop
arrow(3.05, 6.15, 3.85, 6.15)

# === Loop boundary (dashed) ===
loop_box = Rectangle((3.9, 0.7), 6.85, 5.7,
                     linewidth=2, edgecolor=LOOP_EDGE,
                     facecolor='none', linestyle='--')
ax.add_patch(loop_box)
ax.text(4.0, 6.5, 'for r in 1..5:', fontsize=11, weight='bold',
        color=LOOP_EDGE)

# Step 1
box(4.3, 5.5, 6.2, 0.75,
    '①  Forward model on all unlabeled  →  confidence + embeddings',
    fontsize=10)

arrow(7.4, 5.5, 7.4, 5.1)

# Step 2 — 3-metric select
box(4.3, 4.0, 6.2, 1.1,
    '②  Greedy select K=40 unlabeled stations\n'
    'score = α × diversity  +  β × valid-relevance     '
    '(filtered to top-τ confidence)',
    fontsize=10)

arrow(7.4, 4.0, 7.4, 3.55)

# Step 3 — Pseudo label
box(4.3, 2.75, 6.2, 0.8,
    '③  Compute pseudo label for selected K stations\n'
    'source ∈ { self  |  kriging  |  hybrid (0.5/0.5) }',
    fontsize=10)

arrow(7.4, 2.75, 7.4, 2.3)

# Step 4 — Retrain
box(4.3, 1.45, 6.2, 0.85,
    '④  Warm-start retrain (lr=1e-4, 200 epoch)\n'
    'L = Huber(50 train)  +  0.5 × Huber(cumulative pseudo)',
    fontsize=10)

# Loop-back arrow (bottom: cumulative grows)
arrow(7.4, 1.45, 7.4, 1.05)
ax.text(7.45, 0.85,
        'cumulative pseudo: 0 → 40 → 80 → 120 → 160 → 200',
        fontsize=9.5, style='italic', color='#444', ha='center')
ax.text(7.4, 0.55, '↻  next round', fontsize=10, ha='center',
        color=LOOP_EDGE, weight='bold')

# === Output box (final) ===
box(0.3, 2.7, 2.7, 0.95,
    'After R5:\nbest valid RMSE\n( e.g. 0.0428 for hybrid )',
    fontsize=10, bold=True, color=ACCENT)

# Arrow from loop bottom-left to output
arrow(3.9, 2.5, 3.0, 2.7, color=ACCENT)
ax.text(2.0, 2.2, 'best ckpt across\nrounds returned',
        fontsize=9, style='italic', color=ACCENT, ha='center')

# === Side note: 3-metric & pseudo source ===
ax.text(0.3, 1.7, 'Selection metrics:', fontsize=10, weight='bold')
ax.text(0.3, 1.4,
        '• confidence (neighbor / conformal / kriging_struct)\n'
        '• diversity (max dist to selected ∪ train)\n'
        '• valid-relevance (closeness to valid emb)',
        fontsize=9, va='top')

plt.tight_layout()
plt.savefig(OUT, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}")
