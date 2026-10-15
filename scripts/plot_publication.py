"""
Publication-quality figure for Kulfo GWSA prediction.
Panel (a): Time series with observed, Ridge, and TNS
Panel (b): Scatter (observed vs Ridge and TNS)
Panel (c): Model comparison bar chart
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================
# 1. LOAD DATA
# ============================================================
forecasts = pd.read_csv('data/kulfo_gwsa_forecasts.csv', parse_dates=['date'])
forecasts = forecasts.sort_values('date').reset_index(drop=True)

train = pd.read_csv('data/kulfo_training_data.csv', parse_dates=['date'])
train = train.sort_values('date').reset_index(drop=True)

# Model metrics
RIDGE_R2   = 0.890
RIDGE_RMSE = 36.41
RIDGE_CORR = 0.944
TNS_R2     = 0.334
TNS_RMSE   = 88.43
TNS_CORR   = 0.696

# ============================================================
# 2. SETUP FIGURE
# ============================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.linewidth': 0.8,
    'legend.fontsize': 8,
    'legend.frameon': False,
})

fig = plt.figure(figsize=(7.0, 7.0), dpi=300)
gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 0.8], hspace=0.40, wspace=0.35)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# Okabe-Ito colourblind-friendly palette
C_OBS   = '#0072B2'   # blue    — observed
C_RIDGE = '#D55E00'   # orange  — Ridge
C_TNS   = '#009E73'   # green   — TNS

# ============================================================
# 3. PANEL A — TIME SERIES WITH BOTH MODELS
# ============================================================
# Ridge ± RMSE band
ax1.fill_between(forecasts['date'],
                 forecasts['GWSA_predicted_mm'] - RIDGE_RMSE,
                 forecasts['GWSA_predicted_mm'] + RIDGE_RMSE,
                 color=C_RIDGE, alpha=0.12, zorder=2,
                 label=f'Ridge ± RMSE ({RIDGE_RMSE:.1f} mm)')

# Observed
ax1.plot(train['date'], train['GWSA'],
         color=C_OBS, linewidth=1.6, label='Observed (GLDAS)', zorder=5)

# Ridge
ax1.plot(forecasts['date'], forecasts['GWSA_predicted_mm'],
         color=C_RIDGE, linewidth=1.6, linestyle='--',
         label=f'Ridge (R² = {RIDGE_R2:.3f})', zorder=4)

# TNS (only where available)
tns_valid = forecasts.dropna(subset=['TNS_predicted_mm'])
if len(tns_valid) > 0:
    ax1.plot(tns_valid['date'], tns_valid['TNS_predicted_mm'],
             color=C_TNS, linewidth=1.4, linestyle='-.',
             label=f'TNS (R² = {TNS_R2:.3f})', zorder=3)

# Mean baseline
ax1.axhline(train['GWSA'].mean(), color='gray', linewidth=0.7,
            linestyle=':', alpha=0.7, zorder=1,
            label=f'Mean ({train["GWSA"].mean():.1f} mm)')

# Training end marker
split_date = train['date'].max()
ax1.axvline(split_date, color='black', linewidth=0.8,
            linestyle='--', alpha=0.7, zorder=1)
ax1.text(split_date, ax1.get_ylim()[1] * 0.95,
         '  Training end', fontsize=7, color='black',
         va='top', ha='left')

ax1.set_ylabel('GWSA (mm)')
ax1.set_xlabel('Date')
ax1.set_title('(a) Groundwater Storage Anomaly — Observed vs Ridge vs TNS',
              loc='left', fontweight='bold')
ax1.legend(loc='upper left', ncol=3, columnspacing=1.0)
ax1.grid(True, alpha=0.25, linewidth=0.5)
ax1.set_xlim(forecasts['date'].min(), forecasts['date'].max())

# ============================================================
# 4. PANEL B — SCATTER (OBSERVED VS PREDICTED)
# ============================================================
merged = pd.merge(
    train[['date', 'GWSA']],
    forecasts[['date', 'GWSA_predicted_mm', 'TNS_predicted_mm']],
    on='date', how='inner'
).dropna(subset=['GWSA', 'GWSA_predicted_mm'])

if len(merged) > 5:
    obs = merged['GWSA'].values
    pred_ridge = merged['GWSA_predicted_mm'].values

    ax2.scatter(obs, pred_ridge, s=18, alpha=0.65, color=C_RIDGE,
                edgecolor='white', linewidth=0.4, zorder=3,
                label=f'Ridge (R²={RIDGE_R2:.3f})')

    # TNS scatter only for rows where both exist
    merged_tns = merged.dropna(subset=['TNS_predicted_mm'])
    if len(merged_tns) > 5:
        obs_tns = merged_tns['GWSA'].values
        pred_tns = merged_tns['TNS_predicted_mm'].values
        ax2.scatter(obs_tns, pred_tns, s=18, alpha=0.55, color=C_TNS,
                    edgecolor='white', linewidth=0.4, zorder=2,
                    label=f'TNS (R²={TNS_R2:.3f})')
        lims = [min(obs.min(), pred_ridge.min(), pred_tns.min()),
                max(obs.max(), pred_ridge.max(), pred_tns.max())]
    else:
        lims = [min(obs.min(), pred_ridge.min()),
                max(obs.max(), pred_ridge.max())]

    ax2.plot(lims, lims, 'k--', linewidth=0.8, zorder=4, label='1:1 line')

    ax2.text(0.05, 0.95,
             f'Ridge: R²={RIDGE_R2:.3f}\n       RMSE={RIDGE_RMSE:.2f} mm\n'
             f'TNS:    R²={TNS_R2:.3f}\n       RMSE={TNS_RMSE:.2f} mm',
             transform=ax2.transAxes, va='top', fontsize=7,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='gray', linewidth=0.5))
else:
    ax2.text(0.5, 0.5, 'Insufficient overlap\nfor validation',
             transform=ax2.transAxes, ha='center', va='center')

ax2.set_xlabel('Observed GWSA (mm)')
ax2.set_ylabel('Predicted GWSA (mm)')
ax2.set_title('(b) Validation', loc='left', fontweight='bold')
ax2.legend(loc='lower right', fontsize=7)
ax2.grid(True, alpha=0.25, linewidth=0.5)
ax2.set_aspect('equal', adjustable='box')

# ============================================================
# 5. PANEL C — MODEL COMPARISON
# ============================================================
models = ['Ridge', 'TNS']
r2_values   = [RIDGE_R2, TNS_R2]
rmse_values = [RIDGE_RMSE, TNS_RMSE]

x = np.arange(len(models))
width = 0.35

ax3_twin = ax3.twinx()

bars1 = ax3.bar(x - width/2, r2_values, width,
                color=C_RIDGE, label='R²',
                edgecolor='black', linewidth=0.5)
bars2 = ax3_twin.bar(x + width/2, rmse_values, width,
                     color=C_TNS, label='RMSE (mm)',
                     edgecolor='black', linewidth=0.5)

for i, v in enumerate(r2_values):
    ax3.text(i - width/2, v + 0.02, f'{v:.3f}',
             ha='center', fontsize=7.5, fontweight='bold')
for i, v in enumerate(rmse_values):
    ax3_twin.text(i + width/2, v + 2, f'{v:.1f}',
                  ha='center', fontsize=7.5, fontweight='bold')

ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.set_ylabel('R²', color=C_RIDGE, fontweight='bold')
ax3_twin.set_ylabel('RMSE (mm)', color=C_TNS, fontweight='bold')
ax3.set_ylim(0, 1.05)
ax3_twin.set_ylim(0, max(rmse_values) * 1.2)

ax3.set_title('(c) Model Comparison', loc='left', fontweight='bold')
ax3.grid(True, alpha=0.25, linewidth=0.5, axis='y')

lines1, labels1 = ax3.get_legend_handles_labels()
lines2, labels2 = ax3_twin.get_legend_handles_labels()
ax3.legend(lines1 + lines2, labels1 + labels2,
           loc='upper right', fontsize=7.5)

# ============================================================
# 6. SAVE
# ============================================================
out_png = 'data/kulfo_gwsa_publication.png'
out_pdf = 'data/kulfo_gwsa_publication.pdf'

plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
plt.close()

print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")