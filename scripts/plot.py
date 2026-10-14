import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/kulfo_gwsa_forecasts.csv', parse_dates=['date'])
df = df.sort_values('date')

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df['date'], df['GWSA_predicted_mm'], 'o-', color='steelblue')
ax.axhline(df['GWSA_predicted_mm'].mean(), color='red', linestyle='--', label='Mean')
ax.set_title('Kulfo GWSA Predictions (2018–2025)')
ax.set_ylabel('GWSA (mm)')
ax.set_xlabel('Date')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('data/kulfo_gwsa_plot.png', dpi=150)
print("Saved: data/kulfo_gwsa_plot.png")