"""
Monthly GWSA prediction for Kulfo watershed.
Runs Ridge AND TNS models. Auto-detects latest month. Mean-fills missing predictors.
"""

import os
import sys
import io
import json
import warnings
warnings.filterwarnings('ignore')

# ---- Suppress BOTH stdout and stderr during Earth Engine import and init ----
_stdout = sys.stdout
_stderr = sys.stderr
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()

import pandas as pd
import numpy as np
import joblib
import ee
import torch
import torch.nn as nn

creds_json = os.environ.get('EARTHENGINE_TOKEN')
if creds_json is None:
    sys.stdout = _stdout
    sys.stderr = _stderr
    raise ValueError("EARTHENGINE_TOKEN environment variable not set")

creds = json.loads(creds_json)
credentials = ee.ServiceAccountCredentials(
    creds['client_email'],
    key_data=creds_json
)
ee.Initialize(credentials, project='gothic-doodad-356708')

sys.stdout = _stdout
sys.stderr = _stderr

# ============================================================
# 2. LOAD MODELS AND TRAINING DATA
# ============================================================
model    = joblib.load('models/kulfo_ridge_model.pkl')
pca      = joblib.load('models/kulfo_pca.pkl')
scaler   = joblib.load('models/kulfo_scaler.pkl')
features = joblib.load('models/kulfo_features.pkl')

train = pd.read_csv('data/kulfo_training_data.csv')
train['date'] = pd.to_datetime(train['date'])
train = train.sort_values('date').reset_index(drop=True)

# TNS architecture
class TinyTNS(nn.Module):
    def __init__(self, n_features, channels=16):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, channels, 3, padding=1, dilation=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=2, dilation=2)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(channels, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.dropout(x[:, :, -1])
        return self.head(x)

tns_model = TinyTNS(n_features=len(features))
tns_model.load_state_dict(torch.load('models/kulfo_tns_model.pt', map_location='cpu'))
tns_model.eval()

# ============================================================
# 3. AUTO-DETECT MOST RECENT AVAILABLE MONTH
# ============================================================
kulfo = ee.FeatureCollection('projects/gothic-doodad-356708/assets/kulfo')
geom = kulfo.geometry()

def latest_available_month(collection, band_name):
    latest = collection.select(band_name).limit(1, 'system:time_start', False).first()
    ts = latest.get('system:time_start').getInfo()
    end_date = pd.Timestamp(ts, unit='ms', tz='UTC').tz_localize(None)
    return (end_date.replace(day=1) - pd.DateOffset(months=1)).replace(day=1)

chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
era5   = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
modis_et = ee.ImageCollection('MODIS/061/MOD16A2GF')
modis_ndvi = ee.ImageCollection('MODIS/061/MOD13A2')

candidates = [
    latest_available_month(chirps, 'precipitation'),
    latest_available_month(era5, 'temperature_2m'),
    latest_available_month(modis_et, 'ET'),
    latest_available_month(modis_ndvi, 'NDVI'),
    latest_available_month(era5, 'volumetric_soil_water_layer_1'),
]

START = pd.Timestamp('2018-01-01')
END = max(candidates)
months = pd.date_range(START, END, freq='MS')

# ============================================================
# 4. BULK FETCH
# ============================================================
def bulk_fetch(collection, band_name, scale):
    n = len(months)

    def per_month(idx):
        idx = ee.Number(idx)
        m_start = ee.Date(START.strftime('%Y-%m-%d')).advance(idx, 'month')
        m_end = m_start.advance(1, 'month')
        return collection.filterDate(m_start, m_end).select(band_name).mean().set('month_idx', idx)

    monthly_ic = ee.ImageCollection(ee.List.sequence(0, n - 1).map(per_month))
    stacked = monthly_ic.toBands()

    stats = stacked.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=geom,
        scale=scale,
        maxPixels=1e9,
        bestEffort=True,
        tileScale=4
    ).getInfo()

    if stats is None:
        return {}

    values = {}
    for i in range(n):
        keys = [k for k in stats.keys() if k.startswith(f'{i}_')]
        values[i] = stats[keys[0]] if keys else None
    return values

def build_df(collection, band_name, scale, out_col):
    values = bulk_fetch(collection, band_name, scale)
    return pd.DataFrame({
        'date': months,
        out_col: [values.get(i, None) for i in range(len(months))]
    })

rain_df = build_df(chirps, 'precipitation', 5000, 'rainfall')
temp_df = build_df(era5, 'temperature_2m', 11132, 'temperature_2m')
et_df   = build_df(modis_et, 'ET', 500, 'ET')
ndvi_df = build_df(modis_ndvi, 'NDVI', 1000, 'NDVI')
sm_df   = build_df(era5, 'volumetric_soil_water_layer_1', 11132, 'volumetric_soil_water_layer_1')

env = rain_df.merge(temp_df, on='date') \
             .merge(et_df, on='date') \
             .merge(ndvi_df, on='date') \
             .merge(sm_df, on='date')

# ============================================================
# 5. BUILD FEATURE TABLE
# ============================================================
gwsa_history = train[['date', 'GWSA']].copy()
rows = []

predictor_cols = ['rainfall', 'temperature_2m', 'ET', 'NDVI', 'volumetric_soil_water_layer_1']
col_means = {c: env[c].mean() for c in predictor_cols}

for _, row in env.iterrows():
    d = row['date']

    for c in predictor_cols:
        if pd.isna(row[c]):
            row[c] = col_means[c]

    gwsa_past = gwsa_history[gwsa_history['date'] < d].sort_values('date').tail(3)
    if len(gwsa_past) < 3:
        continue

    GWSA_lag1 = gwsa_past.iloc[-1]['GWSA']
    GWSA_lag2 = gwsa_past.iloc[-2]['GWSA']
    GWSA_lag3 = gwsa_past.iloc[-3]['GWSA']

    prev = env[env['date'] < d].sort_values('date')
    rain_lag1 = prev.iloc[-1]['rainfall'] if len(prev) >= 1 and pd.notna(prev.iloc[-1]['rainfall']) else col_means['rainfall']
    rain_lag2 = prev.iloc[-2]['rainfall'] if len(prev) >= 2 and pd.notna(prev.iloc[-2]['rainfall']) else col_means['rainfall']
    rain_lag3 = prev.iloc[-3]['rainfall'] if len(prev) >= 3 and pd.notna(prev.iloc[-3]['rainfall']) else col_means['rainfall']
    ET_lag1   = prev.iloc[-1]['ET'] if len(prev) >= 1 and pd.notna(prev.iloc[-1]['ET']) else col_means['ET']
    ET_lag2   = prev.iloc[-2]['ET'] if len(prev) >= 2 and pd.notna(prev.iloc[-2]['ET']) else col_means['ET']
    NDVI_lag1 = prev.iloc[-1]['NDVI'] if len(prev) >= 1 and pd.notna(prev.iloc[-1]['NDVI']) else col_means['NDVI']
    SM_lag1   = prev.iloc[-1]['volumetric_soil_water_layer_1'] if len(prev) >= 1 and pd.notna(prev.iloc[-1]['volumetric_soil_water_layer_1']) else col_means['volumetric_soil_water_layer_1']

    month_num = d.month
    rows.append({
        'date': d,
        'rainfall': row['rainfall'],
        'temperature_2m': row['temperature_2m'],
        'ET': row['ET'],
        'NDVI': row['NDVI'],
        'volumetric_soil_water_layer_1': row['volumetric_soil_water_layer_1'],
        'GWSA_lag1': GWSA_lag1, 'GWSA_lag2': GWSA_lag2, 'GWSA_lag3': GWSA_lag3,
        'rain_lag1': rain_lag1, 'rain_lag2': rain_lag2, 'rain_lag3': rain_lag3,
        'ET_lag1': ET_lag1, 'ET_lag2': ET_lag2,
        'NDVI_lag1': NDVI_lag1, 'SM_lag1': SM_lag1,
        'sin_month': np.sin(2*np.pi*month_num/12),
        'cos_month': np.cos(2*np.pi*month_num/12),
    })

feature_df = pd.DataFrame(rows).reset_index(drop=True)

if len(feature_df) == 0:
    print("No predictions produced. Check Earth Engine data availability.")
    exit(1)

# ============================================================
# 6. RIDGE PREDICTION
# ============================================================
X = feature_df[features].values
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)
feature_df['GWSA_predicted_mm'] = model.predict(X_pca)

# ============================================================
# 7. TNS PREDICTION
# ============================================================
LOOKBACK = 3

X_seq = []
for i in range(len(X_scaled) - LOOKBACK):
    X_seq.append(X_scaled[i:i+LOOKBACK])
X_seq = np.array(X_seq, dtype=np.float32)

feature_df['TNS_predicted_mm'] = np.nan

if len(X_seq) > 0:
    with torch.no_grad():
        tns_preds = tns_model(torch.tensor(X_seq)).numpy().flatten()

    tns_dates = feature_df['date'].iloc[LOOKBACK:].values
    feature_df.loc[feature_df['date'].isin(tns_dates), 'TNS_predicted_mm'] = tns_preds

# ============================================================
# 8. SAVE
# ============================================================
out_path = 'data/kulfo_gwsa_forecasts.csv'
feature_df[['date', 'GWSA_predicted_mm', 'TNS_predicted_mm']].to_csv(out_path, index=False)

print(f"Saved {len(feature_df)} predictions to {out_path}")