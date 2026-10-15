# Kulfo GWSA Monitor

Monthly groundwater storage anomaly (GWSA) prediction for the Kulfo watershed, Ethiopia.

## Data

Latest predictions: [`data/kulfo_gwsa_forecasts.csv`](data/kulfo_gwsa_forecasts.csv)

Columns:
- `date` — month (YYYY-MM-01)
- `GWSA_predicted_mm` — predicted groundwater storage anomaly (mm)

## How It Works

1. GitHub Actions runs on the 1st of every month at 06:00 UTC
2. Fetches CHIRPS rainfall, ERA5 temperature/soil moisture, MODIS ET/NDVI from Google Earth Engine
3. Runs a Ridge regression model (R² = 0.890) trained on 2018–2024 data
4. Commits the updated CSV back to this repo

## Use the Data

Download directly:
https://raw.githubusercontent.com/DemisoDaba/kulfo-gw-monitor/main/data/kulfo_gwsa_forecasts.csv

## Run It Yourself

```bash
git clone https://github.com/DemisoDaba/kulfo-gw-monitor.git
cd kulfo-gw-monitor
pip install -r requirements.txt
export EARTHENGINE_TOKEN="your-service-account-json-on-one-line"
python scripts/predict.py