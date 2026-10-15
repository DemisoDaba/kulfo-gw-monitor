# Kulfo GWSA Monitor

Monthly groundwater storage anomaly (GWSA) prediction for the Kulfo watershed, Ethiopia.

## Data

Latest predictions: [`data/kulfo_gwsa_forecasts.csv`](data/kulfo_gwsa_forecasts.csv)

Columns:
- `date`: month (YYYY-MM-01)
- `GWSA_predicted_mm` : Ridge-predicted groundwater storage anomaly (mm)
- `TNS_predicted_mm` : TNS-predicted groundwater storage anomaly (mm)

Publication figure:
- [`data/kulfo_gwsa_publication.png`](data/kulfo_gwsa_publication.png) — 300 DPI raster

## How It Works

1. GitHub Actions runs on the 1st of every month at 06:00 UTC
2. Fetches CHIRPS rainfall, ERA5 temperature/soil moisture, MODIS ET/NDVI from Google Earth Engine
3. Runs a Ridge regression model (R² = 0.899) and a TNS model (R² = 0.340) trained on 2018–2024 data
4. Generates a publication figure
5. Commits the updated CSV and figure back to this repo

## Model Performance

| Model | R² | RMSE (mm) | Correlation |
| :--- | :--- | :--- | :--- |
| Ridge (α=10) | 0.899 | 34.44 | 0.948 |
| TNS (no PCA) | 0.340 | 89.65 | 0.714 |


## Use the Data

Download directly: [data/kulfo_gwsa_forecasts.csv](https://raw.githubusercontent.com/DemisoDaba/kulfo-gw-monitor/main/data/kulfo_gwsa_forecasts.csv)

## Run It Yourself

```bash
git clone https://github.com/DemisoDaba/kulfo-gw-monitor.git
```
```bash
cd kulfo-gw-monitor
```
```bash
pip install -r requirements.txt
```
```bash
export EARTHENGINE_TOKEN="your-service-account-json-on-one-line"
python scripts/predict.py
python scripts/plot_publication.py
```

## License

This project is licensed under the MIT License see the [LICENSE](LICENSE) file for details.
