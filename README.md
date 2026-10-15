# Kulfo GWSA Monitor

Monthly groundwater storage anomaly (GWSA) prediction for the Kulfo watershed, Ethiopia.

## Data

Latest predictions: [`data/kulfo_gwsa_forecasts.csv`](data/kulfo_gwsa_forecasts.csv)

Columns:
- `date`: month (YYYY-MM-01)
- `GWSA_predicted_mm` : Ridge-predicted groundwater storage anomaly (mm)
- `TNS_predicted_mm` : TNS-predicted groundwater storage anomaly (mm)

Publication figure:
- [`data/kulfo_gwsa_publication.png`](data/kulfo_gwsa_publication.png) : 300 DPI raster

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


## Two Ways to Use This

### 1. Basin-average time series (automatic)

The file [`data/kulfo_gwsa_forecasts.csv`](data/kulfo_gwsa_forecasts.csv) is updated automatically every month Just download it:

```
https://raw.githubusercontent.com/DemisoDaba/kulfo-gw-monitor/main/data/kulfo_gwsa_forecasts.csv
```

### 2. Point-specific time series (manual)

Get monthly GWSA for any lat/lon inside the Kulfo watershed. Requires your own Google Earth Engine service account.

**One-time setup:**

1. Register at https://code.earthengine.google.com
2. Create a service account with the **Earth Engine Resource Viewer** role
3. Download the JSON key and minify it to one line

**Run a point prediction:**

```bash
git clone https://github.com/DemisoDaba/kulfo-gw-monitor.git
cd kulfo-gw-monitor
pip install -r requirements.txt
export EARTHENGINE_TOKEN="your-service-account-json-on-one-line"
python scripts/predict_point.py <lat> <lon>
```

Example:

```bash
python scripts/predict_point.py 6.15 37.45
```

Output: `data/kulfo_point_6.1500_37.4500.csv`

### 3. Basin-average prediction (manual, for developers)

To regenerate the basin CSV locally:

```bash
python scripts/predict.py
python scripts/plot_publication.py
```

## License

This project is licensed under the MIT License see the [LICENSE](LICENSE) file for details.
