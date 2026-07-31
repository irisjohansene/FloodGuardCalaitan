# FloodGuard Calaitan

A local-first Python/Streamlit research prototype for the GIS-integrated flash-flood dashboard described in Chapter One.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The application uses Open-Meteo when available and falls back to clearly labelled deterministic demo data when offline. The current score is an interpretable screening index, not a validated flood forecast.

## Research implementation boundary

The dashboard is intentionally usable before the research dataset is complete. Advanced chaos and LSTM components should only be enabled after historical event labels, verified spatial layers, and held-out validation results are available.
