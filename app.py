from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import pydeck as pdk


st.set_page_config(page_title="FloodGuard Calaitan", page_icon="🌧️", layout="wide")

LATITUDE = 8.7900
LONGITUDE = 125.7450
LOCATION = "Barangay Calaitan, Bayugan City, Agusan del Sur"
RISK_COLORS = {
    "Very Low": [34, 197, 94, 190],
    "Low": [132, 204, 22, 190],
    "Moderate": [234, 179, 8, 200],
    "High": [249, 115, 22, 210],
    "Severe": [220, 38, 38, 220],
}


@st.cache_data(ttl=600, show_spinner=False)
def load_weather() -> tuple[dict[str, Any], bool, str]:
    """Load Open-Meteo data, with a clearly labelled fallback for offline demos."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "hourly": "precipitation,soil_moisture_0_to_7cm,temperature_2m,relative_humidity_2m",
        "forecast_days": 2,
        "timezone": "Asia/Manila",
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json(), True, "Live Open-Meteo data"
    except requests.RequestException:
        now = pd.Timestamp.now(tz="Asia/Manila").tz_localize(None).floor("h")
        hours = pd.date_range(now, periods=48, freq="h")
        rng = np.random.default_rng(7)
        rain = np.maximum(0, rng.normal(3.2, 5.0, len(hours)))
        return {
            "current": {"temperature_2m": 27.0, "relative_humidity_2m": 82, "precipitation": 2.4, "wind_speed_10m": 8.0},
            "hourly": {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist(), "precipitation": rain.tolist(), "soil_moisture_0_to_7cm": np.full(len(hours), 0.31).tolist(), "temperature_2m": np.full(len(hours), 27.0).tolist(), "relative_humidity_2m": np.full(len(hours), 82).tolist()},
        }, False, "Offline demo data — connect to Open-Meteo to refresh"


def risk_level(score: int) -> str:
    if score >= 81:
        return "Severe"
    if score >= 61:
        return "High"
    if score >= 41:
        return "Moderate"
    if score >= 21:
        return "Low"
    return "Very Low"


def calculate_risk(weather: dict[str, Any], rainfall_multiplier: float, terrain_factor: float) -> tuple[int, dict[str, float]]:
    current = weather["current"]
    hourly = weather["hourly"]
    rain = np.asarray(hourly["precipitation"][:24], dtype=float)
    rain_1h = float(current.get("precipitation", rain[0]))
    rain_3h = float(rain[:3].sum())
    rain_24h = float(rain.sum())
    soil = float(np.mean(hourly.get("soil_moisture_0_to_7cm", [0.3])[:24]))
    # Screening index only. Thresholds are explicit so they can later be replaced
    # by calibrated, event-labelled model output.
    components = {
        "Rainfall intensity": min(35, rain_1h * 1.3) * rainfall_multiplier,
        "Accumulated rainfall": min(25, rain_24h / 4.0) * rainfall_multiplier,
        "Soil moisture": min(15, soil * 35),
        "Terrain susceptibility": min(25, 25 * terrain_factor),
    }
    score = int(round(min(100, sum(components.values()))))
    return score, components


def build_map(score: int) -> pdk.Deck:
    level = risk_level(score)
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{"lat": LATITUDE, "lon": LONGITUDE, "risk": score, "color": RISK_COLORS[level]}]),
        get_position="[lon, lat]",
        get_fill_color="color",
        get_radius=2600,
        pickable=True,
    )
    marker = pdk.Layer("ScatterplotLayer", data=pd.DataFrame([{"lat": LATITUDE, "lon": LONGITUDE}]), get_position="[lon, lat]", get_fill_color=[255, 255, 255, 255], get_radius=180)
    return pdk.Deck(layers=[layer, marker], initial_view_state=pdk.ViewState(latitude=LATITUDE, longitude=LONGITUDE, zoom=11), map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json", tooltip={"text": f"{LOCATION}\nRisk score: {score} ({level})"})


st.markdown("# 🌧️ FloodGuard Calaitan")
st.caption("GIS-integrated flash-flood risk screening dashboard · Research prototype")

weather, live, source_label = load_weather()
score, components = calculate_risk(weather, 1.0, 0.72)
level = risk_level(score)
current = weather["current"]
hourly = pd.DataFrame(weather["hourly"])
hourly["time"] = pd.to_datetime(hourly["time"])

with st.sidebar:
    st.header("Controls")
    rainfall_multiplier = st.slider("Rainfall sensitivity", 0.6, 1.5, 1.0, 0.05)
    terrain_factor = st.slider("Terrain susceptibility", 0.2, 1.0, 0.72, 0.01)
    st.divider()
    st.markdown("**Model status**")
    st.info("Screening index active")
    st.caption("Chaos/LSTM forecasting is not activated until historical flood labels and validation metrics are available.")
    if st.button("Refresh weather"):
        st.cache_data.clear()
        st.rerun()

score, components = calculate_risk(weather, rainfall_multiplier, terrain_factor)
level = risk_level(score)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Risk score", f"{score}/100", level)
col2.metric("Temperature", f"{current.get('temperature_2m', 0):.1f} °C")
col3.metric("Rainfall (24h)", f"{hourly['precipitation'].head(24).sum():.1f} mm")
col4.metric("Humidity", f"{current.get('relative_humidity_2m', 0):.0f}%")

st.warning(f"{source_label}. This score is a screening indicator, not an official warning.") if not live else st.success(f"{source_label} · Updated {datetime.now(timezone.utc).astimezone().strftime('%H:%M')}")

left, right = st.columns([2.1, 1])
with left:
    st.subheader("Flash-flood risk map")
    st.pydeck_chart(build_map(score), use_container_width=True)
    st.caption(f"Map center: {LOCATION} · Coordinates used: {LATITUDE:.4f}, {LONGITUDE:.4f}. Replace with verified GIS boundary layers before deployment.")
with right:
    st.subheader("Current assessment")
    st.markdown(f"## {level}")
    st.progress(score / 100)
    for name, value in components.items():
        st.write(f"**{name}** · {value:.1f}/35")
    if level in {"High", "Severe"}:
        st.error("Prepare local alerts and confirm evacuation routes with authorities.")
    elif level == "Moderate":
        st.warning("Continue monitoring rainfall and local stream conditions.")
    else:
        st.success("No elevated screening signal from the available inputs.")

st.subheader("Weather and rainfall outlook")
chart = hourly.head(24).set_index("time")[["precipitation", "soil_moisture_0_to_7cm"]].rename(columns={"precipitation": "Rainfall (mm)", "soil_moisture_0_to_7cm": "Soil moisture"})
st.line_chart(chart)

with st.expander("Research and data notes"):
    st.markdown("""
    - **Current implementation:** transparent weighted screening index using live weather inputs and an adjustable terrain factor.
    - **Required next data layer:** verified DEM, slope, drainage, land-cover, soil, and river-network layers for Calaitan.
    - **Required validation dataset:** timestamped rainfall/soil/weather observations paired with confirmed flood or no-flood outcomes.
    - **Next modelling stage:** benchmark the screening index against persistence, Random Forest/XGBoost, and then validated phase-space/LSTM models using RMSE, MAE, NSE, precision, recall, and lead time.
    """)
