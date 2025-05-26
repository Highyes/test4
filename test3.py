# test.py

import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta, date
import requests
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

SERVICE_KEY    = "pEbdNSKcRZaG3TUinCWWmQ"
MODEL_DIR      = "/home/iwh/season_models_final"
HIST_DATA_PATH = "/mnt/c/Users/user/Desktop/3학기/딥러닝/term project/deep learning data_revised.csv"

EXOG_COLS = ["air_temp", "precipitable_water", "humidity"]

@st.cache_data
def load_historical(path):
    df = pd.read_csv(path, parse_dates=["period"])
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)
        df["period"] += pd.to_timedelta(df["hour"], unit="h")
    df["dayofweek"] = df["period"].dt.dayofweek
    df = df.set_index("period").sort_index()
    return df[~df.index.duplicated(keep="first")]

hist_df = load_historical(HIST_DATA_PATH)

@st.cache_resource
def load_resources():
    models, scalers = {}, {}
    for s in [1,2,3,4]:
        mpath = os.path.join(MODEL_DIR, f"season{s}_best.h5")
        spath = os.path.join(MODEL_DIR, f"scaler_season{s}.pkl")
        if not os.path.exists(mpath) or not os.path.exists(spath):
            st.error(f"Missing files for season{s}:\n  {mpath}\n  {spath}")
            st.stop()
        models[s]  = load_model(mpath, compile=False)
        scalers[s] = pickle.load(open(spath,"rb"))
    return models, scalers

models, scalers = load_resources()

@st.cache_data(ttl=3600)
def fetch_kma(nx, ny):
    now = datetime.utcnow() + timedelta(hours=9)
    bd  = now.strftime("%Y%m%d")
    bt  = f"{(now.hour//3)*3:02d}00"
    url = "http://apis.data.go.kr/1360000/VilageFcstInfoService/getUltraSrtNcst"
    params = {
        "serviceKey": SERVICE_KEY,
        "pageNo":     "1",
        "numOfRows":  "1000",
        "dataType":   "JSON",
        "base_date":  bd,
        "base_time":  bt,
        "nx":         nx,
        "ny":         ny,
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    try:
        items = r.json()["response"]["body"]["items"]["item"]
    except:
        return pd.DataFrame()
    df = pd.DataFrame(items)
    df["dt"] = pd.to_datetime(
        df["baseDate"] + df["baseTime"],
        format="%Y%m%d%H%M",
        errors="coerce"
    )
    df = (
        df[df["category"].isin(["T1H","RN1","REH"])]
          .pivot(index="dt", columns="category", values="obsrValue")
    )
    return df.rename(
        columns={
            "T1H": "air_temp",
            "RN1": "precipitable_water",
            "REH": "humidity"
        }
    )

st.title("전력 수요 예측 데모")
col1, col2 = st.columns(2)
with col1:
    selected_date = st.date_input("예측 기준일", date.today())
with col2:
    nx = st.number_input("격자 X (nx)", value=33, format="%d")
    ny = st.number_input("격자 Y (ny)", value=127, format="%d")

if st.button("예측 실행"):
    start   = pd.to_datetime(selected_date)
    periods = pd.date_range(start+timedelta(hours=1), periods=24, freq="h")

    hist_exog = hist_df[EXOG_COLS].reindex(periods)
    api_raw   = fetch_kma(nx, ny).reindex(periods)
    api_exog  = api_raw[EXOG_COLS] if not api_raw.empty else pd.DataFrame(index=periods, columns=EXOG_COLS)
    exog_df   = hist_exog.combine_first(api_exog)
    if exog_df.isna().any().any():
        st.warning("일부 외생변수가 누락되어 과거 전체 평균으로 대체합니다.")
        means     = hist_df[EXOG_COLS].mean()
        exog_df   = exog_df.fillna(means.to_dict())

    exog_df["dayofweek"] = selected_date.weekday()

    dummy = np.zeros((24,1))
    mat   = exog_df[ EXOG_COLS + ["dayofweek"] ].values   
    arr   = np.hstack([dummy, mat])                       

    season = ((selected_date.month%12+3)//3)
    scaler = scalers[season]
    sc     = scaler.transform(arr)                       
    X_in   = sc[:,1:][np.newaxis,...]                    

    y_s    = models[season].predict(X_in)[0,:,0]
    mn, mx = scaler.data_min_[0], scaler.data_max_[0]
    y_pred = y_s * (mx - mn) + mn

    out = pd.DataFrame({
        "period": periods,
        "predicted_demand(MWh)": y_pred
    }).set_index("period")

    st.subheader("24시간 전력 수요 예측 결과")
    st.dataframe(out)
    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(out.index, out["predicted_demand(MWh)"], marker="o")
    ax.set_xlabel("Time"); ax.set_ylabel("Demand (MWh)"); ax.grid(True)
    st.pyplot(fig)