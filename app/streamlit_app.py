import glob
import json
import os

import pandas as pd
import streamlit as st

st.set_page_config(page_title="AutoRAG-Live Dashboard", layout="wide")

st.title("AutoRAG-Live Dashboard")

runs_dir = "runs"
os.makedirs(runs_dir, exist_ok=True)
run_files = sorted(glob.glob(os.path.join(runs_dir, "*.json")))

if not run_files:
    st.info("No runs found. Run 'autorag eval --suite small' to generate a run.")
else:
    selected = st.selectbox("Select run", run_files[::-1])
    with open(selected) as f:
        data = json.load(f)
    st.subheader("Summary")
    st.json(data["metrics"])

    st.subheader("Per-Item Results")
    df = pd.DataFrame(data["results"])  # type: ignore[arg-type]
    st.dataframe(df)
