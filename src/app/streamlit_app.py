# streamlit_app.py — the deployed web interface

import streamlit as st
import pandas as pd
import joblib
import yaml
from pathlib import Path

st.set_page_config(page_title="Joburg Property Intelligence", page_icon="🏠")


@st.cache_resource
def load_model():
    """Load the trained model from disk. Cached so it only loads once."""
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    path = Path(config["model"]["model_path"])
    if not path.exists():
        return None, None
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_names"]


@st.cache_data
def load_data():
    """Load the processed features CSV for comparables chart."""
    path = Path("data/processed/features.csv")
    if path.exists():
        return pd.read_csv(path)
    return None


model, feature_names = load_model()
df = load_data()

st.title("🏠 Johannesburg Property Intelligence Engine")
st.markdown(
    "Enter a property's details below to get a predicted fair market price "
    "based on suburb, size, and features."
)

# sidebar inputs
st.sidebar.header("Property details")
area      = st.sidebar.number_input("Size (sqm)", min_value=10, max_value=5000, value=120)
bedrooms  = st.sidebar.number_input("Bedrooms",   min_value=0,  max_value=10,   value=3)
bathrooms = st.sidebar.number_input("Bathrooms",  min_value=0,  max_value=10,   value=2)

suburbs = sorted([
    "Wilgeheuwel", "Honeydew", "Ruimsig", "Northgate", "Florida",
    "Roodepoort", "Constantia Kloof", "Weltevreden Park",
    "Little Falls", "Radiokop", "Fleurhof", "Discovery",
    "Kenmare", "Laser Park", "Helderkruin",
])
location  = st.sidebar.selectbox("Suburb", suburbs)
prop_type = st.sidebar.selectbox(
    "Property type", ["House", "Apartment", "Townhouse", "Cluster", "Simplex"]
)

# prediction
if st.sidebar.button("Predict fair price"):
    if model is None:
        st.error("Model not trained yet. Run `make run` first, then redeploy.")
    else:
        # build one row of zeros matching the model's exact column order
        input_data = {col: 0 for col in feature_names}
        input_data["area_sqm"]  = area
        input_data["bedrooms"]  = bedrooms
        input_data["bathrooms"] = bathrooms

        loc_col  = f"location_{location}"
        type_col = f"property_type_{prop_type}"
        if loc_col  in input_data: input_data[loc_col]  = 1
        if type_col in input_data: input_data[type_col] = 1

        prediction = model.predict(pd.DataFrame([input_data]))[0]
        st.success(f"### Predicted fair price: R {prediction:,.0f}")

        # listing checker
        st.subheader("Check a real listing")
        asking = st.number_input(
            "Paste the asking price from a listing (R)",
            min_value=0, value=int(prediction)
        )
        diff = asking - prediction
        if diff > 0:
            st.warning(f"This listing is **R {diff:,.0f} ABOVE** fair value.")
        elif diff < 0:
            st.info(f"This listing is **R {abs(diff):,.0f} BELOW** fair value — possible good deal!")
        else:
            st.success("This listing is exactly at predicted fair value.")

# comparables chart
if df is not None:
    st.subheader(f"Comparable properties — {location}")
    loc_col = f"location_{location}"
    suburb_df = df[df[loc_col] == 1][["price", "area_sqm", "bedrooms"]] \
        if loc_col in df.columns else df[["price", "area_sqm", "bedrooms"]].head(20)

    if not suburb_df.empty:
        st.dataframe(suburb_df.head(20))
        st.bar_chart(suburb_df["price"].head(20))
    else:
        st.info("No comparables found for this suburb in the dataset.")

elif model is None:
    st.warning("Model not trained yet. Run `make run` first, then redeploy.")