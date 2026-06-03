# Johannesburg Property Intelligence Engine

A data science tool that helps first-time home buyers in Gauteng 
figure out whether a property listing is fairly priced.

**Live app:** https://property-intel-4ecysnm3fhmy4dkr3ndioz.streamlit.app/

---

## The Problem

Property is the largest purchase most South Africans ever make, 
and the market is opaque. This tool predicts a fair market price 
given a property's suburb, size, and key features.

---

## How to Run Locally

```bash
# install dependencies
make install

# train the model
make run

# run tests
make test

# launch the app locally
streamlit run src/app/streamlit_app.py
```

---

## Project Structure

- `src/property_intel/sources/` — data source classes (Kaggle, Crime, Gautrain)
- `src/property_intel/pipeline.py` — combines all data sources
- `src/property_intel/features.py` — feature engineering
- `src/property_intel/modeling.py` — model training and evaluation
- `src/app/streamlit_app.py` — Streamlit web interface
- `tests/` — 10 pytest unit tests
- `config.yaml` — all tunable settings

---

## Data Sources

- **Property listings:** Kaggle — prospernkomo/roodepoort-property-listings
- **Crime index:** Manually compiled from SAPS annual crime statistics reports for the Roodepoort policing area
- **Gautrain distances:** Computed from GPS coordinates using the Haversine formula

---

## Model Results

| Metric | Value |
|--------|-------|
| RMSE | R1,332,834 |
| MAE | R665,760 |
| R² | 0.598 |

The model explains approximately 60% of price variation across 3,856 
cleaned property listings. Linear Regression was used as required.

---

## The Five Analysis Questions

### Q1 — What percentage of records were dropped and why?

10,205 raw rows were loaded. 3,856 remained after cleaning — 
meaning 62% were dropped. The main reasons were:
- Duplicate listings (the dataset contained many repeated entries)
- Price listed as "Not specified"
- Area listed as "Not specified"
- Prices outside the R100,000–R50,000,000 range

The remaining sample still covers the full range of Roodepoort 
suburbs and price bands relevant to a typical first-time buyer, 
so it remains representative for the project's purpose.

### Q2 — Which features correlate strongest with price?

Strong correlations: area_sqm, bedrooms, bathrooms — as expected. 
Surprising non-correlations: gautrain_km showed weak correlation 
because all Roodepoort suburbs are roughly equally far from 
Gautrain stations, so there was little variation to learn from.

### Q3 — Where does the model fail worst?

The model fails most on industrial and commercial properties 
(Cosmo Business Park, Industria North) which have extreme prices 
far outside the residential range. A purely residential dataset 
would significantly improve accuracy.

### Q4 — Where should a first-time buyer with R1.5M look?

Based on the data, suburbs with the most listings under R1.5M are 
Honeydew, Wilgeheuwel, and Northgate. These suburbs also have 
relatively low crime index scores (3–5) making them suitable 
options for a first-time buyer on that budget.

### Q5 — Ethical concern (weighted heavily)

South Africa's property market was shaped by decades of apartheid 
spatial planning. The Group Areas Act forcibly segregated suburbs 
by race, and historically white suburbs received better 
infrastructure, schools, and services — all of which drove 
property prices higher in those areas.

A model trained on current prices learns and replicates that 
pattern. When the model predicts a township property is "fairly 
priced" at R400,000 and a similar property in a historically 
white suburb at R3,000,000, it is not making a neutral 
observation — it is encoding decades of legislated inequality 
and calling it objective market truth.

Specific risks:
- A bank using this model for mortgage decisions could 
  systematically undervalue township properties, making it 
  harder for Black South Africans to build wealth through 
  homeownership
- The crime_index feature may encode racial bias — policing 
  is historically more intense in certain areas regardless 
  of actual safety levels
- Small sample sizes for some suburbs make predictions 
  unreliable for exactly those areas most likely to be 
  historically underserved

What I would do differently with more time: include 
infrastructure quality variables (school rankings, clinic 
access), test the model for disparate impact across demographic 
areas, and add a clear disclaimer in the app that the model 
predicts market price — not intrinsic value — and that these 
are not the same thing.

---

## Lessons Learned

- Separating code from configuration (config.yaml) makes 
  projects much easier to maintain
- The class hierarchy (BaseDataSource → children) enforces 
  consistency across all data sources
- Logging is far more useful than print() for debugging 
  a multi-file project
- Train/test split is non-negotiable — evaluating on training 
  data produces meaningless metrics
