import pandas as pd
import streamlit as st
from PIL import Image
from src.clean_data_and_predict import Data

data = Data()
raw_data = data.raw_data
clean_data = data.clean()
prediction = data.predict(clean_data)

if prediction < .25:
    ordinal_prediction = 'Low'
elif prediction > .25 and prediction < .70:
    ordinal_prediction = 'Moderate'
elif prediction > .70:
    ordinal_prediction = 'High'

event_name = raw_data['name'][0]
org_name = raw_data['org_name'][0]

st.write(f"Raw data")
st.write(raw_data)
st.write(f"Organization name: {org_name}")

st.write(f"Fraud risk: {ordinal_prediction}")
st.write(f"Likelihood of fraud: {prediction}")

img = Image.open('imgs/Final_Prediction_Pipeline.png')
st.image(img)