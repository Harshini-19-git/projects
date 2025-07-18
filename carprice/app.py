import pandas as pd
import numpy as np
import pickle as pk
import streamlit as st

# -----------------------------
# Load model
# -----------------------------
model = pk.load(open('model.pkl', 'rb'))

# Page Configuration
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="centered")

# Header
st.markdown(
    """
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        text-align:center;
        color:#FF4B4B;
    }
    .sub-title {
        font-size:20px;
        text-align:center;
        color:#555;
    }
    </style>
    <p class="main-title">🚗 Car Price Prediction ML Model</p>
    <p class="sub-title">Enter car details below to predict its price</p>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(r"C:\Users\2201c\Downloads\Cardetails.csv")

# Extract Brand Name
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

df['name'] = df['name'].apply(get_brand_name)

# -----------------------------
# Input Section with Columns
# -----------------------------
st.sidebar.markdown("### ℹ️ Car Input Details")

name = st.sidebar.selectbox('**Select the Car Brand**', df['name'].unique())
year = st.sidebar.slider('**Car Manufacture Year**', 1994, 2025, step=1)
km_driven = st.sidebar.slider('**Kilometers Driven**', 1, 200000, step=1000)
fuel = st.sidebar.selectbox('**Fuel Type**', df['fuel'].unique())
seller_type = st.sidebar.selectbox('**Seller Type**', df['seller_type'].unique())
transmission = st.sidebar.selectbox('**Transmission**', df['transmission'].unique())
owner = st.sidebar.selectbox('**Owner Type**', df['owner'].unique())
mileage = st.sidebar.slider('**Mileage (kmpl)**', 10, 40, step=1)
engine = st.sidebar.slider('**Engine CC**', 700, 5000, step=100)
max_power = st.sidebar.slider('**Max Power (bhp)**', 0, 200, step=1)
seats = st.sidebar.slider('**No. of Seats**', 5, 10, step=1)

# -----------------------------
# Predict Button
# -----------------------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'>🔍 Prediction Result</h3>", unsafe_allow_html=True)

if st.button("🚀 Predict Price", use_container_width=True):
    example_input = pd.DataFrame([{
        'name': name,
        'year': year,
        'km_driven': km_driven,
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats,
    }])

    # Encoding
    example_input['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner',
                                    'Test Drive Car'], [1, 2, 3, 4, 5], inplace=True)
    example_input['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    example_input['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    example_input['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault', 'Mahindra',
                                    'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz', 'Mitsubishi', 'Audi',
                                    'Volkswagen', 'BMW', 'Nissan', 'Lexus', 'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo',
                                    'Kia', 'Fiat', 'Force', 'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                   list(range(1, 32)), inplace=True)
    example_input['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)

    # Predict
    car_price = model.predict(example_input)[0]

    # Result Display
    st.success(f"💰 Estimated Car showroom Price: **₹ {car_price:,.2f}**")
    st.write("### ✅ Entered Car Details:")
    st.dataframe(example_input)

# Footer
st.markdown("<br><p style='text-align:center; color:gray;'>Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
