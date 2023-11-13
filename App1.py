import streamlit as st
import numpy as np
import joblib
import pydeck as pdk
from sklearn.preprocessing import StandardScaler
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from geopy.distance import geodesic

# Load the pre-trained SVR model
loaded_model = joblib.load(open("svr_model.sav", "rb"))

# Load the fitted StandardScaler
loaded_scaler = joblib.load(open("scaled_data.sav", "rb"))

geolocator = Nominatim(user_agent="fuel_consumption_app")

def input_converter(inp):
    vcl = ['Two-seater','Minicompact','Compact','Subcompact','Mid-size','Full-size','SUV: Small','SUV: Standard','Minivan','Station wagon: Small','Station wagon: Mid-size','Pickup truck: Small','Special purpose vehicle','Pickup truck: Standard']
    trans = ['AV','AM','M','AS','A']
    fuel = ["D","E","X","Z"]
    lst = []

    for i in range(6):
        if isinstance(inp[i], str):
            if inp[i] in vcl:
                lst.append(vcl.index(inp[i]))
            elif inp[i] in trans:
                lst.append(trans.index(inp[i]))
            elif inp[i] in fuel:
                if fuel.index(inp[i]) == 0:
                    lst.extend([1, 0, 0, 0])
                    break
                elif fuel.index(inp[i]) == 1:
                    lst.extend([0, 1, 0, 0])
                    break
                elif fuel.index(inp[i]) == 2:
                    lst.extend([0, 0, 1, 0])
                    break
                elif fuel.index(inp[i]) == 3:
                    lst.extend([0, 0, 0, 1])
        else:
            lst.append(inp[i])

    arr = np.asarray(lst)
    arr = arr.reshape(1, -1)

    with st.spinner("Predicting..."):
        with st.expander("Advanced Options"):
            st.markdown("You can customize the advanced options here.")
        with st.spinner("Processing..."):
            arr_scaled = loaded_scaler.transform(arr)

    prediction = loaded_model.predict(arr_scaled)
    return round(prediction[0], 2)

def calculate_distance(point1, point2):
    return geodesic(point1, point2).kilometers  # Calculate distance in kilometers

def get_coordinates(location):
    try:
        location_info = geolocator.geocode(location)
        if location_info:
            return location_info.latitude, location_info.longitude
        else:
            return None
    except GeocoderTimedOut:
        return None

def main():
    # Set page configuration
    st.set_page_config(
        page_title="Fuel Consumption Prediction",
        page_icon="✈️",
        layout="wide",
    )

    # Custom styles
    st.markdown(
        """
        <style>
            body {
                background-image: url('https://www.satisgps.com/wp-content/uploads/2019/12/Paliwo_artykul04-1.png.');
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }
            .main {
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .sidebar {
                padding: 20px;
                background-color: rgba(255, 255, 255, 0.8);
                border-radius: 10px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
            .footer {
                padding: 10px;
                background-color: rgba(0, 0, 0, 0.6);
                color: white;
                text-align: center;
                border-radius: 5px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Fuel Consumption Prediction")

    st.sidebar.header("Input Values")

    # Use number input instead of slider
    vehicle_class = st.sidebar.selectbox("Vehicle Class", ['Two-seater','Minicompact','Compact','Subcompact','Mid-size','Full-size','SUV: Small','SUV: Standard','Minivan','Station wagon: Small','Station wagon: Mid-size','Pickup truck: Small','Special purpose vehicle','Pickup truck: Standard'])
    engine_size = st.sidebar.number_input("Engine Size", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    cylinders = st.sidebar.number_input("Cylinders", min_value=0, max_value=16, step=1, value=4)
    CO2_rating = st.sidebar.number_input("CO2 Rating", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    transmission = st.sidebar.selectbox("Transmission", ['AV', 'AM', 'M', 'AS', 'A'])
    fuel_type = st.sidebar.selectbox("Fuel Type", ["D", "E", "X", "Z"])

    user_input = [vehicle_class, engine_size, cylinders, transmission, CO2_rating, fuel_type]

    # Add a map to the app using st.pydeck_chart
    deck = pdk.Deck(
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=[
                    {"position": [37.7749, -122.4194], "color": [255, 0, 0], "radius": 100}
                ],
                radius_scale=10,
            )
        ]
    )

    st.pydeck_chart(deck)

    if st.button("Predict"):
        result = input_converter(user_input)
        st.success(f"The predicted fuel consumption is: {result} L/100km")

        # Get user input for two locations
        st.sidebar.subheader("Calculate Distance:")
        location1_name = st.sidebar.text_input("Enter Location 1", "San Francisco, CA")
        location2_name = st.sidebar.text_input("Enter Location 2", "Los Angeles, CA")

        # Convert location names to coordinates
        location1_coords = get_coordinates(location1_name)
        location2_coords = get_coordinates(location2_name)

        if location1_coords and location2_coords:
            # Calculate distance between two locations
            distance = calculate_distance(location1_coords, location2_coords)
            st.info(f"The distance between the two locations is: {distance:.2f} kilometers")

            # Calculate total fuel consumption based on distance
            total_fuel_used = result * distance
            st.info(f"The estimated total fuel used is: {total_fuel_used:.2f} liters")
        else:
            st.error("Error retrieving coordinates for one or both locations.")

    # Footer
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p class='footer'>© 2023 Fuel Consumption App</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
