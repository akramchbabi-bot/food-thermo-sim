import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Food & Industrial Process Simulator", layout="wide")

# Physics Constants
SIGMA = 5.670374419e-8  # Stefan-Boltzmann Constant (W/m²·K⁴)
L_V_WATER = 2.26e6      # Latent Heat of Vaporization of Water (J/kg)
L_V_ETHANOL = 8.41e5    # Latent Heat of Vaporization of Ethanol (J/kg)
R_GAS = 8.314           # Universal Gas Constant (J/mol·K)
T_BOIL_WATER = 100.0    # Water Boiling Point (°C)
T_BOIL_ETHANOL = 78.37  # Ethanol Boiling Point (°C)
DT_BOIL = 2.0           # Phase change transition interval half-width (°C)

# Antoine Equation Constants for Vapor Pressure (P in kPa, T in °C)
# log10(P_kPa) = A - B / (T + C)
ANTOINE = {
    "Water": {"A": 7.19621, "B": 1730.63, "C": 233.426, "MolarMass": 0.018015, "L_v": 2.26e6},
    "Ethanol": {"A": 7.33827, "B": 1642.89, "C": 230.300, "MolarMass": 0.04607, "L_v": 8.41e5}
}

# =====================================================================
# GLOBAL NAVIGATION & MODULE SELECTION
# =====================================================================
st.sidebar.title("🛠️ Navigation")
app_mode = st.sidebar.radio(
    "Select Simulator Module:",
    [
        "1. Transient Heat Transfer & Penetration",
        "2. Evaporation & Concentration Thermodynamics"
    ]
)

st.sidebar.divider()

# =====================================================================
# MODULE 1: TRANSIENT HEAT TRANSFER & PENETRATION SIMULATOR
# =====================================================================
if app_mode == "1. Transient Heat Transfer & Penetration":
    st.title("🍲 Food Physical Modeling & Heat Transfer Simulator")
    st.write("Simulate heat transfer (Conduction, Convection, Radiation, Phase Change) through food over time.")

    # Session State for slider visibilities
    if "show_h_slider" not in st.session_state:
        st.session_state.show_h_slider = False
    if "show_h_contact_slider" not in st.session_state:
        st.session_state.show_h_contact_slider = False

    FOOD_DATABASE = {
        "Beef Patty (Raw)": {"water": 0.73, "protein": 0.22, "fat": 0.04, "carbs": 0.0, "ash": 0.01, "rho": 1050.0},
        "Chicken Breast (Raw)": {"water": 0.75, "protein": 0.23, "fat": 0.012, "carbs": 0.0, "ash": 0.01, "rho": 1060.0},
        "Pork Chop (Raw)": {"water": 0.71, "protein": 0.20, "fat": 0.08, "carbs": 0.0, "ash": 0.01, "rho": 1040.0},
        "Lamb Meat": {"water": 0.73, "protein": 0.20, "fat": 0.06, "carbs": 0.0, "ash": 0.01, "rho": 1030.0},
        "Salmon Filet": {"water": 0.68, "protein": 0.20, "fat": 0.11, "carbs": 0.0, "ash": 0.01, "rho": 1040.0},
        "Cod Filet": {"water": 0.81, "protein": 0.18, "fat": 0.007, "carbs": 0.0, "ash": 0.012, "rho": 1050.0},
        "Shrimp": {"water": 0.78, "protein": 0.20, "fat": 0.01, "carbs": 0.0, "ash": 0.01, "rho": 1080.0},
        "Tomato": {"water": 0.94, "protein": 0.009, "fat": 0.002, "carbs": 0.039, "ash": 0.005, "rho": 1010.0},
        "Potato (Raw)": {"water": 0.79, "protein": 0.02, "fat": 0.001, "carbs": 0.17, "ash": 0.01, "rho": 1070.0},
        "Carrot": {"water": 0.88, "protein": 0.009, "fat": 0.002, "carbs": 0.10, "ash": 0.01, "rho": 1040.0},
        "Broccoli": {"water": 0.89, "protein": 0.028, "fat": 0.004, "carbs": 0.07, "ash": 0.008, "rho": 1030.0},
        "Apple": {"water": 0.85, "protein": 0.003, "fat": 0.002, "carbs": 0.14, "ash": 0.003, "rho": 840.0},
        "Banana": {"water": 0.75, "protein": 0.011, "fat": 0.003, "carbs": 0.23, "ash": 0.008, "rho": 970.0},
        "Strawberry": {"water": 0.91, "protein": 0.007, "fat": 0.003, "carbs": 0.077, "ash": 0.004, "rho": 1020.0},
        "Whole Milk": {"water": 0.88, "protein": 0.032, "fat": 0.032, "carbs": 0.048, "ash": 0.007, "rho": 1030.0},
        "Cheddar Cheese": {"water": 0.37, "protein": 0.25, "fat": 0.33, "carbs": 0.013, "ash": 0.037, "rho": 1080.0},
        "Whole Egg (Raw)": {"water": 0.76, "protein": 0.13, "fat": 0.10, "carbs": 0.01, "ash": 0.01, "rho": 1035.0},
        "Bread Dough": {"water": 0.40, "protein": 0.09, "fat": 0.03, "carbs": 0.46, "ash": 0.02, "rho": 600.0},
        "Cooked Rice": {"water": 0.68, "protein": 0.027, "fat": 0.003, "carbs": 0.28, "ash": 0.005, "rho": 1120.0},
        "Pancake Batter": {"water": 0.55, "protein": 0.06, "fat": 0.05, "carbs": 0.32, "ash": 0.02, "rho": 980.0},
        "Custom Food": None
    }

    EQUIPMENT_PRESETS = {
        "Custom / Manual Input": {"h": 25, "temp": 150.0, "desc": "Manually adjust h and temperature."},
        "Still Air Oven (Proofing / Drying)": {"h": 10, "temp": 150.0, "desc": "Natural convection air (baking, slow roasting)."},
        "Forced-Air Spiral / Tunnel Oven": {"h": 35, "temp": 180.0, "desc": "Circulating hot air (commercial baking / roasting)."},
        "High-Velocity Air Impingement Oven": {"h": 120, "temp": 200.0, "desc": "Hot air jets directed at surface (pizza, cookies, nuggets)."},
        "Continuous Industrial Deep Fryer": {"h": 450, "temp": 175.0, "desc": "Hot circulating frying oil (french fries, chips)."},
        "Water Blancher / Pasteurizer": {"h": 600, "temp": 95.0, "desc": "Agitated boiling/hot water bath (vegetable blanching)."},
        "Industrial Steam Tunnel / Retort": {"h": 150
    
