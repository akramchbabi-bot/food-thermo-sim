import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food Thermodynamics Simulator", layout="wide")

st.title("🍲 Food Physical Modeling & Heat Conduction Simulator")
st.write("Simulate temperature distribution across food thickness over time.")

# =====================================================================
# 1. EXPANDED DATABASE OF FOOD COMPOSITIONS
# (Values derived from USDA FoodData Central & ASHRAE handbooks)
# Fractions: water, protein, fat, carbs, ash | Density (rho) in kg/m³
# =====================================================================
FOOD_DATABASE = {
    # --- MEATS & POULTRY ---
    "Beef Patty (Raw)": {"water": 0.73, "protein": 0.22, "fat": 0.04, "carbs": 0.0, "ash": 0.01, "rho": 1050.0},
    "Chicken Breast (Raw)": {"water": 0.75, "protein": 0.23, "fat": 0.012, "carbs": 0.0, "ash": 0.01, "rho": 1060.0},
    "Pork Chop (Raw)": {"water": 0.71, "protein": 0.20, "fat": 0.08, "carbs": 0.0, "ash": 0.01, "rho": 1040.0},
    "Lamb Meat": {"water": 0.73, "protein": 0.20, "fat": 0.06, "carbs": 0.0, "ash": 0.01, "rho": 1030.0},
    
    # --- SEAFOOD ---
    "Salmon Filet": {"water": 0.68, "protein": 0.20, "fat": 0.11, "carbs": 0.0, "ash": 0.01, "rho": 1040.0},
    "Cod Filet": {"water": 0.81, "protein": 0.18, "fat": 0.007, "carbs": 0.0, "ash": 0.012, "rho": 1050.0},
    "Shrimp": {"water": 0.78, "protein": 0.20, "fat": 0.01, "carbs": 0.0, "ash": 0.01, "rho": 1080.0},
    
    # --- VEGETABLES & LEGUMES ---
    "Potato (Raw)": {"water": 0.79, "protein": 0.02, "fat": 0.001, "carbs": 0.17, "ash": 0.01, "rho": 1070.0},
    "Carrot": {"water": 0.88, "protein": 0.009, "fat": 0.002, "carbs": 0.10, "ash": 0.01, "rho": 1040.0},
    "Broccoli": {"water": 0.89, "protein": 0.028, "fat": 0.004, "carbs": 0.07, "ash": 0.008, "rho": 1030.0},
    "Tomato": {"water": 0.94, "protein": 0.009, "fat": 0.002, "carbs": 0.039, "ash": 0.005, "rho": 1010.0},
    
    # --- FRUITS ---
    "Apple": {"water": 0.85, "protein": 0.003, "fat": 0.002, "carbs": 0.14, "ash": 0.003, "rho": 840.0},
    "Banana": {"water": 0.75, "protein": 0.011, "fat": 0.003, "carbs": 0.23, "ash": 0.008, "rho": 970.0},
    "Strawberry": {"water": 0.91, "protein": 0.007, "fat": 0.003, "carbs": 0.077, "ash": 0.004, "rho": 1020.0},
    
    # --- DAIRY & EGGS ---
    "Whole Milk": {"water": 0.88, "protein": 0.032, "fat": 0.032, "carbs": 0.048, "ash": 0.007, "rho": 1030.0},
    "Cheddar Cheese": {"water": 0.37, "protein": 0.25, "fat": 0.33, "carbs": 0.013, "ash": 0.037, "rho": 1080.0},
    "Whole Egg (Raw)": {"water": 0.76, "protein": 0.13, "fat": 0.10, "carbs": 0.01, "ash": 0.01, "rho": 1035.0},
    
    # --- GRAINS & BAKERY ---
    "Bread Dough": {"water": 0.40, "protein": 0.09, "fat": 0.03, "carbs": 0.46, "ash": 0.02, "rho": 600.0},
    "Cooked Rice": {"water": 0.68, "protein": 0.027, "fat": 0.003, "carbs": 0.28, "ash": 0.005, "rho": 1120.0},
    "Pancake Batter": {"water": 0.55, "protein": 0.06, "fat": 0.05, "carbs": 0.32, "ash": 0.02, "rho": 980.0},
    
    # --- CUSTOM FOOD ---
    "Custom Food": None
}

# =====================================================================
# 2. CHOI & OKOS PROPERTY ESTIMATOR
# =====================================================================
def get_food_properties(X_w, X_p, X_f, X_c, X_a, rho, T=20.0):
    # Choi & Okos specific heat equations (J/kg·K)
    cp_w = 4176.2 - 0.0908 * T + 0.0054 * (T**2)
    cp_p = 2008.2 + 1.2089 * T - 0.0013 * (T**2)
    cp_f = 1984.2 + 1.4733 * T - 0.0048 * (T**2)
    cp_c = 1548.8 + 1.9625 * T - 0.0059 * (T**2)
    cp_a = 1092.6 + 1.8896 * T - 0.0036 * (T**2)
    
    cp = (X_w * cp_w + X_p * cp_p + X_f * cp_f + X_c * cp_c + X_a * cp_a)
    
    # Choi & Okos thermal conductivity equations (W/m·K)
    k_w = 0.57109 + 0.001762 * T - 0.0000067 * (T**2)
    k_p = 0.17881 + 0.001198 * T - 0.0000027 * (T**2)
    k_f = 0.18071 - 0.000276 * T - 0.0000001 * (T**2)
    k_c = 0.20141 + 0.001387 * T - 0.0000043 * (T**2)
    k_a = 0.32962 + 0.001401 * T - 0.0000029 * (T**2)
    
    k = (X_w * k_w + X_p * k_p + X_f * k_f + X_c * k_c + X_a * k_a)
    
    return k, cp, rho

# =====================================================================
# 3. SIDEBAR CONTROLS & INPUTS
# =====================================================================
st.sidebar.header("1. Select or Customize Food")
food_choice = st.sidebar.selectbox("Choose a food item:", list(FOOD_DATABASE.keys()))

if food_choice != "Custom Food":
    data = FOOD_DATABASE[food_choice]
    st.sidebar.success(f"Loaded database values for {food_choice}")
    
    # Allow user to tweak water percentage
    water_pct = st.sidebar.slider("Adjust Water Content (%)", 5.0, 95.0, float(data["water"] * 100)) / 100.0
    
    # Normalize remaining fraction proportionately
    rem_factor = (1.0 - water_pct) / (1.0 - data["water"]) if data["water"] < 1.0 else 0
    k_val, cp_val, rho_val = get_food_properties(
        water_pct, 
        data["protein"] * rem_factor, 
        data["fat"] * rem_factor, 
        data["carbs"] * rem_factor, 
        data["ash"] * rem_factor, 
        data["rho"]
    )
else:
    st.sidebar.subheader("Custom Food Composition")
    
    # Composition Sliders (%)
    water_pct = st.sidebar.slider("Water Content (%)", 0.0, 100.0, 75.0) / 100.0
    protein_pct = st.sidebar.slider("Protein Content (%)", 0.0, 100.0, 15.0) / 100.0
    fat_pct = st.sidebar.slider("Fat Content (%)", 0.0, 100.0, 5.0) / 100.0
    carbs_pct = st.sidebar.slider("Carbohydrates Content (%)", 0.0, 100.0, 4.0) / 100.0
    ash_pct = st.sidebar.slider("Ash / Minerals Content (%)", 0.0, 10.0, 1.0) / 100.0
    
    rho_val = st.sidebar.number_input("Density ρ (kg/m³)", value=1020.0, step=10.0)
    
    # Check if total fraction adds up to 100%
    total_pct = (water_pct + protein_pct + fat_pct + carbs_pct + ash_pct) * 100
    if round(total_pct, 1) != 100.0:
        st.sidebar.warning(f"⚠️ Total composition is **{total_pct:.1f}%**. For best results, components should total 100%.")
    
    # Compute thermal properties automatically via Choi & Okos
    k_val, cp_val, _ = get_food_properties(
        water_pct, protein_pct, fat_pct, carbs_pct, ash_pct, rho_val
    )

st.sidebar.header("2. Process Parameters")
thickness_cm = st.sidebar.slider("Food Thickness (cm)", 0.5, 10.0, 2.0)
L = thickness_cm / 100.0  # meters

T_top = st.sidebar.number_input("Top/Boundary Temperature (°C)", value=150.0, step=5.0)
T_init = st.sidebar.number_input("Initial Food Temperature (°C)", value=5.0, step=1.0)
cooking_time_min = st.sidebar.slider("Simulation Time (minutes)", 1, 60, 10)

# =====================================================================
# 4. SOLVER & DISPLAY
# =====================================================================
if st.button("🚀 Generate Heating Curves"):
    alpha = k_val / (rho_val * cp_val)
    t_final = cooking_time_min * 60.0
    
    N = 25
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    
    def heat_equation(t, T):
        dTdt = np.zeros_like(T)
        dTdt[0] = 0.0
        dTdt[-1] = 0.0
        for i in range(1, N - 1):
            dTdt[i] = alpha * (T[i + 1] - 2 * T[i] + T[i - 1]) / (dx ** 2)
        return dTdt

    T0 = np.full(N, T_init)
    T0[0] = T_top
    T0[-1] = T_top

    t_eval = np.linspace(0, t_final, 200)
    solution = solve_ivp(heat_equation, [0, t_final], T0, t_eval=t_eval, method='RK45')

    # Display Calculated Thermal Properties
    col1, col2, col3 = st.columns(3)
    col1.metric("Thermal Conductivity (k)", f"{k_val:.3f} W/m·K")
    col2.metric("Specific Heat (Cp)", f"{cp_val:.1f} J/kg·K")
    col3.metric("Thermal Diffusivity (α)", f"{alpha:.2e} m²/s")

    # Plot results
    fig, ax = plt.subplots(figsize=(10, 5))
    time_min = solution.t / 60.0

    # Specific depths to plot
    depth_indices = [0, N//8, N//4, 3*N//8, N//2]
    for idx in depth_indices:
        pos_cm = x[idx] * 100
        label_text = f"Surface ({pos_cm:.2f} cm)" if idx == 0 else (
            f"Center ({pos_cm:.2f} cm)" if idx == N//2 else f"Depth {pos_cm:.2f} cm"
        )
        ax.plot(time_min, solution.y[idx, :], label=label_text, linewidth=2)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Temperature Evolution in {food_choice} ({thickness_cm} cm thick)")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    st.pyplot(fig)
    
