import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

st.set_page_config(page_title="Food Thermodynamics Simulator", layout="wide")

st.title("🍲 Food Physical Modeling & Heat Transfer Simulator")
st.write("Simulate heat transfer (Conduction, Convection, Radiation, Phase Change) through food over time.")

# Physics Constants
SIGMA = 5.670374419e-8  # Stefan-Boltzmann Constant (W/m²·K⁴)
L_V = 2.26e6            # Latent Heat of Vaporization of Water (J/kg)
T_BOIL = 100.0          # Water Boiling Point (°C)
DT_BOIL = 2.0           # Phase change transition interval half-width (°C)

# Initialize Session State for h slider visibility
if "show_h_slider" not in st.session_state:
    st.session_state.show_h_slider = False

# =====================================================================
# 1. EXPANDED DATABASE OF FOOD COMPOSITIONS
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
    "Tomato": {"water": 0.94, "protein": 0.009, "fat": 0.002, "carbs": 0.039, "ash": 0.005, "rho": 1010.0},
    "Potato (Raw)": {"water": 0.79, "protein": 0.02, "fat": 0.001, "carbs": 0.17, "ash": 0.01, "rho": 1070.0},
    "Carrot": {"water": 0.88, "protein": 0.009, "fat": 0.002, "carbs": 0.10, "ash": 0.01, "rho": 1040.0},
    "Broccoli": {"water": 0.89, "protein": 0.028, "fat": 0.004, "carbs": 0.07, "ash": 0.008, "rho": 1030.0},
    
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

EQUIPMENT_PRESETS = {
    "Custom / Manual Input": {"h": 25, "temp": 150.0, "desc": "Manually adjust h and temperature."},
    "Still Air Oven (Proofing / Drying)": {"h": 10, "temp": 150.0, "desc": "Natural convection air (baking, slow roasting)."},
    "Forced-Air Spiral / Tunnel Oven": {"h": 35, "temp": 180.0, "desc": "Circulating hot air (commercial baking / roasting)."},
    "High-Velocity Air Impingement Oven": {"h": 120, "temp": 200.0, "desc": "Hot air jets directed at surface (pizza, cookies, nuggets)."},
    "Continuous Industrial Deep Fryer": {"h": 450, "temp": 175.0, "desc": "Hot circulating frying oil (french fries, chips)."},
    "Water Blancher / Pasteurizer": {"h": 600, "temp": 95.0, "desc": "Agitated boiling/hot water bath (vegetable blanching)."},
    "Industrial Steam Tunnel / Retort": {"h": 1500, "temp": 121.0, "desc": "Condensing pressurized steam (sterilization, frankfurters)."}
}

# =====================================================================
# 2. CHOI & OKOS & PHASE CHANGE CALCULATOR
# =====================================================================
def get_food_properties(X_w, X_p, X_f, X_c, X_a, rho, T=20.0):
    cp_w = 4176.2 - 0.0908 * T + 0.0054 * (T**2)
    cp_p = 2008.2 + 1.2089 * T - 0.0013 * (T**2)
    cp_f = 1984.2 + 1.4733 * T - 0.0048 * (T**2)
    cp_c = 1548.8 + 1.9625 * T - 0.0059 * (T**2)
    cp_a = 1092.6 + 1.8896 * T - 0.0036 * (T**2)
    
    cp_base = (X_w * cp_w + X_p * cp_p + X_f * cp_f + X_c * cp_c + X_a * cp_a)
    
    k_w = 0.57109 + 0.001762 * T - 0.0000067 * (T**2)
    k_p = 0.17881 + 0.001198 * T - 0.0000027 * (T**2)
    k_f = 0.18071 - 0.000276 * T - 0.0000001 * (T**2)
    k_c = 0.20141 + 0.001387 * T - 0.0000043 * (T**2)
    k_a = 0.32962 + 0.001401 * T - 0.0000029 * (T**2)
    
    k = (X_w * k_w + X_p * k_p + X_f * k_f + X_c * k_c + X_a * k_a)
    
    return k, cp_base, rho

def get_effective_cp(T_celsius, base_cp, X_water, enable_phase_change=True):
    if not enable_phase_change or X_water <= 0.0:
        return base_cp
    gaussian = (1.0 / (DT_BOIL * np.sqrt(np.pi))) * np.exp(-((T_celsius - T_BOIL) / DT_BOIL)**2)
    return base_cp + (X_water * L_V * gaussian)

# =====================================================================
# 3. SIDEBAR CONTROLS & INPUTS
# =====================================================================
st.sidebar.header("1. Select or Customize Food")
food_choice = st.sidebar.selectbox("Choose a food item:", list(FOOD_DATABASE.keys()))

if food_choice != "Custom Food":
    data = FOOD_DATABASE[food_choice]
    st.sidebar.success(f"Loaded database values for {food_choice}")
    
    water_pct = st.sidebar.slider("Adjust Water Content (%)", 5.0, 95.0, float(data["water"] * 100)) / 100.0
    rem_factor = (1.0 - water_pct) / (1.0 - data["water"]) if data["water"] < 1.0 else 0
    protein_val = data["protein"] * rem_factor
    fat_val = data["fat"] * rem_factor
    carbs_val = data["carbs"] * rem_factor
    ash_val = data["ash"] * rem_factor
    rho_val = data["rho"]
else:
    st.sidebar.subheader("Custom Food Composition")
    water_pct = st.sidebar.slider("Water Content (%)", 0.0, 100.0, 75.0) / 100.0
    protein_val = st.sidebar.slider("Protein Content (%)", 0.0, 100.0, 15.0) / 100.0
    fat_val = st.sidebar.slider("Fat Content (%)", 0.0, 100.0, 5.0) / 100.0
    carbs_val = st.sidebar.slider("Carbohydrates Content (%)", 0.0, 100.0, 4.0) / 100.0
    ash_val = st.sidebar.slider("Ash / Minerals Content (%)", 0.0, 10.0, 1.0) / 100.0
    rho_val = st.sidebar.number_input("Density ρ (kg/m³)", value=1020.0, step=10.0)

enable_phase_change = st.sidebar.checkbox("Enable Water Evaporation / Phase Change (100°C Plateau)", value=True)

st.sidebar.header("2. Heat Transfer Mode & Boundary")
mode = st.sidebar.selectbox(
    "Select Surface Heat Mechanism:",
    [
        "Direct Contact (Griddle / Pan Conductance)",
        "Convection (Air / Water / Oil / Steam)",
        "Radiation (Infrared / Broiler)",
        "Combined Convection + Radiation"
    ]
)

if "Convection" in mode:
    st.sidebar.markdown("**Industrial Equipment Preset**")
    eq_choice = st.sidebar.selectbox("Choose Equipment Preset:", list(EQUIPMENT_PRESETS.keys()))
    eq_data = EQUIPMENT_PRESETS[eq_choice]
    
    if eq_choice != "Custom / Manual Input":
        st.sidebar.caption(f"ℹ️ {eq_data['desc']}")
        
    # Button to show/hide the h slider
    if st.sidebar.button("⚙️ Tweak Equipment h Value"):
        st.session_state.show_h_slider = not st.session_state.show_h_slider

    # Show slider only if button pressed or if Custom preset chosen
    if st.session_state.show_h_slider or eq_choice == "Custom / Manual Input":
        h_val = st.sidebar.slider(
            "Heat Transfer Coeff. h (W/m²·K)", 
            min_value=5, max_value=2000, value=int(eq_data["h"]),
            help="Still Air: 5-15 | Forced Air: 20-50 | Air Impingement: 50-180 | Frying Oil: 250-600 | Steam: 1000+"
        )
    else:
        h_val = float(eq_data["h"])
        st.sidebar.info(f"Using default $h = {int(h_val)}\\text{{ W/m}}^2\\cdot\\text{{K}}$ for {eq_choice}")

    T_ambient = st.sidebar.number_input("Fluid / Medium Temp (°C)", value=float(eq_data["temp"]), step=5.0)

if "Radiation" in mode:
    st.sidebar.markdown("**Radiative Parameters**")
    emissivity = st.sidebar.slider("Surface Emissivity (ε)", 0.1, 1.0, 0.90)
    T_rad = st.sidebar.number_input("Radiant Element Temp (°C)", value=250.0, step=10.0)

if "Direct Contact" in mode:
    st.sidebar.markdown("**Contact Parameters**")
    T_surface_contact = st.sidebar.number_input("Cooking Surface / Pan Temp (°C)", value=150.0, step=5.0)
    h_contact = st.sidebar.slider("Contact Conductance (W/m²·K)", 500, 5000, 2000, help="Heat transfer rate between metal plate and food surface.")

st.sidebar.header("3. Process Parameters")
thickness_cm = st.sidebar.slider("Food Thickness (cm)", 0.5, 10.0, 2.0)
L = thickness_cm / 100.0
T_init = st.sidebar.number_input("Initial Food Temp (°C)", value=5.0, step=1.0)
cooking_time_min = st.sidebar.slider("Simulation Time (minutes)", 1, 60, 10)

# =====================================================================
# 4. SOLVER & PDE DISCRETIZATION
# =====================================================================
if st.button("🚀 Run Heat Transfer Simulation"):
    k_init, cp_base_init, _ = get_food_properties(water_pct, protein_val, fat_val, carbs_val, ash_val, rho_val, T=T_init)
    cp_eff_init = get_effective_cp(T_init, cp_base_init, water_pct, enable_phase_change)
    alpha_init = k_init / (rho_val * cp_eff_init)
    t_final = cooking_time_min * 60.0
    
    N = 25
    x = np.linspace(0, L, N)
    dx = x[1] - x[0]
    
    def heat_equation(t, T):
        dTdt = np.zeros_like(T)
        
        # Interior Nodes
        for i in range(1, N - 1):
            k_i, cp_base_i, _ = get_food_properties(water_pct, protein_val, fat_val, carbs_val, ash_val, rho_val, T=T[i])
            cp_eff_i = get_effective_cp(T[i], cp_base_i, water_pct, enable_phase_change)
            alpha_i = k_i / (rho_val * cp_eff_i)
            
            dTdt[i] = alpha_i * (T[i + 1] - 2 * T[i] + T[i - 1]) / (dx ** 2)
            
        # Dynamic Surface Boundary Nodes
        q_surf_left = 0.0
        q_surf_right = 0.0
        
        if "Direct Contact" in mode:
            q_surf_left += h_contact * (T_surface_contact - T[0])
            q_surf_right += h_contact * (T_surface_contact - T[-1])
            
        if "Convection" in mode:
            q_surf_left += h_val * (T_ambient - T[0])
            q_surf_right += h_val * (T_ambient - T[-1])
            
        if "Radiation" in mode:
            T0_K, TN_K, Trad_K = T[0] + 273.15, T[-1] + 273.15, T_rad + 273.15
            q_surf_left += emissivity * SIGMA * (Trad_K**4 - T0_K**4)
            q_surf_right += emissivity * SIGMA * (Trad_K**4 - TN_K**4)
        
        k_0, cp_b0, _ = get_food_properties(water_pct, protein_val, fat_val, carbs_val, ash_val, rho_val, T=T[0])
        cp_e0 = get_effective_cp(T[0], cp_b0, water_pct, enable_phase_change)
        
        k_N, cp_bN, _ = get_food_properties(water_pct, protein_val, fat_val, carbs_val, ash_val, rho_val, T=T[-1])
        cp_eN = get_effective_cp(T[-1], cp_bN, water_pct, enable_phase_change)
        
        dTdt[0] = (2.0 / (rho_val * cp_e0 * dx)) * (q_surf_left + k_0 * (T[1] - T[0]) / dx)
        dTdt[-1] = (2.0 / (rho_val * cp_eN * dx)) * (q_surf_right + k_N * (T[-2] - T[-1]) / dx)
            
        return dTdt

    T0 = np.full(N, T_init)
    t_eval = np.linspace(0, t_final, 200)
    solution = solve_ivp(heat_equation, [0, t_final], T0, t_eval=t_eval, method='Radau')

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Thermal Conductivity (k)", f"{k_init:.3f} W/m·K")
    col2.metric("Sensible Specific Heat (Cp)", f"{cp_base_init:.1f} J/kg·K")
    col3.metric("Density (ρ)", f"{rho_val:.0f} kg/m³")
    col4.metric("Thermal Diffusivity (α)", f"{alpha_init:.2e} m²/s")

    fig, ax = plt.subplots(figsize=(10, 5))
    time_min = solution.t / 60.0

    depth_indices = [0, N//8, N//4, 3*N//8, N//2]
    for idx in depth_indices:
        pos_cm = x[idx] * 100
        label_text = f"Surface ({pos_cm:.2f} cm)" if idx == 0 else (
            f"Center ({pos_cm:.2f} cm)" if idx == N//2 else f"Depth {pos_cm:.2f} cm"
        )
        ax.plot(time_min, solution.y[idx, :], label=label_text, linewidth=2)

    ax.axhline(100, color='red', linestyle=':', alpha=0.7, label="Water Boiling Point (100°C)")
    ax.set_xlabel("Time (minutes)", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_title(f"Temperature Evolution in {food_choice} (Phase Change: {'Enabled' if enable_phase_change else 'Disabled'})", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")

    st.pyplot(fig)
