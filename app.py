import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

st.set_page_config(page_title="Food & Industrial Process Simulator", layout="wide")

# Physics Constants
SIGMA = 5.670374419e-8  
L_V_WATER = 2.26e6      
L_V_ETHANOL = 8.41e5    
R_GAS = 8.314           
T_BOIL_WATER = 100.0    
T_BOIL_ETHANOL = 78.37  
DT_BOIL = 2.0           

ANTOINE = {
    "Water": {"A": 7.19621, "B": 1730.63, "C": 233.426, "MolarMass": 0.018015, "L_v": 2.26e6},
    "Ethanol": {"A": 7.33827, "B": 1642.89, "C": 230.300, "MolarMass": 0.04607, "L_v": 8.41e5}
}

st.sidebar.title("🛠️ Navigation")
app_mode = st.sidebar.radio(
    "Select Simulator Module:",
    [
        "1. Transient Heat Transfer & Penetration",
        "2. Evaporation & Concentration Thermodynamics"
    ]
)

st.sidebar.divider()

if app_mode == "1. Transient Heat Transfer & Penetration":
    st.title("🍲 Food Physical Modeling & Heat Transfer Simulator")
    st.write("Simulate heat transfer (Conduction, Convection, Radiation, Phase Change) through food over time.")

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
        "Industrial Steam Tunnel / Retort": {"h": 1500, "temp": 121.0, "desc": "Condensing pressurized steam (sterilization, frankfurters)."}
    }

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
        k_a = 0.32962 + 0.001401 * T - 0.00000029 * (T**2)
        k = (X_w * k_w + X_p * k_p + X_f * k_f + X_c * k_c + X_a * k_a)
        return k, cp_base, rho

    def get_effective_cp(T_celsius, base_cp, X_water, enable_phase_change=True):
        if not enable_phase_change or X_water <= 0.0:
            return base_cp
        gaussian = (1.0 / (DT_BOIL * np.sqrt(np.pi))) * np.exp(-((T_celsius - T_BOIL_WATER) / DT_BOIL)**2)
        return base_cp + (X_water * L_V_WATER * gaussian)

    def generate_pixel_art(T_profile, x_array, grid_size=33):
        grid = np.full((grid_size, grid_size), np.nan)
        center = (grid_size - 1) / 2.0
        radius_px = grid_size / 2.0 - 1.0
        half_N = len(x_array) // 2

        for r in range(grid_size):
            for c in range(grid_size):
                dist = np.sqrt((r - center)**2 + (c - center)**2)
                if dist <= radius_px:
                    norm_dist = dist / radius_px
                    depth_cm = (1.0 - norm_dist) * (x_array[-1] * 100 / 2.0)
                    grid[r, c] = np.interp(depth_cm, x_array[:half_N+1] * 100, T_profile[:half_N+1])
        return grid

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
        if st.sidebar.button("⚙️ Tweak Equipment h Value"):
            st.session_state.show_h_slider = not st.session_state.show_h_slider
        if st.session_state.show_h_slider or eq_choice == "Custom / Manual Input":
            h_val = st.sidebar.slider("Heat Transfer Coeff. h (W/m²·K)", 5, 2000, int(eq_data["h"]))
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
        if st.sidebar.button("⚙️ Tweak Contact Conductance"):
            st.session_state.show_h_contact_slider = not st.session_state.show_h_contact_slider
        if st.session_state.show_h_contact_slider:
            h_contact = st.sidebar.slider("Contact Conductance (W/m²·K)", 500, 5000, 2000)
        else:
            h_contact = 2000.0
            st.sidebar.info("Using default $h_{\\text{contact}} = 2000\\text{ W/m}^2\\cdot\\text{K}$")

    st.sidebar.header("3. Process Parameters")
    thickness_cm = st.sidebar.slider("Food Thickness (cm)", 0.5, 10.0, 2.0)
    L = thickness_cm / 100.0
    T_init = st.sidebar.number_input("Initial Food Temp (°C)", value=5.0, step=1.0)
    cooking_time_min = st.sidebar.slider("Simulation Time (minutes)", 1, 60, 10)

    if st.button("🚀 Run Heat Transfer Simulation"):
        with st.spinner("⚡ Solving transient non-linear PDE... Please wait..."):
            k_init, cp_base_init, _ = get_food_properties(water_pct, protein_val, fat_val, carbs_val, ash_val, rho_val, T=T_init)
            cp_eff_init = get_effective_cp(T_init, cp_base_init, water_pct, enable_phase_change)
            alpha_init = k_init / (rho_val * cp_eff_init)
            t_final = cooking_time_min * 60.0
            
            N = 25
            x = np.linspace(0, L, N)
            dx = x[1] - x[0]
            
            def heat_equation(t, T):
                dTdt = np.zeros_like(T)
                for i in range(1, N - 1):
                    k_i, cp_base_i, _ = get_food_properties(water_pct, protein_val, fat_val, carbs_val, ash_val, rho_val, T=T[i])
                    cp_eff_i = get_effective_cp(T[i], cp_base_i, water_pct, enable_phase_change)
                    alpha_i = k_i / (rho_val * cp_eff_i)
                    dTdt[i] = alpha_i * (T[i + 1] - 2 * T[i] + T[i - 1]) / (dx ** 2)
                    
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

            st.session_state.solution = solution
            st.session_state.x_array = x
            st.session_state.T_init = T_init
            st.session_state.k_init = k_init
            st.session_state.cp_base_init = cp_base_init
            st.session_state.rho_val = rho_val
            st.session_state.alpha_init = alpha_init

    if "solution" in st.session_state:
        solution = st.session_state.solution
        x = st.session_state.x_array
        N = len(x)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Thermal Conductivity (k)", f"{st.session_state.k_init:.3f} W/m·K")
        col2.metric("Sensible Specific Heat (Cp)", f"{st.session_state.cp_base_init:.1f} J/kg·K")
        col3.metric("Density (ρ)", f"{rho_val:.0f} kg/m³")
        col4.metric("Thermal Diffusivity (α)", f"{st.session_state.alpha_init:.2e} m²/s")

        st.divider()
        st.subheader("⏱️ Temperature Penetration & Pixel Art Animator")

        time_min = solution.t / 60.0
        selected_time = st.slider(
            "Move slider to inspect temperature penetration over time:", 
            min_value=0.0, 
            max_value=float(time_min[-1]), 
            value=float(time_min[-1] / 2.0), 
            step=float(time_min[1] - time_min[0])
        )

        time_idx = np.argmin(np.abs(time_min - selected_time))
        current_temps = solution.y[:, time_idx]

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Selected Time", f"{selected_time:.2f} min")
        m_col2.metric("Surface Temp", f"{current_temps[0]:.1f} °C")
        m_col3.metric("Core Center Temp", f"{current_temps[N//2]:.1f} °C")

        plot_col1, plot_col2 = st.columns([1.2, 1])

        with plot_col1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            depth_indices = [0, N//8, N//4, 3*N//8, N//2]
            for idx in depth_indices:
                pos_cm = x[idx] * 100
                label_text = f"Surface ({pos_cm:.2f} cm)" if idx == 0 else (
                    f"Center ({pos_cm:.2f} cm)" if idx == N//2 else f"Depth {pos_cm:.2f} cm"
                )
                ax1.plot(time_min, solution.y[idx, :], label=label_text, linewidth=2)

            ax1.axvline(selected_time, color='black', linestyle='--', linewidth=2, label="Current Timestamp")
            ax1.axhline(100, color='red', linestyle=':', alpha=0.7, label="Boiling Point (100°C)")

            max_temp_val = max(110.0, float(np.max(solution.y)) * 1.05)
            ax1.set_xlim(left=0, right=max(time_min[-1], 0.1))
            ax1.set_ylim(bottom=0, top=max_temp_val)

            ax1.set_xlabel("Time (minutes)", fontsize=10)
            ax1.set_ylabel("Temperature (°C)", fontsize=10)
            ax1.set_title(f"Thermal Profile Curves", fontsize=11)
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend(loc="lower right", fontsize=8)
            st.pyplot(fig1)

        with plot_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            pixel_grid = generate_pixel_art(current_temps, x, grid_size=33)

            cmap = mcolors.LinearSegmentedColormap.from_list(
                "food_temp", ["#1e88e5", "#4caf50", "#ffeb3b", "#ff9800", "#f44336", "#b71c1c"]
            )
            cmap.set_bad(color="#0e1117")

            max_temp_bound = max(150.0, float(np.max(solution.y)) * 1.05)
            im = ax2.imshow(pixel_grid, cmap=cmap, vmin=0, vmax=max_temp_bound, interpolation='nearest')
            ax2.set_title(f"🔴 Pixel Art Cross-Section at t = {selected_time:.2f} min", fontsize=11)
            ax2.axis('off')
            
            cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
            cbar.set_label("Temperature (°C)", fontsize=9)
            st.pyplot(fig2)

elif app_mode == "2. Evaporation & Concentration Thermodynamics":
    st.title("⚗️ Evaporation & Product Concentration Thermodynamics")
    st.write("Model liquid mass reduction, vapor accumulation, boiling point changes, and thermal energy usage.")

    st.sidebar.header("1. Fluid / Mixture Selection")
    fluid_type = st.sidebar.selectbox(
        "Volatile Solvent Component:",
        ["Water + Dry Solids (Juice / Jam / Sauce)", "Ethanol + Water Mixture (Distillation / Evaporation)"]
    )
    solvent_key = "Water" if "Water +" in fluid_type else "Ethanol"

    input_mode = st.sidebar.radio("Input Composition By:", ["Mass Percentage (%)", "Direct Mass Inputs (kg)"])

    if input_mode == "Mass Percentage (%)":
        total_initial_mass = st.sidebar.number_input("Total Batch Initial Mass (kg)", value=100.0, step=10.0)
        volatile_pct = st.sidebar.slider(f"Initial {solvent_key} Content (%)", 10.0, 99.0, 85.0) / 100.0
        m_volatile_0 = total_initial_mass * volatile_pct
        m_other_0 = total_initial_mass * (1.0 - volatile_pct)
    else:
        m_volatile_0 = st.sidebar.number_input(f"Initial Mass of {solvent_key} (kg)", value=85.0, step=5.0)
        m_other_0 = st.sidebar.number_input("Initial Mass of Non-Volatiles / Other (kg)", value=15.0, step=5.0)
        total_initial_mass = m_volatile_0 + m_other_0
        volatile_pct = m_volatile_0 / total_initial_mass if total_initial_mass > 0 else 0

    initial_conc_pct = (m_other_0 / total_initial_mass) * 100.0 if total_initial_mass > 0 else 0.0
    st.sidebar.info(f"💡 Initial Concentration: **{initial_conc_pct:.1f}% Non-Volatiles**")

    st.sidebar.header("2. Calculation Goal")
    calc_goal = st.sidebar.radio(
        "Simulation Goal:",
        ["Find Time to Reach Target Concentration", "Simulate Fixed Time Duration"]
    )

    if calc_goal == "Find Time to Reach Target Concentration":
        min_target = min(98.0, float(np.ceil(initial_conc_pct + 1.0)))
        target_conc_pct = st.sidebar.slider(
            "Target Product Concentration (%)",
            min_value=min_target,
            max_value=99.0,
            value=min(95.0, float(np.ceil(initial_conc_pct + 20.0)))
        )
        max_timeout_min = st.sidebar.number_input("Maximum Safety Timeout (minutes)", value=180, step=15)
    else:
        evap_time_min = st.sidebar.slider("Evaporation Simulation Time (minutes)", 1, 120, 30)

    st.sidebar.header("3. Evaporation Pot & Safety Valve Setup")
    vessel_type = st.sidebar.selectbox("Vessel Operating Mode:", ["Open Pot (Constant Atmospheric Pressure)", "Closed Sealed Pot (Variable Pressure Buildup)"])
    pot_volume = st.sidebar.number_input("Total Pot Vessel Volume (Liters)", value=200.0, step=10.0) / 1000.0
    initial_pressure_atm = st.sidebar.number_input("Initial Vessel Pressure (atm)", value=1.0, step=0.1)

    enable_prv = False
    prv_threshold_atm = 2.5
    if vessel_type == "Closed Sealed Pot (Variable Pressure Buildup)":
        enable_prv = st.sidebar.checkbox("Enable Pressure Relief Valve (PRV)", value=True)
        if enable_prv:
            prv_threshold_atm = st.sidebar.number_input("PRV Setpoint Threshold (atm)", value=2.5, min_value=1.1, max_value=10.0, step=0.1)

    heating_mode = st.sidebar.radio(
        "Heating Energy Input Model:",
        [
            "Pressure-Coupled Driving Force (Q = U·A·ΔT)",
            "Fixed Heating Power Input (kW)"
        ]
    )

    if heating_mode == "Pressure-Coupled Driving Force (Q = U·A·ΔT)":
        T_heating = st.sidebar.number_input("Heating Jacket / Surface Temp (°C)", value=140.0, step=5.0)
        U_heating = st.sidebar.slider("Jacket Heat Transfer Coeff. U (W/m²·K)", 100, 3000, 1200)
        A_heating = st.sidebar.slider("Heating Surface Area A (m²)", 0.1, 5.0, 1.2)
        st.sidebar.caption("ℹ️ As pressure rises, boiling point T_b increases. ΔT = (T_heating - T_b) collapses, slowing boiling speed.")
    else:
        heating_power_kw = st.sidebar.slider("Thermal Heating Power Input (kW)", 1.0, 100.0, 15.0)

    st.sidebar.header("4. Condenser Settings (Optional)")
    enable_condenser = st.sidebar.checkbox("Enable Overhead Condenser Surface", value=True)
    if enable_condenser:
        T_condenser = st.sidebar.slider("Condenser Wall Temp (°C)", 5.0, 60.0, 20.0)
        condenser_area = st.sidebar.slider("Condenser Surface Area (m²)", 0.1, 5.0, 1.0)
        h_condenser = st.sidebar.slider("Condensation Heat Transfer Coeff. (W/m²·K)", 100, 3000, 1000)

    if st.button("🧪 Run Evaporation & Concentration Model"):
        with st.spinner("⚡ Computing mass balance, energy usage & PRV thermodynamics..."):
            if calc_goal == "Find Time to Reach Target Concentration":
                m_solv_target = m_other_0 * ((100.0 / target_conc_pct) - 1.0)
                max_sec = max_timeout_min * 60.0
                dt = 1.0
                time_limit_steps = int(max_sec / dt)
            else:
                m_solv_target = -1.0
                max_sec = evap_time_min * 60.0
                dt = max_sec / 300.0
                time_limit_steps = 300

            t_list, m_solv_list, conc_list, press_list, boil_list, energy_list = [], [], [], [], [], []

            m_curr = m_volatile_0
            density_liq = 1000.0 if solvent_key == "Water" else 789.0
            v_liq_0 = total_initial_mass / density_liq
            v_headspace = max(0.01, pot_volume - v_liq_0)

            antoine_params = ANTOINE[solvent_key]
            L_v = antoine_params["L_v"]
            M_molar = antoine_params["MolarMass"]

            P_curr_kPa = initial_pressure_atm * 101.325
            n_vapor = (P_curr_kPa * 1000.0 * v_headspace) / (R_GAS * (25.0 + 273.15))

            target_reached = False
            final_time_sec = 0.0
            cum_energy_kwh = 0.0

            for step in range(time_limit_steps):
                current_time_sec = step * dt
                
                T_b = (antoine_params["B"] / (antoine_params["A"] - np.log10(max(1.0, P_curr_kPa)))) - antoine_params["C"]

                if heating_mode == "Pressure-Coupled Driving Force (Q = U·A·ΔT)":
                    delta_T = max(0.0, T_heating - T_b)
                    Q_gross_watts = U_heating * A_heating * delta_T
                else:
                    Q_gross_watts = heating_power_kw * 1000.0

                evap_rate = Q_gross_watts / L_v if m_curr > 0.001 else 0.0

                step_kwh = (Q_gross_watts / 1000.0) * (dt / 3600.0)
                cum_energy_kwh += step_kwh

                cond_rate = 0.0
                if enable_condenser and evap_rate > 0:
                    q_cond = h_condenser * condenser_area * max(0.0, T_b - T_condenser)
                    cond_rate = min(evap_rate, q_cond / L_v)

                net_evap = (evap_rate - cond_rate) * dt
                m_curr = max(0.0, m_curr - net_evap)

                curr_total = m_curr + m_other_0
                curr_conc = (m_other_0 / curr_total) * 100.0 if curr_total > 0 else 100.0

                if vessel_type == "Open Pot (Constant Atmospheric Pressure)":
                    P_curr_kPa = initial_pressure_atm * 101.325
                else:
                    dn_vapor = ((evap_rate - cond_rate) * dt) / M_molar
                    n_vapor += dn_vapor
                    P_curr_kPa = (n_vapor * R_GAS * (T_b + 273.15)) / (v_headspace * 1000.0)

                    if enable_prv and (P_curr_kPa > prv_threshold_atm * 101.325):
                        P_curr_kPa = prv_threshold_atm * 101.325
                        n_vapor = (P_curr_kPa * 1000.0 * v_headspace) / (R_GAS * (T_b + 273.15))

                t_list.append(current_time_sec / 60.0)
                m_solv_list.append(m_curr)
                conc_list.append(curr_conc)
                press_list.append(P_curr_kPa)
                boil_list.append(T_b)
                energy_list.append(cum_energy_kwh)

                if calc_goal == "Find Time to Reach Target Concentration" and m_curr <= m_solv_target:
                    target_reached = True
                    final_time_sec = current_time_sec
                    break

            st.session_state.m_solv = np.array(m_solv_list)
            st.session_state.concentration = np.array(conc_list)
            st.session_state.pressure_kPa = np.array(press_list)
            st.session_state.boiling_temp = np.array(boil_list)
            st.session_state.energy_kwh = np.array(energy_list)
            st.session_state.time_min = np.array(t_list)
            st.session_state.solvent_key = solvent_key
            st.session_state.calc_goal = calc_goal
            st.session_state.target_reached = target_reached
            st.session_state.final_time_min = final_time_sec / 60.0
            st.session_state.target_conc_pct = target_conc_pct if calc_goal == "Find Time to Reach Target Concentration" else 0.0
            st.session_state.enable_prv = enable_prv
            st.session_state.prv_threshold_atm = prv_threshold_atm
            st.session_state.vessel_type = vessel_type
            st.session_state.m_volatile_0 = m_volatile_0

    if ("m_solv" in st.session_state and 
        st.session_state.get("m_solv") is not None and 
        len(st.session_state.get("m_solv")) > 0 and
        "energy_kwh" in st.session_state and 
        st.session_state.get("energy_kwh") is not None and 
        len(st.session_state.get("energy_kwh")) > 0):
        
        time_min = st.session_state.get("time_min")
        m_solv = st.session_state.get("m_solv")
        concentration = st.session_state.get("concentration")
        pressure_kPa = st.session_state.get("pressure_kPa")
        boiling_temp = st.session_state.get("boiling_temp")
        energy_kwh = st.session_state.get("energy_kwh")
        solv_key = st.session_state.get("solvent_key", "Water")
        current_calc_goal = st.session_state.get("calc_goal", "Simulate Fixed Time Duration")

        m_vol_0 = st.session_state.get("m_volatile_0", m_solv[0])
        evaporated_kg = m_vol_0 - m_solv[-1]
        
        if evaporated_kg > 0:
            specific_energy = energy_kwh[-1] / evaporated_kg
        else:
            specific_energy = 0.0

        if current_calc_goal == "Find Time to Reach Target Concentration":
            if st.session_state.get("target_reached", False):
                st.success(f"🎯 **Target Concentration of {st.session_state.get('target_conc_pct', 0.0):.1f}% reached in {st.session_state.get('final_time_min', 0.0):.2f} minutes!**")
            else:
                st.error(f"⚠️ Target concentration was not reached within the safety limit ({time_min[-1]:.0f} min). Try increasing heating jacket temperature or surface area.")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Final Concentration", f"{concentration[-1]:.1f} %")
        c2.metric(f"Remaining {solv_key}", f"{m_solv[-1]:.1f} kg")
        c3.metric("Final Pressure", f"{pressure_kPa[-1]/101.325:.2f} atm")
        c4.metric("Boiling Temp", f"{boiling_temp[-1]:.1f} °C")
        c5.metric("Thermal Energy Used", f"{energy_kwh[-1]:.2f} kWh", help=f"Specific Energy: {specific_energy:.3f} kWh/kg evaporated")

        st.caption(f"⚡ **Energy Efficiency Metric:** **{specific_energy:.3f} kWh** per kg of {solv_key} evaporated ({evaporated_kg:.1f} kg evaporated total).")

        st.divider()
        res_col1, res_col2 = st.columns(2)

        with res_col1:
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            ax1.plot(time_min, concentration, color="darkgreen", linewidth=2.5, label="Solids Concentration (%)")
            if current_calc_goal == "Find Time to Reach Target Concentration":
                ax1.axhline(st.session_state.get("target_conc_pct", 0.0), color="red", linestyle="--", label=f"Target ({st.session_state.get('target_conc_pct', 0.0):.1f}%)")
            ax1.set_xlabel("Time (minutes)")
            ax1.set_ylabel("Non-Volatile Concentration (%)", color="darkgreen")
            ax1.grid(True, linestyle="--", alpha=0.5)

            max_conc_val = max(100.0, float(np.max(concentration)) * 1.05)
            ax1.set_xlim(left=0, right=max(time_min[-1], 0.1))
            ax1.set_ylim(bottom=0, top=max_conc_val)

            ax1_twin = ax1.twinx()
            ax1_twin.plot(time_min, m_solv, color="blue", linestyle="--", linewidth=2, label=f"{solv_key} Mass (kg)")
            ax1_twin.set_ylabel(f"Mass of {solv_key} (kg)", color="blue")
            
            max_mass_val = max(10.0, float(np.max(m_solv)) * 1.05)
            ax1_twin.set_ylim(bottom=0, top=max_mass_val)

            ax1.set_title("Product Concentration vs. Solvent Evaporation", fontsize=11)
            ax1.legend(loc="upper left")
            st.pyplot(fig1)

        with res_col2:
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            press_atm = pressure_kPa / 101.325
            ax2.plot(time_min, press_atm, color="crimson", linewidth=2.5, label="Vessel Pressure (atm)")

            if st.session_state.get("vessel_type") == "Closed Sealed Pot (Variable Pressure Buildup)" and st.session_state.get("enable_prv", False):
                ax2.axhline(st.session_state.get("prv_threshold_atm", 2.5), color="darkred", linestyle=":", linewidth=2, label=f"PRV Limit ({st.session_state.get('prv_threshold_atm', 2.5):.1f} atm)")

            ax2.set_xlabel("Time (minutes)")
            ax2.set_ylabel("Pressure (atm)", color="crimson")
            ax2.grid(True, linestyle="--", alpha=0.5)

            max_press_val = max(1.5, float(np.max(press_atm)) * 1.05)
            ax2.set_xlim(left=0, right=max(time_min[-1], 0.1))
            ax2.set_ylim(bottom=0, top=max_press_val)

            ax2_twin = ax2.twinx()
            ax2_twin.plot(time_min, energy_kwh, color="purple", linestyle="-.", linewidth=2, label="Cumulative Heat (kWh)")
            ax2_twin.set_ylabel("Thermal Energy Consumed (kWh)", color="purple")
            
            max_energy_val = max(1.0, float(np.max(energy_kwh)) * 1.05)
            ax2_twin.set_ylim(bottom=0, top=max_energy_val)

            ax2.set_title("Vessel Pressure & Cumulative Energy (kWh)", fontsize=11)
            ax2.legend(loc="upper left")
            st.pyplot(fig2)
            
