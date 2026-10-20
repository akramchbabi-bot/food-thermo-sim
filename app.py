import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.subplots as plt
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

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
        "Potato (Raw)": {"water": 0.79, "protein": 0.02, "fat": 0.001, "carbs": 0.17, "ash": 0.01, "rho": 1070.0},
        "Custom Food": None
    }

    EQUIPMENT_PRESETS = {
        "Custom / Manual Input": {"h": 25, "temp": 150.0, "desc": "Manually adjust h and temperature."},
        "Still Air Oven (Proofing / Drying)": {"h": 10, "temp": 150.0, "desc": "Natural convection air (baking, slow roasting)."},
        "Continuous Industrial Deep Fryer": {"h": 450, "temp": 175.0, "desc": "Hot circulating frying oil (french fries, chips)."},
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
        water_pct = st.sidebar.slider("Adjust Water Content (%)", 5.0, 95.0, float(data["water"] * 100)) / 100.0
        rem_factor = (1.0 - water_pct) / (1.0 - data["water"]) if data["water"] < 1.0 else 0
        protein_val = data["protein"] * rem_factor
        fat_val = data["fat"] * rem_factor
        carbs_val = data["carbs"] * rem_factor
        ash_val = data["ash"] * rem_factor
        rho_val = data["rho"]
    else:
        water_pct = st.sidebar.slider("Water Content (%)", 0.0, 100.0, 75.0) / 100.0
        protein_val = 0.15
        fat_val = 0.05
        carbs_val = 0.04
        ash_val = 0.01
        rho_val = 1020.0

    enable_phase_change = st.sidebar.checkbox("Enable Water Evaporation / Phase Change", value=True)

    st.sidebar.header("2. Heat Transfer Mode & Boundary")
    mode = st.sidebar.selectbox("Select Surface Heat Mechanism:", ["Convection", "Direct Contact"])

    if mode == "Convection":
        eq_choice = st.sidebar.selectbox("Choose Equipment Preset:", list(EQUIPMENT_PRESETS.keys()))
        eq_data = EQUIPMENT_PRESETS[eq_choice]
        h_val = float(eq_data["h"])
        T_ambient = st.sidebar.number_input("Fluid / Medium Temp (°C)", value=float(eq_data["temp"]), step=5.0)
    else:
        h_contact = 2000.0
        T_surface_contact = st.sidebar.number_input("Cooking Surface / Pan Temp (°C)", value=150.0, step=5.0)

    thickness_cm = st.sidebar.slider("Food Thickness (cm)", 0.5, 10.0, 2.0)
    L = thickness_cm / 100.0
    T_init = st.sidebar.number_input("Initial Food Temp (°C)", value=5.0, step=1.0)
    cooking_time_min = st.sidebar.slider("Simulation Time (minutes)", 1, 60, 10)

    if st.button("🚀 Run Heat Transfer Simulation"):
        with st.spinner("⚡ Solving transient non-linear PDE..."):
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
                    
                q_surf_left = h_contact * (T_surface_contact - T[0]) if mode == "Direct Contact" else h_val * (T_ambient - T[0])
                q_surf_right = h_contact * (T_surface_contact - T[-1]) if mode == "Direct Contact" else h_val * (T_ambient - T[-1])
                
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
            st.session_state.k_init = k_init
            st.session_state.cp_base_init = cp_base_init
            st.session_state.alpha_init = alpha_init

    if "solution" in st.session_state:
        solution = st.session_state.solution
        x = st.session_state.x_array
        time_min = solution.t / 60.0
        
        st.divider()
        st.subheader("⏱️ Temperature Penetration")
        selected_time = st.slider("Inspect temperature at time:", 0.0, float(time_min[-1]), float(time_min[-1] / 2.0))
        time_idx = np.argmin(np.abs(time_min - selected_time))
        current_temps = solution.y[:, time_idx]

        fig1, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(time_min, solution.y[0, :], label="Surface")
        ax1.plot(time_min, solution.y[len(x)//2, :], label="Center")
        ax1.axvline(selected_time, color='black', linestyle='--')
        ax1.set_xlabel("Time (min)")
        ax1.set_ylabel("Temperature (°C)")
        ax1.legend()
        st.pyplot(fig1)


elif app_mode == "2. Evaporation & Concentration Thermodynamics":
    st.title("⚗️ Evaporation & Product Concentration Thermodynamics")

    st.sidebar.header("1. Fluid / Mixture Selection")
    fluid_type = st.sidebar.selectbox("Volatile Solvent:", ["Water", "Ethanol"])
    solvent_key = fluid_type

    m_volatile_0 = st.sidebar.number_input(f"Initial Mass of {solvent_key} (kg)", value=85.0, step=5.0)
    m_other_0 = st.sidebar.number_input("Initial Mass of Non-Volatiles (kg)", value=15.0, step=5.0)
    total_initial_mass = m_volatile_0 + m_other_0
    initial_conc_pct = (m_other_0 / total_initial_mass) * 100.0 if total_initial_mass > 0 else 0.0

    st.sidebar.header("2. Evaporation Pot & Heating Setup")
    vessel_type = st.sidebar.selectbox("Vessel Mode:", ["Open Pot (Constant Pressure)", "Closed Sealed Pot (Variable Pressure Buildup)"])
    pot_volume = st.sidebar.number_input("Pot Volume (Liters)", value=200.0) / 1000.0
    initial_pressure_atm = st.sidebar.number_input("Initial Pressure (atm)", value=1.0)

    enable_prv = False
    prv_threshold_atm = 2.5
    dynamic_boiling = False

    if vessel_type == "Closed Sealed Pot (Variable Pressure Buildup)":
        enable_prv = st.sidebar.checkbox("Enable Pressure Relief Valve (PRV)", value=True)
        if enable_prv:
            prv_threshold_atm = st.sidebar.number_input("PRV Setpoint (atm)", value=2.5)
        
        # NEW FEATURE: Toggle pressure-dependent boiling point
        dynamic_boiling = st.sidebar.checkbox("Pressure increases Boiling Point (Antoine Equation)", value=True, help="If checked, boiling temperature rises as pressure builds. If unchecked, it boils at a fixed temperature.")

    st.sidebar.markdown("**Energy Input**")
    heating_power_kw = st.sidebar.number_input("Thermal Heating Power Input (kW)", min_value=0.1, max_value=1000.0, value=15.0, step=1.0)
    evap_time_min = st.sidebar.slider("Evaporation Time (min)", 1, 120, 30)

    if st.button("🧪 Run Evaporation Model"):
        with st.spinner("Computing mass balance & energy..."):
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
            
            # FIXED SYNTAX ERROR HERE:
            n_vapor = (P_curr_kPa * 1000.0 * v_headspace) / (R_GAS * (25.0 + 273.15))

            cum_energy_kwh = 0.0

            for step in range(time_limit_steps):
                current_time_sec = step * dt
                
                # Apply Dynamic Boiling logic
                if vessel_type == "Closed Sealed Pot (Variable Pressure Buildup)" and dynamic_boiling:
                    T_b = (antoine_params["B"] / (antoine_params["A"] - np.log10(max(1.0, P_curr_kPa)))) - antoine_params["C"]
                else:
                    T_b = (antoine_params["B"] / (antoine_params["A"] - np.log10(max(1.0, initial_pressure_atm * 101.325)))) - antoine_params["C"]

                Q_gross_watts = heating_power_kw * 1000.0 # Strict kW input
                evap_rate = Q_gross_watts / L_v if m_curr > 0.001 else 0.0

                step_kwh = (Q_gross_watts / 1000.0) * (dt / 3600.0)
                cum_energy_kwh += step_kwh

                net_evap = evap_rate * dt
                m_curr = max(0.0, m_curr - net_evap)

                curr_total = m_curr + m_other_0
                curr_conc = (m_other_0 / curr_total) * 100.0 if curr_total > 0 else 100.0

                if vessel_type == "Open Pot (Constant Pressure)":
                    P_curr_kPa = initial_pressure_atm * 101.325
                else:
                    dn_vapor = net_evap / M_molar
                    n_vapor += dn_vapor
                    
                    # FIXED SYNTAX ERROR HERE:
                    P_curr_kPa = (n_vapor * R_GAS * (T_b + 273.15)) / (v_headspace * 1000.0)

                    if enable_prv and (P_curr_kPa > prv_threshold_atm * 101.325):
                        P_curr_kPa = prv_threshold_atm * 101.325
                        # FIXED SYNTAX ERROR HERE:
                        n_vapor = (P_curr_kPa * 1000.0 * v_headspace) / (R_GAS * (T_b + 273.15))

                t_list.append(current_time_sec / 60.0)
                m_solv_list.append(m_curr)
                conc_list.append(curr_conc)
                press_list.append(P_curr_kPa)
                boil_list.append(T_b)
                energy_list.append(cum_energy_kwh)

            st.session_state.m_solv = np.array(m_solv_list)
            st.session_state.concentration = np.array(conc_list)
            st.session_state.pressure_kPa = np.array(press_list)
            st.session_state.boiling_temp = np.array(boil_list)
            st.session_state.energy_kwh = np.array(energy_list)
            st.session_state.time_min = np.array(t_list)

    if "m_solv" in st.session_state:
        time_min = st.session_state.time_min
        concentration = st.session_state.concentration
        pressure_kPa = st.session_state.pressure_kPa
        boiling_temp = st.session_state.boiling_temp
        energy_kwh = st.session_state.energy_kwh

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Final Concentration", f"{concentration[-1]:.1f} %")
        c2.metric("Final Pressure", f"{pressure_kPa[-1]/101.325:.2f} atm")
        c3.metric("Boiling Temp", f"{boiling_temp[-1]:.1f} °C")
        c4.metric("Thermal Energy Used", f"{energy_kwh[-1]:.2f} kWh")

        fig2, ax2 = plt.subplots(figsize=(7, 4))
        ax2.plot(time_min, pressure_kPa / 101.325, color="crimson", label="Pressure (atm)")
        ax2.set_ylabel("Pressure (atm)")
        ax2_twin = ax2.twinx()
        ax2_twin.plot(time_min, boiling_temp, color="orange", linestyle="--", label="Boiling Temp (°C)")
        ax2_twin.set_ylabel("Boiling Temp (°C)")
        fig2.legend(loc="upper left")
        st.pyplot(fig2)
