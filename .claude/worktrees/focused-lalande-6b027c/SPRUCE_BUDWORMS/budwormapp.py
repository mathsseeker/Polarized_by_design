import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# Import model functions from local module
from budworm import spruce_budworm, evolve_spruce_budworm, spruce_budworm_no_carrying_capacity


st.set_page_config(page_title="Spruce Budworm Interactive Model", layout="wide")
st.title("🐛 Spruce Budworm Population Model")

# --- Session state initialization ---
if "t" not in st.session_state:
    st.session_state.t = np.array([0.0])

if "x" not in st.session_state:
    st.session_state.x = np.array([1.0])


# ============ SIDEBAR: Parameters ============
with st.sidebar:
    st.header("⚙️ Model Parameters")
    r = st.slider("Growth rate (r)", min_value=0.0, max_value=2.0, value=0.5, step=0.01,
                  help="Intrinsic growth rate of the population")
    k = st.slider("Carrying capacity (k)", min_value=1.0, max_value=50.0, value=10.0, step=1.0,
                  help="Maximum sustainable population")
    
    st.divider()
    st.header("🎯 Initial Conditions")
    x0 = st.number_input("Initial population (x₀)", min_value=0.0, value=1.0, step=0.1,
                         help="Starting population size")
    
    st.divider()
    st.header("📊 Simulation Settings")
    evolve_duration = st.number_input("Evolution time (Δt)", min_value=0.1, value=10.0, step=1.0,
                                      help="How long to simulate forward")
    points = st.slider("Plot resolution", min_value=100, max_value=1000, value=500, step=50,
                      help="Number of points for phase portrait")
    
    st.divider()
    if st.button("🔄 Reset Simulation", use_container_width=True):
        st.session_state.t = np.array([0.0])
        st.session_state.x = np.array([x0])
        st.success("Simulation reset!")


# ============ MAIN CONTENT ============
col1, col2 = st.columns(2)

# -------- LEFT: Phase Portrait --------
with col1:
    st.subheader("📈 Phase Portrait: dx/dt vs Population")
    
    # Compute dx/dt over [0, k]
    x_values = np.linspace(0, k, points)
    dxdt_values = np.array([spruce_budworm(0.0, xv, r, k) for xv in x_values])
    
    # Find equilibria
    equilibria = []
    for guess in np.linspace(0, k, 20):
        try:
            root = fsolve(lambda xv: spruce_budworm(0.0, xv, r, k), guess)[0]
            if 0.0 <= root <= k and abs(spruce_budworm(0.0, root, r, k)) < 1e-6:
                if not any(abs(root - eq) < 1e-3 for eq in equilibria):
                    equilibria.append(root)
        except Exception:
            pass
    equilibria = sorted(equilibria)
    
    # Classify stability
    stable_eq, unstable_eq = [], []
    eps = 1e-6
    for eq in equilibria:
        left = spruce_budworm(0.0, eq - eps, r, k)
        right = spruce_budworm(0.0, eq + eps, r, k)
        if left > 0 and right < 0:
            stable_eq.append(eq)
        else:
            unstable_eq.append(eq)
    
    # Plot phase portrait
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(x_values, dxdt_values, "k-", linewidth=2.5, label="dx/dt")
    ax1.axhline(0, color="gray", linestyle="--", alpha=0.6, linewidth=1.5, label="Null rate (y=0)")
    ax1.axvline(st.session_state.x[-1], color="green", linestyle="--", linewidth=2, alpha=0.8,
               label=f"Current x = {st.session_state.x[-1]:.2f}")
    
    for i, eq in enumerate(stable_eq):
        ax1.plot(eq, 0, "bo", markersize=10, label="Stable equilibrium" if i == 0 else "")
    for i, eq in enumerate(unstable_eq):
        ax1.plot(eq, 0, "ro", markersize=10, label="Unstable equilibrium" if i == 0 else "")
    
    ax1.set_xlim(0, k)
    ypad = (dxdt_values.max() - dxdt_values.min()) * 0.1 if dxdt_values.size else 1
    ax1.set_ylim(dxdt_values.min() - ypad, dxdt_values.max() + ypad)
    ax1.set_xlabel("Population (x)", fontsize=11)
    ax1.set_ylabel("Rate of change (dx/dt)", fontsize=11)
    ax1.set_title(f"Phase Portrait (r={r}, k={k})", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.legend(fontsize=9, loc="best")
    st.pyplot(fig1)
    
    if equilibria:
        st.info(f"**Equilibria:** {', '.join(f'{eq:.3f}' for eq in equilibria)}")


# -------- RIGHT: Time Series --------
with col2:
    st.subheader("⏱️ Population Dynamics Over Time")
    
    # Button to evolve simulation
    if st.button("▶️ Evolve Forward", use_container_width=True):
        t_old, x_old = st.session_state.t, st.session_state.x
        t_upd, x_upd = evolve_spruce_budworm(t_old, x_old, r=r, k=k, t_eval=float(evolve_duration))
        x_upd = np.clip(x_upd, 0.0, None)
        st.session_state.t, st.session_state.x = t_upd, x_upd
        st.success(f"✅ Evolved from t={t_old[-1]:.2f} → t={t_upd[-1]:.2f} | Final x={x_upd[-1]:.3f}")
    
    # Plot time series
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(st.session_state.t, st.session_state.x, "g-", linewidth=2.5, label="Population x(t)")
    ax2.set_ylim(0, max(st.session_state.x) * 1.1 if st.session_state.x.size else 1)
    ax2.grid(True, alpha=0.3, linestyle="--")
    ax2.set_xlabel("Time (t)", fontsize=11)
    ax2.set_ylabel("Population (x)", fontsize=11)
    ax2.set_title("Population Trajectory", fontsize=12, fontweight="bold")
    ax2.legend(loc="best", fontsize=10)
    st.pyplot(fig2)
    
    st.metric("Current Time", f"{st.session_state.t[-1]:.2f}")
    st.metric("Current Population", f"{st.session_state.x[-1]:.3f}")


# ============ MODEL COMPARISON SECTION ============
st.divider()
st.header("🔬 Model Comparison: With vs Without Carrying Capacity Term")
st.markdown("""
Compare the **original model** `r*x*(1-x/k) - x²/(1+x²)` with the **modified model** `r*x - x²/(1+x²)` 
that removes the carrying capacity limiting term `-x/k`.
""")

comp_col1, comp_col2 = st.columns(2)

with comp_col1:
    st.subheader("📊 Phase Portrait Comparison")
    
    # Compute dx/dt for both models
    x_comp = np.linspace(0, k, points)
    dxdt_original = np.array([spruce_budworm(0.0, xv, r, k) for xv in x_comp])
    dxdt_no_cap = np.array([spruce_budworm_no_carrying_capacity(0.0, xv, r, k) for xv in x_comp])
    
    # Create comparison plot
    fig3, ax3 = plt.subplots(figsize=(7, 5))
    ax3.plot(x_comp, dxdt_original, "b-", linewidth=2.5, label="Original (with -x/k)")
    ax3.plot(x_comp, dxdt_no_cap, "r-", linewidth=2.5, label="Modified (no -x/k)")
    ax3.axhline(0, color="gray", linestyle="--", alpha=0.5, linewidth=1.5)
    ax3.axvline(st.session_state.x[-1], color="green", linestyle=":", linewidth=2, alpha=0.7,
               label=f"Current x = {st.session_state.x[-1]:.2f}")
    ax3.set_xlim(0, k)
    ax3.set_xlabel("Population (x)", fontsize=11)
    ax3.set_ylabel("Rate of change (dx/dt)", fontsize=11)
    ax3.set_title("Phase Portrait Comparison", fontsize=12, fontweight="bold")
    ax3.grid(True, alpha=0.3, linestyle="--")
    ax3.legend(fontsize=9, loc="best")
    st.pyplot(fig3)

with comp_col2:
    st.subheader("⏱️ Evolution Comparison")
    
    # Allow user to set comparison evolution parameters
    comp_t_max = st.number_input("Comparison time span", min_value=1.0, value=50.0, step=5.0,
                                 help="How long to simulate both models")
    comp_x0 = st.number_input("Comparison initial population", min_value=0.1, value=1.0, step=0.1,
                              help="Starting population for comparison")
    
    if st.button("🚀 Run Comparison", use_container_width=True):
        from scipy.integrate import solve_ivp
        
        t_span = (0, comp_t_max)
        t_eval = np.linspace(0, comp_t_max, 500)
        
        # Solve both models
        sol_orig = solve_ivp(spruce_budworm, t_span, [comp_x0], t_eval=t_eval, args=(r, k), method="RK45")
        sol_no_cap = solve_ivp(spruce_budworm_no_carrying_capacity, t_span, [comp_x0], 
                               t_eval=t_eval, args=(r, k), method="RK45")
        
        # Plot comparison
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.plot(sol_orig.t, sol_orig.y[0], "b-", linewidth=2.5, label="Original (with -x/k)")
        ax4.plot(sol_no_cap.t, sol_no_cap.y[0], "r-", linewidth=2.5, label="Modified (no -x/k)")
        ax4.set_ylim(0, None)
        ax4.grid(True, alpha=0.3, linestyle="--")
        ax4.set_xlabel("Time (t)", fontsize=11)
        ax4.set_ylabel("Population (x)", fontsize=11)
        ax4.set_title(f"Evolution Comparison (x₀={comp_x0}, r={r}, k={k})", fontsize=12, fontweight="bold")
        ax4.legend(loc="best", fontsize=10)
        st.pyplot(fig4)
        
        # Display comparison metrics
        st.success(f"**Original model** final population: {sol_orig.y[0][-1]:.4f}")
        st.success(f"**Modified model** final population: {sol_no_cap.y[0][-1]:.4f}")
