import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="HospitalNet AI",
    page_icon="🏥",
    layout="wide"
)

# --------------------------------------------------
# DASHBOARD STYLING
# --------------------------------------------------
st.markdown("""
<style>
.stApp { background-color: #0B1730; color: #EAF2FF; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1250px; }
h1, h2, h3 { color: #59C7FA !important; font-weight: 700; }
[data-testid="stMetric"] {
    background-color: #172A49; padding: 18px;
    border-radius: 12px; border: 1px solid #29496B;
}
[data-testid="stMetricValue"] { color: #FFFFFF; }
[data-testid="stSidebar"] { background-color: #101F39; }
.stButton > button {
    background-color: #1683C5; color: white;
    border-radius: 8px; border: none;
}
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HOSPITAL APPLICATION PROFILES (original fixed profiles)
# --------------------------------------------------
NORMAL = {
    "Remote Surgery": {"demand": 12, "weight": 5},
    "Patient Monitoring": {"demand": 10, "weight": 4},
    "Medical Telemetry": {"demand": 8, "weight": 3},
    "Doctor Communication": {"demand": 8, "weight": 2},
    "Regular Internet": {"demand": 20, "weight": 1},
}

EMERGENCY = {
    "Remote Surgery": {"demand": 20, "weight": 10},
    "Patient Monitoring": {"demand": 15, "weight": 8},
    "Medical Telemetry": {"demand": 10, "weight": 5},
    "Doctor Communication": {"demand": 8, "weight": 3},
    "Regular Internet": {"demand": 20, "weight": 1},
}

NORMAL_WEIGHTS = {k: v["weight"] for k, v in NORMAL.items()}
EMERGENCY_WEIGHTS = {k: v["weight"] for k, v in EMERGENCY.items()}

CRITICAL_APPS = ["Remote Surgery", "Patient Monitoring"]

# Per-patient bandwidth demand (Mbps) used by the ICU-aware mode
CRITICAL_PATIENT_MBPS = 2.0   # ventilator, ECG, SpO2, BP streams
STABLE_ICU_PATIENT_MBPS = 0.8
WARD_PATIENT_MBPS = 0.2
SURGERY_MBPS = 12
SURGERY_EMERGENCY_MBPS = 20


# --------------------------------------------------
# ICU-AWARE DYNAMIC PROFILE BUILDER
# Demand AND priority are computed from who is in the
# hospital right now, not from a single toggle.
# --------------------------------------------------
def build_profile(critical, stable, ward, surgeries, beds, manual_emergency):
    occupied = critical + stable

    # ICU stress index (0-100): critical patients count fully,
    # stable ICU patients count half.
    stress_index = (
        min(100.0, (critical + 0.5 * stable) / beds * 100)
        if beds else 0.0
    )

    if stress_index >= 70:
        level = "Critical"
    elif stress_index >= 40:
        level = "Busy"
    else:
        level = "Normal"

    # Emergency mode turns on automatically at Critical load,
    # or manually as an override.
    emergency = manual_emergency or level == "Critical"
    stress = 1.0 if manual_emergency else stress_index / 100
    multiplier = 1.5 if emergency else 1.0

    demands = {
        "Remote Surgery": surgeries * (
            SURGERY_EMERGENCY_MBPS if emergency else SURGERY_MBPS
        ),
        "Patient Monitoring": (
            critical * CRITICAL_PATIENT_MBPS
            + stable * STABLE_ICU_PATIENT_MBPS
        ) * multiplier,
        "Medical Telemetry": ward * WARD_PATIENT_MBPS + occupied * 0.3,
        "Doctor Communication": 4 + 0.6 * critical,
        "Regular Internet": 20,
    }

    profile = {}
    for name in NORMAL_WEIGHTS:
        weight = NORMAL_WEIGHTS[name] + (
            EMERGENCY_WEIGHTS[name] - NORMAL_WEIGHTS[name]
        ) * stress
        profile[name] = {
            "demand": round(demands[name], 2),
            "weight": round(weight, 2),
        }

    return {
        "profile": profile,
        "stress_index": stress_index,
        "level": level,
        "emergency": emergency,
        "auto_triggered": (level == "Critical" and not manual_emergency),
        "occupied": occupied,
    }


def static_baseline(profile):
    """Same demands, but fixed weights that never react to load."""
    return {
        name: {"demand": item["demand"], "weight": NORMAL_WEIGHTS[name]}
        for name, item in profile.items()
    }


# --------------------------------------------------
# SYNTHETIC CONGESTION PREDICTION MODEL (original)
# --------------------------------------------------
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 3000

    capacities = rng.integers(10, 101, n)
    emergencies = rng.integers(0, 2, n)

    demands = np.array([
        sum(
            item["demand"] * rng.uniform(0.6, 1.4)
            for item in (EMERGENCY if e else NORMAL).values()
        )
        for e in emergencies
    ])

    utilizations = 100 * demands / capacities

    unmet = np.maximum(0, demands - capacities)
    unmet_fraction = unmet / demands

    latencies = np.clip(
        15 + 150 * unmet_fraction + rng.normal(0, 5, n), 1, 300
    )
    losses = np.clip(
        15 * unmet_fraction + rng.normal(0, 0.5, n), 0, 30
    )

    X = pd.DataFrame({
        "Capacity_Mbps": capacities,
        "Demand_Mbps": demands,
        "Utilization_Percent": utilizations,
        "Latency_ms": latencies,
        "Packet_Loss_Percent": losses,
        "Emergency": emergencies
    })

    y = (
        (demands > capacities) | (latencies > 100) | (losses > 5)
    ).astype(int)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    return model


model = train_model()


# --------------------------------------------------
# WEIGHTED BANDWIDTH ALLOCATION
# --------------------------------------------------
def allocate(profile, capacity):
    allocation = {name: 0.0 for name in profile}
    remaining = float(capacity)
    active = [n for n in profile if profile[n]["demand"] > 1e-9]

    while remaining > 1e-9 and active:
        total_weight = sum(profile[n]["weight"] for n in active)
        used = 0.0
        next_active = []

        for name in active:
            share = remaining * profile[name]["weight"] / total_weight
            grant = min(share, profile[name]["demand"] - allocation[name])
            allocation[name] += grant
            used += grant

            if allocation[name] < profile[name]["demand"] - 1e-9:
                next_active.append(name)

        remaining -= used
        if used <= 1e-9:
            break
        active = next_active

    return allocation


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------
st.title("🏥 HospitalNet AI")
st.markdown(
    "**Intelligent Hospital Network Management | "
    "Adaptive Bandwidth Allocation**"
)
st.caption("TEJAS 2026 | 5G/6G Healthcare Networking Prototype")
st.divider()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Network Controls")

mode = st.sidebar.radio(
    "Simulation mode",
    ["ICU-Aware (Dynamic)", "Classic (Manual Emergency)"]
)
icu_mode = mode.startswith("ICU")

capacity = st.sidebar.slider(
    "Available bandwidth (Mbps)", 10, 100, 30, step=5
)

if icu_mode:
    st.sidebar.header("🏥 Live Hospital Load")

    icu_beds = st.sidebar.slider("Total ICU beds", 4, 30, 10)

    critical_patients = st.sidebar.slider(
        "Critical ICU patients (ventilator / unstable)",
        0, icu_beds, min(3, icu_beds)
    )

    stable_patients = st.sidebar.slider(
        "Stable ICU patients",
        0, icu_beds - critical_patients,
        min(3, icu_beds - critical_patients)
    )

    ward_patients = st.sidebar.slider(
        "General ward patients (monitored)", 0, 200, 40, step=5
    )

    active_surgeries = st.sidebar.slider(
        "Ongoing remote surgeries", 0, 3, 1
    )

emergency_toggle = st.sidebar.toggle(
    "Activate emergency mode" + (" (manual override)" if icu_mode else ""),
    value=False
)

if icu_mode:
    st.sidebar.caption(
        "In ICU-Aware mode, emergency mode also triggers "
        "automatically when ICU stress is Critical."
    )

# --------------------------------------------------
# NETWORK SIMULATION
# --------------------------------------------------
if icu_mode:
    state = build_profile(
        critical_patients, stable_patients, ward_patients,
        active_surgeries, icu_beds, emergency_toggle
    )
    profile = state["profile"]
    emergency = state["emergency"]
    stress_index = state["stress_index"]
    icu_level = state["level"]
else:
    state = None
    emergency = emergency_toggle
    stress_index = 0.0
    icu_level = "N/A"
    profile = {
        name: item.copy()
        for name, item in (EMERGENCY if emergency else NORMAL).items()
    }

demand = sum(item["demand"] for item in profile.values())
utilization = 100 * demand / capacity
unmet_demand = max(0, demand - capacity)
unmet_fraction = unmet_demand / demand if demand else 0

latency = 15 + 150 * unmet_fraction
loss = 15 * unmet_fraction

features = pd.DataFrame([{
    "Capacity_Mbps": capacity,
    "Demand_Mbps": demand,
    "Utilization_Percent": utilization,
    "Latency_ms": latency,
    "Packet_Loss_Percent": loss,
    "Emergency": int(emergency)
}])

probability = float(model.predict_proba(features)[0][1])
congested = demand > capacity

# --------------------------------------------------
# ADAPTIVE PRIORITY POLICY (original rule, Classic mode)
# In ICU-Aware mode the weights are already set dynamically.
# --------------------------------------------------
if not icu_mode and (congested or emergency):
    for name, weight in EMERGENCY_WEIGHTS.items():
        profile[name]["weight"] = weight

allocation = allocate(profile, capacity)

results = pd.DataFrame([
    {
        "Application": name,
        "Priority Weight": item["weight"],
        "Demand (Mbps)": item["demand"],
        "Allocated (Mbps)": round(allocation[name], 2),
        "Unmet Demand (Mbps)": round(
            max(0, item["demand"] - allocation[name]), 2
        )
    }
    for name, item in profile.items()
])

# --------------------------------------------------
# BASELINE COMPARISON CALCULATIONS
# --------------------------------------------------
baseline_profile = static_baseline(profile)
baseline_allocation = allocate(baseline_profile, capacity)

baseline_critical = sum(baseline_allocation[n] for n in CRITICAL_APPS)
adaptive_critical = sum(allocation[n] for n in CRITICAL_APPS)

baseline_unmet = sum(
    max(0, item["demand"] - baseline_allocation[name])
    for name, item in baseline_profile.items()
)
adaptive_unmet = sum(
    max(0, item["demand"] - allocation[name])
    for name, item in profile.items()
)

improvement_mbps = adaptive_critical - baseline_critical
improvement_percent = (
    improvement_mbps / baseline_critical * 100
    if baseline_critical > 0 else 0
)

# --------------------------------------------------
# ICU LOAD MONITOR (ICU-Aware mode)
# --------------------------------------------------
if icu_mode:
    st.subheader("🛏️ ICU Load Monitor")

    free_beds = icu_beds - state["occupied"]

    i1, i2, i3, i4 = st.columns(4)
    i1.metric("ICU Occupancy", f"{state['occupied']} / {icu_beds} beds")
    i2.metric("Critical Patients", critical_patients)
    i3.metric("ICU Stress Index", f"{stress_index:.0f}%")
    i4.metric("ICU Load Level", icu_level)

    st.progress(min(max(stress_index / 100, 0.0), 1.0))

    if state["auto_triggered"]:
        st.error(
            "🚨 Emergency mode AUTO-ACTIVATED: ICU stress is Critical "
            f"({critical_patients} critical patients, "
            f"{free_beds} free beds)."
        )
    elif emergency_toggle:
        st.error("🚨 Emergency mode activated manually.")
    elif icu_level == "Busy":
        st.warning(
            "🟡 ICU is busy. Priority weights are already shifting "
            "toward patient monitoring and surgery."
        )
    else:
        st.success("🟢 ICU load is normal.")

    st.caption(
        "Priority weights are recalculated from the live ICU patient "
        "mix. Every admission, discharge or condition change shifts "
        "the weights."
    )

# --------------------------------------------------
# MAIN NETWORK METRICS
# --------------------------------------------------
a, b, c, d = st.columns(4)
a.metric("Network Capacity", f"{capacity} Mbps")
b.metric("Total Demand", f"{demand:.1f} Mbps")
c.metric("Estimated Latency", f"{latency:.1f} ms")
d.metric("Estimated Packet Loss", f"{loss:.2f}%")

# --------------------------------------------------
# LIVE NETWORK HEALTH MONITOR
# --------------------------------------------------
st.subheader("📡 Live Network Health Monitor")

st.metric("Network Utilization", f"{utilization:.1f}%")
st.progress(min(max(utilization / 100, 0.0), 1.0))

if utilization > 100:
    st.error("🔴 OVERLOADED: Network demand exceeds capacity.")
elif utilization >= 80:
    st.warning("🟡 HIGH LOAD: Network approaching full capacity.")
else:
    st.success("🟢 STABLE: Network operating within available capacity.")

st.caption(
    "Interactive simulation. Values update when the sidebar "
    "controls change."
)

# --------------------------------------------------
# CONGESTION PREDICTION
# --------------------------------------------------
st.subheader("🧠 AI Congestion Prediction")

st.write(
    "**Simulated congestion status:**",
    "⚠️ Congestion detected" if congested else "✅ No congestion detected"
)
st.write(f"Random Forest model congestion probability: {probability:.1%}")
st.caption(
    "The Random Forest is trained on synthetic data. Its probability "
    "is experimental and does not directly control bandwidth allocation."
)

# --------------------------------------------------
# DECISION EXPLANATION PANEL
# --------------------------------------------------
st.subheader("🧠 Network Allocation Decision Explanation")

if emergency:
    st.error("🚨 Emergency Mode Activated")
    if icu_mode:
        st.write(
            f"ICU stress is {stress_index:.0f}% with "
            f"{critical_patients} critical patients. Priority for "
            "Remote Surgery, Patient Monitoring, and Medical "
            "Telemetry is increased."
        )
    else:
        st.write(
            "The system increases priority for Remote Surgery, "
            "Patient Monitoring, and Medical Telemetry because "
            "emergency mode is active."
        )
elif congested:
    st.warning("⚠️ Network Congestion Detected")
    st.write(
        "Total demand exceeds available capacity. The adaptive "
        "priority policy allocates more bandwidth to critical "
        "hospital applications."
    )
else:
    st.success("✅ Normal Network Operation")
    st.write(
        "Available capacity is sufficient to satisfy all "
        "application demands."
    )

st.write("**Current Allocation Decision:**")

if emergency:
    st.info("Emergency-priority allocation enabled.")
elif congested:
    st.info("Congestion-priority allocation enabled.")
else:
    st.info("Normal allocation priorities enabled.")

st.caption(
    "The allocation decision uses demand, capacity, and "
    + ("ICU patient load. " if icu_mode else "emergency settings. ")
    + "It is rule-based, not directly triggered by the ML prediction."
)

# --------------------------------------------------
# ICU PATIENT-LEVEL PRIORITY + ACTIONS (ICU-Aware mode)
# --------------------------------------------------
if icu_mode:
    st.subheader("🩺 ICU Patient-Level Bandwidth Priority")

    mon_multiplier = 1.5 if emergency else 1.0
    icu_patients = {}

    for i in range(critical_patients):
        icu_patients[f"ICU-{i + 1:02d} (Critical)"] = {
            "demand": CRITICAL_PATIENT_MBPS * mon_multiplier,
            "weight": 3
        }
    for j in range(stable_patients):
        idx = critical_patients + j + 1
        icu_patients[f"ICU-{idx:02d} (Stable)"] = {
            "demand": STABLE_ICU_PATIENT_MBPS * mon_multiplier,
            "weight": 1
        }

    if icu_patients:
        monitoring_budget = allocation["Patient Monitoring"]
        patient_alloc = allocate(icu_patients, monitoring_budget)

        patient_df = pd.DataFrame([
            {
                "Patient": pid,
                "Priority": "High" if info["weight"] == 3 else "Normal",
                "Demand (Mbps)": round(info["demand"], 2),
                "Allocated (Mbps)": round(patient_alloc[pid], 2),
                "Served (%)": round(
                    100 * patient_alloc[pid] / info["demand"], 1
                )
            }
            for pid, info in icu_patients.items()
        ])

        st.dataframe(patient_df, use_container_width=True, hide_index=True)
        st.caption(
            "The Patient Monitoring budget is split among ICU patients "
            "by severity. When bandwidth is short, critical patients "
            "are served first and stable patients absorb the shortfall."
        )
    else:
        st.info("No ICU patients at the moment.")

    st.subheader("📋 Recommended Actions")

    actions = []
    if free_beds <= 1:
        actions.append(
            "ICU nearly full: divert non-critical admissions to a "
            "nearby hospital."
        )
    if critical_patients >= 0.5 * icu_beds:
        actions.append(
            "Half or more of the ICU is critical: alert on-call staff "
            "and reserve bandwidth for monitoring."
        )
    if congested:
        actions.append(
            "Network congested: throttle Regular Internet and "
            "non-urgent traffic."
        )
    if active_surgeries > 0 and congested:
        actions.append(
            "Remote surgery running on a congested network: keep the "
            "surgery link on its dedicated slice."
        )
    if not actions:
        actions.append(
            "No action needed. Hospital and network are stable."
        )

    for action in actions:
        st.write(f"• {action}")

# --------------------------------------------------
# ADAPTIVE BANDWIDTH ALLOCATION
# --------------------------------------------------
st.subheader("Adaptive Bandwidth Allocation")

st.dataframe(results, use_container_width=True, hide_index=True)

chart = results.melt(
    id_vars="Application",
    value_vars=["Demand (Mbps)", "Allocated (Mbps)"],
    var_name="Measure",
    value_name="Bandwidth (Mbps)"
)

fig = px.bar(
    chart, x="Application", y="Bandwidth (Mbps)",
    color="Measure", barmode="group",
    title="Demand vs Allocated Bandwidth"
)
st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# BASELINE VS ADAPTIVE COMPARISON
# --------------------------------------------------
st.subheader("Baseline vs AI-Assisted Allocation")

comparison = pd.DataFrame([
    {
        "Method": "Baseline",
        "Critical Apps Bandwidth (Mbps)": round(baseline_critical, 2),
        "Total Unmet Demand (Mbps)": round(baseline_unmet, 2)
    },
    {
        "Method": "Adaptive Priority",
        "Critical Apps Bandwidth (Mbps)": round(adaptive_critical, 2),
        "Total Unmet Demand (Mbps)": round(adaptive_unmet, 2)
    }
])

st.dataframe(comparison, use_container_width=True, hide_index=True)

comparison_chart = px.bar(
    comparison, x="Method",
    y=["Critical Apps Bandwidth (Mbps)", "Total Unmet Demand (Mbps)"],
    barmode="group",
    title="Baseline vs Adaptive Performance"
)
st.plotly_chart(comparison_chart, use_container_width=True)

# --------------------------------------------------
# NETWORK OPTIMIZATION SUMMARY
# --------------------------------------------------
st.subheader("📊 Network Optimization Summary")

col1, col2, col3 = st.columns(3)
col1.metric("Baseline Critical Bandwidth", f"{baseline_critical:.2f} Mbps")
col2.metric("Adaptive Critical Bandwidth", f"{adaptive_critical:.2f} Mbps")
col3.metric(
    "Critical Bandwidth Improvement",
    f"{improvement_percent:+.1f}%",
    delta=f"{improvement_mbps:+.2f} Mbps"
)

if improvement_mbps > 1e-9:
    st.success(
        f"Adaptive prioritization provides {improvement_mbps:.2f} Mbps "
        f"more bandwidth to Remote Surgery and Patient Monitoring "
        f"({improvement_percent:.1f}% improvement)."
    )
elif abs(improvement_mbps) <= 1e-9:
    st.info(
        "Both allocation methods provide the same bandwidth to "
        "critical applications."
    )
else:
    st.warning(
        "Adaptive prioritization provides less bandwidth to the "
        "selected critical applications."
    )

st.caption(
    "Improvement reflects simulated priority-weighted allocation, "
    "not independently validated ML gains."
)

# --------------------------------------------------
# PER-APPLICATION BANDWIDTH COMPARISON
# --------------------------------------------------
st.subheader("Bandwidth Allocation by Hospital Application")

application_comparison = pd.DataFrame([
    {
        "Application": name,
        "Baseline (Mbps)": round(baseline_allocation[name], 2),
        "Adaptive (Mbps)": round(allocation[name], 2)
    }
    for name in profile
])

application_comparison_long = application_comparison.melt(
    id_vars="Application",
    value_vars=["Baseline (Mbps)", "Adaptive (Mbps)"],
    var_name="Allocation Method",
    value_name="Bandwidth (Mbps)"
)

application_chart = px.bar(
    application_comparison_long,
    x="Application", y="Bandwidth (Mbps)",
    color="Allocation Method", barmode="group",
    title="Baseline vs Adaptive Bandwidth per Application",
    labels={
        "Application": "Hospital Application",
        "Bandwidth (Mbps)": "Allocated Bandwidth (Mbps)"
    }
)
st.plotly_chart(application_chart, use_container_width=True)

st.caption(
    "Both methods are tested with identical simulated capacity and "
    "application demands. The adaptive method changes application "
    "priority weights."
)

# --------------------------------------------------
# AUTOMATED NETWORK SCENARIO TESTING (original)
# --------------------------------------------------
st.subheader("🧪 Automated Network Scenario Testing")

scenarios = [
    ("Normal Network", 100, False),
    ("Network Congestion", 30, False),
    ("Hospital Emergency", 60, True)
]

scenario_results = []

for scenario_name, scenario_capacity, scenario_emergency in scenarios:

    scenario_profile = {
        name: item.copy()
        for name, item in (
            EMERGENCY if scenario_emergency else NORMAL
        ).items()
    }

    scenario_demand = sum(
        item["demand"] for item in scenario_profile.values()
    )
    scenario_congested = scenario_demand > scenario_capacity

    scenario_baseline = {
        name: {"demand": item["demand"], "weight": weight}
        for (name, item), weight in zip(
            scenario_profile.items(), [5, 4, 3, 2, 1]
        )
    }

    if scenario_congested or scenario_emergency:
        for name, weight in zip(scenario_profile, [10, 8, 5, 3, 1]):
            scenario_profile[name]["weight"] = weight

    base_alloc = allocate(scenario_baseline, scenario_capacity)
    adaptive_alloc = allocate(scenario_profile, scenario_capacity)

    base_critical = sum(base_alloc[n] for n in CRITICAL_APPS)
    adaptive_critical_scenario = sum(
        adaptive_alloc[n] for n in CRITICAL_APPS
    )

    improvement = (
        (adaptive_critical_scenario - base_critical)
        / base_critical * 100
        if base_critical > 0 else 0
    )

    scenario_results.append({
        "Scenario": scenario_name,
        "Capacity (Mbps)": scenario_capacity,
        "Demand (Mbps)": scenario_demand,
        "Utilization (%)": round(
            100 * scenario_demand / scenario_capacity, 1
        ),
        "Baseline Critical (Mbps)": round(base_critical, 2),
        "Adaptive Critical (Mbps)": round(
            adaptive_critical_scenario, 2
        ),
        "Improvement (%)": round(improvement, 1)
    })

scenario_df = pd.DataFrame(scenario_results)
st.dataframe(scenario_df, use_container_width=True, hide_index=True)

scenario_fig = px.bar(
    scenario_df, x="Scenario",
    y=["Baseline Critical (Mbps)", "Adaptive Critical (Mbps)"],
    barmode="group",
    title="Critical Application Bandwidth Across Scenarios"
)
st.plotly_chart(scenario_fig, use_container_width=True)

st.caption(
    "Results are based on simulated application demands and "
    "priority-weighted bandwidth allocation."
)

# --------------------------------------------------
# AUTOMATED ICU SCENARIO TESTING (new)
# --------------------------------------------------
st.subheader("🧪 Automated ICU Scenario Testing")

# (name, capacity, beds, critical, stable, ward, surgeries)
icu_scenarios = [
    ("Quiet Night", 100, 10, 1, 2, 30, 0),
    ("Busy Day", 50, 10, 3, 4, 80, 1),
    ("ICU Nearly Full", 40, 10, 6, 3, 100, 1),
    ("Mass Casualty Event", 60, 12, 10, 2, 150, 2),
]

icu_scenario_results = []

for (s_name, s_cap, s_beds, s_crit, s_stable,
     s_ward, s_surg) in icu_scenarios:

    s_state = build_profile(
        s_crit, s_stable, s_ward, s_surg, s_beds, False
    )
    s_profile = s_state["profile"]
    s_demand = sum(i["demand"] for i in s_profile.values())

    base_alloc = allocate(static_baseline(s_profile), s_cap)
    adapt_alloc = allocate(s_profile, s_cap)

    base_crit = sum(base_alloc[n] for n in CRITICAL_APPS)
    adapt_crit = sum(adapt_alloc[n] for n in CRITICAL_APPS)

    improvement = (
        (adapt_crit - base_crit) / base_crit * 100
        if base_crit > 0 else 0
    )

    icu_scenario_results.append({
        "Scenario": s_name,
        "ICU Load": f"{s_crit + s_stable}/{s_beds} ({s_state['level']})",
        "Emergency Mode": "Auto ON" if s_state["emergency"] else "Off",
        "Capacity (Mbps)": s_cap,
        "Demand (Mbps)": round(s_demand, 1),
        "Baseline Critical (Mbps)": round(base_crit, 2),
        "Adaptive Critical (Mbps)": round(adapt_crit, 2),
        "Improvement (%)": round(improvement, 1)
    })

icu_scenario_df = pd.DataFrame(icu_scenario_results)
st.dataframe(icu_scenario_df, use_container_width=True, hide_index=True)

icu_scenario_fig = px.bar(
    icu_scenario_df, x="Scenario",
    y=["Baseline Critical (Mbps)", "Adaptive Critical (Mbps)"],
    barmode="group",
    title="Critical Application Bandwidth Across ICU Scenarios"
)
st.plotly_chart(icu_scenario_fig, use_container_width=True)

st.caption(
    "Each scenario changes the ICU patient mix, which changes both "
    "demand and priority weights automatically."
)

# --------------------------------------------------
# NETWORK TRAFFIC SIMULATION (original)
# --------------------------------------------------
st.subheader("📈 Network Traffic Simulation")

st.write(
    "Simulate changing hospital network demand and identify "
    "congestion periods."
)

simulation_steps = st.slider(
    "Number of simulation steps", 10, 100, 20, step=10
)

if st.button("▶ Generate Network Traffic"):
    rng = np.random.default_rng()

    traffic = rng.normal(loc=demand, scale=12, size=simulation_steps)
    traffic = np.maximum(traffic, 0)

    traffic_df = pd.DataFrame({
        "Time Step": range(1, simulation_steps + 1),
        "Demand (Mbps)": traffic,
        "Capacity (Mbps)": [capacity] * simulation_steps
    })
    traffic_df["Congested"] = (
        traffic_df["Demand (Mbps)"] > capacity
    )

    st.session_state["traffic_df"] = traffic_df

if "traffic_df" in st.session_state:
    traffic_df = st.session_state["traffic_df"]

    traffic_chart = px.line(
        traffic_df, x="Time Step",
        y=["Demand (Mbps)", "Capacity (Mbps)"],
        markers=True,
        title="Simulated Hospital Network Traffic"
    )
    st.plotly_chart(traffic_chart, use_container_width=True)

    congestion_count = int(traffic_df["Congested"].sum())
    total_steps = len(traffic_df)
    congestion_percentage = (
        congestion_count / total_steps * 100 if total_steps else 0
    )

    a, b, c = st.columns(3)
    a.metric("Simulation Steps", total_steps)
    b.metric("Congested Steps", congestion_count)
    c.metric("Congestion Frequency", f"{congestion_percentage:.1f}%")

    if congestion_count > 0:
        st.warning(
            f"⚠️ Congestion occurred during {congestion_count} of "
            f"{total_steps} simulated time steps."
        )
    else:
        st.success("✅ No congestion occurred during the simulation.")
