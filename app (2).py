
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
.stApp {
    background-color: #0B1730;
    color: #EAF2FF;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}
h1, h2, h3 {
    color: #59C7FA !important;
    font-weight: 700;
}
[data-testid="stMetric"] {
    background-color: #172A49;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #29496B;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF;
}
[data-testid="stSidebar"] {
    background-color: #101F39;
}
.stButton > button {
    background-color: #1683C5;
    color: white;
    border-radius: 8px;
    border: none;
}
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HOSPITAL APPLICATION PROFILES
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

# --------------------------------------------------
# SYNTHETIC CONGESTION PREDICTION MODEL
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
            for item in (
                EMERGENCY if e else NORMAL
            ).values()
        )
        for e in emergencies
    ])

    utilizations = 100 * demands / capacities

    unmet = np.maximum(0, demands - capacities)
    unmet_fraction = unmet / demands

    latencies = np.clip(
        15 + 150 * unmet_fraction
        + rng.normal(0, 5, n),
        1, 300
    )

    losses = np.clip(
        15 * unmet_fraction
        + rng.normal(0, 0.5, n),
        0, 30
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
        (demands > capacities)
        | (latencies > 100)
        | (losses > 5)
    ).astype(int)

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X, y)
    return model


model = train_model()

# --------------------------------------------------
# WEIGHTED BANDWIDTH ALLOCATION
# --------------------------------------------------
def allocate(profile, capacity):
    allocation = {
        name: 0.0 for name in profile
    }

    remaining = float(capacity)
    active = list(profile)

    while remaining > 1e-9 and active:
        total_weight = sum(
            profile[name]["weight"]
            for name in active
        )

        used = 0.0
        next_active = []

        for name in active:
            share = (
                remaining
                * profile[name]["weight"]
                / total_weight
            )

            grant = min(
                share,
                profile[name]["demand"]
                - allocation[name]
            )

            allocation[name] += grant
            used += grant

            if (
                allocation[name]
                < profile[name]["demand"] - 1e-9
            ):
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

st.caption(
    "TEJAS 2026 | 5G/6G Healthcare Networking Prototype"
)

st.divider()

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Network Controls")

capacity = st.sidebar.slider(
    "Available bandwidth (Mbps)",
    min_value=10,
    max_value=100,
    value=30,
    step=5
)

emergency = st.sidebar.toggle(
    "Activate emergency mode",
    value=False
)

# --------------------------------------------------
# NETWORK SIMULATION
# --------------------------------------------------
profile = {
    name: item.copy()
    for name, item in (
        EMERGENCY if emergency else NORMAL
    ).items()
}

demand = sum(
    item["demand"]
    for item in profile.values()
)

utilization = 100 * demand / capacity

unmet_demand = max(0, demand - capacity)

unmet_fraction = (
    unmet_demand / demand if demand else 0
)

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

probability = float(
    model.predict_proba(features)[0][1]
)

congested = demand > capacity

# --------------------------------------------------
# ADAPTIVE PRIORITY POLICY
# --------------------------------------------------
if congested or emergency:
    profile["Remote Surgery"]["weight"] = 10
    profile["Patient Monitoring"]["weight"] = 8
    profile["Medical Telemetry"]["weight"] = 5
    profile["Doctor Communication"]["weight"] = 3
    profile["Regular Internet"]["weight"] = 1

allocation = allocate(profile, capacity)

results = pd.DataFrame([
    {
        "Application": name,
        "Demand (Mbps)": item["demand"],
        "Allocated (Mbps)": round(
            allocation[name], 2
        ),
        "Unmet Demand (Mbps)": round(
            max(
                0,
                item["demand"] - allocation[name]
            ),
            2
        )
    }
    for name, item in profile.items()
])

# --------------------------------------------------
# BASELINE COMPARISON CALCULATIONS
# --------------------------------------------------
# Calculate these BEFORE displaying the summary.

baseline_profile = {
    name: {
        "demand": item["demand"],
        "weight": weight
    }
    for (name, item), weight in zip(
        profile.items(),
        [5, 4, 3, 2, 1]
    )
}

baseline_allocation = allocate(
    baseline_profile,
    capacity
)

critical_apps = [
    "Remote Surgery",
    "Patient Monitoring"
]

baseline_critical = sum(
    baseline_allocation[name]
    for name in critical_apps
)

adaptive_critical = sum(
    allocation[name]
    for name in critical_apps
)

baseline_unmet = sum(
    max(
        0,
        item["demand"]
        - baseline_allocation[name]
    )
    for name, item in baseline_profile.items()
)

adaptive_unmet = sum(
    max(
        0,
        item["demand"]
        - allocation[name]
    )
    for name, item in profile.items()
)

improvement_mbps = (
    adaptive_critical - baseline_critical
)

improvement_percent = (
    improvement_mbps / baseline_critical * 100
    if baseline_critical > 0 else 0
)

# --------------------------------------------------
# MAIN NETWORK METRICS
# --------------------------------------------------
a, b, c, d = st.columns(4)

a.metric(
    "Network Capacity",
    f"{capacity} Mbps"
)

b.metric(
    "Total Demand",
    f"{demand} Mbps"
)

c.metric(
    "Estimated Latency",
    f"{latency:.1f} ms"
)

d.metric(
    "Estimated Packet Loss",
    f"{loss:.2f}%"
)

# --------------------------------------------------
# LIVE NETWORK HEALTH MONITOR
# --------------------------------------------------
st.subheader("📡 Live Network Health Monitor")

st.metric(
    "Network Utilization",
    f"{utilization:.1f}%"
)

st.progress(
    min(max(utilization / 100, 0.0), 1.0)
)

if utilization > 100:
    st.error(
        "🔴 OVERLOADED: Network demand exceeds capacity."
    )
elif utilization >= 80:
    st.warning(
        "🟡 HIGH LOAD: Network approaching full capacity."
    )
else:
    st.success(
        "🟢 STABLE: Network operating within available capacity."
    )

st.caption(
    "Interactive simulation. Values update when "
    "the sidebar controls change."
)

# --------------------------------------------------
# CONGESTION PREDICTION
# --------------------------------------------------
st.subheader("🧠 AI Congestion Prediction")

st.write(
    "**Simulated congestion status:**",
    "⚠️ Congestion detected"
    if congested
    else "✅ No congestion detected"
)

st.write(
    f"Random Forest model congestion probability: "
    f"{probability:.1%}"
)

st.caption(
    "The Random Forest is trained on synthetic data. "
    "Its probability is experimental and does not "
    "directly control bandwidth allocation."
)

# --------------------------------------------------
# DECISION EXPLANATION PANEL
# --------------------------------------------------
st.subheader(
    "🧠 Network Allocation Decision Explanation"
)

if emergency:
    st.error("🚨 Emergency Mode Activated")
    st.write(
        "The system increases priority for Remote Surgery, "
        "Patient Monitoring, and Medical Telemetry because "
        "emergency mode is active."
    )

elif congested:
    st.warning("⚠️ Network Congestion Detected")
    st.write(
        "Total demand exceeds available capacity. "
        "The adaptive priority policy allocates more "
        "bandwidth to critical hospital applications."
    )

else:
    st.success("✅ Normal Network Operation")
    st.write(
        "Available capacity is sufficient to satisfy "
        "all application demands."
    )

st.write("**Current Allocation Decision:**")

if emergency:
    st.info("Emergency-priority allocation enabled.")
elif congested:
    st.info("Congestion-priority allocation enabled.")
else:
    st.info("Normal allocation priorities enabled.")

st.caption(
    "The allocation decision uses demand, capacity, "
    "and emergency settings. It is rule-based, "
    "not directly triggered by the ML prediction."
)

# --------------------------------------------------
# ADAPTIVE BANDWIDTH ALLOCATION
# --------------------------------------------------
st.subheader("Adaptive Bandwidth Allocation")

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)

chart = results.melt(
    id_vars="Application",
    value_vars=[
        "Demand (Mbps)",
        "Allocated (Mbps)"
    ],
    var_name="Measure",
    value_name="Bandwidth (Mbps)"
)

fig = px.bar(
    chart,
    x="Application",
    y="Bandwidth (Mbps)",
    color="Measure",
    barmode="group",
    title="Demand vs Allocated Bandwidth"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# --------------------------------------------------
# BASELINE VS ADAPTIVE COMPARISON
# --------------------------------------------------
st.subheader("Baseline vs AI-Assisted Allocation")

comparison = pd.DataFrame([
    {
        "Method": "Baseline",
        "Critical Apps Bandwidth (Mbps)":
            round(baseline_critical, 2),
        "Total Unmet Demand (Mbps)":
            round(baseline_unmet, 2)
    },
    {
        "Method": "Adaptive Priority",
        "Critical Apps Bandwidth (Mbps)":
            round(adaptive_critical, 2),
        "Total Unmet Demand (Mbps)":
            round(adaptive_unmet, 2)
    }
])

st.dataframe(
    comparison,
    use_container_width=True,
    hide_index=True
)

comparison_chart = px.bar(
    comparison,
    x="Method",
    y=[
        "Critical Apps Bandwidth (Mbps)",
        "Total Unmet Demand (Mbps)"
    ],
    barmode="group",
    title="Baseline vs Adaptive Performance"
)

st.plotly_chart(
    comparison_chart,
    use_container_width=True
)

# --------------------------------------------------
# NETWORK OPTIMIZATION SUMMARY
# --------------------------------------------------
st.subheader("📊 Network Optimization Summary")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Baseline Critical Bandwidth",
    f"{baseline_critical:.2f} Mbps"
)

col2.metric(
    "Adaptive Critical Bandwidth",
    f"{adaptive_critical:.2f} Mbps"
)

col3.metric(
    "Critical Bandwidth Improvement",
    f"{improvement_percent:+.1f}%",
    delta=f"{improvement_mbps:+.2f} Mbps"
)

if improvement_mbps > 1e-9:
    st.success(
        f"Adaptive prioritization provides "
        f"{improvement_mbps:.2f} Mbps more bandwidth "
        f"to Remote Surgery and Patient Monitoring "
        f"({improvement_percent:.1f}% improvement)."
    )
elif abs(improvement_mbps) <= 1e-9:
    st.info(
        "Both allocation methods provide the same "
        "bandwidth to critical applications."
    )
else:
    st.warning(
        "Adaptive prioritization provides less "
        "bandwidth to the selected critical applications."
    )

st.caption(
    "Improvement reflects simulated priority-weighted "
    "allocation, not independently validated ML gains."
)

# --------------------------------------------------
# PER-APPLICATION BANDWIDTH COMPARISON
# --------------------------------------------------
st.subheader(
    "Bandwidth Allocation by Hospital Application"
)

application_comparison = pd.DataFrame([
    {
        "Application": name,
        "Baseline (Mbps)": round(
            baseline_allocation[name], 2
        ),
        "Adaptive (Mbps)": round(
            allocation[name], 2
        )
    }
    for name in profile
])

application_comparison_long = (
    application_comparison.melt(
        id_vars="Application",
        value_vars=[
            "Baseline (Mbps)",
            "Adaptive (Mbps)"
        ],
        var_name="Allocation Method",
        value_name="Bandwidth (Mbps)"
    )
)

application_chart = px.bar(
    application_comparison_long,
    x="Application",
    y="Bandwidth (Mbps)",
    color="Allocation Method",
    barmode="group",
    title="Baseline vs Adaptive Bandwidth per Application",
    labels={
        "Application": "Hospital Application",
        "Bandwidth (Mbps)":
            "Allocated Bandwidth (Mbps)"
    }
)

st.plotly_chart(
    application_chart,
    use_container_width=True
)

st.caption(
    "Both methods are tested with identical simulated "
    "capacity and application demands. The adaptive "
    "method changes application priority weights."
)

# --------------------------------------------------
# AUTOMATED NETWORK SCENARIO TESTING
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
        name: {
            "demand": item["demand"],
            "weight": weight
        }
        for (name, item), weight in zip(
            scenario_profile.items(),
            [5, 4, 3, 2, 1]
        )
    }

    if scenario_congested or scenario_emergency:
        for name, weight in zip(
            scenario_profile,
            [10, 8, 5, 3, 1]
        ):
            scenario_profile[name]["weight"] = weight

    base_alloc = allocate(
        scenario_baseline, scenario_capacity
    )

    adaptive_alloc = allocate(
        scenario_profile, scenario_capacity
    )

    base_critical = sum(
        base_alloc[name] for name in critical_apps
    )

    adaptive_critical_scenario = sum(
        adaptive_alloc[name] for name in critical_apps
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
        "Baseline Critical (Mbps)": round(
            base_critical, 2
        ),
        "Adaptive Critical (Mbps)": round(
            adaptive_critical_scenario, 2
        ),
        "Improvement (%)": round(improvement, 1)
    })

scenario_df = pd.DataFrame(scenario_results)

st.dataframe(
    scenario_df,
    use_container_width=True,
    hide_index=True
)

scenario_fig = px.bar(
    scenario_df,
    x="Scenario",
    y=[
        "Baseline Critical (Mbps)",
        "Adaptive Critical (Mbps)"
    ],
    barmode="group",
    title="Critical Application Bandwidth Across Scenarios"
)

st.plotly_chart(
    scenario_fig,
    use_container_width=True
)

st.caption(
    "Results are based on simulated application demands "
    "and priority-weighted bandwidth allocation."
)

# --------------------------------------------------
# STEP 10: NETWORK TRAFFIC SIMULATION
# --------------------------------------------------
st.subheader("📈 Network Traffic Simulation")

st.write(
    "Simulate changing hospital network demand "
    "and identify congestion periods."
)

simulation_steps = st.slider(
    "Number of simulation steps",
    min_value=10,
    max_value=100,
    value=20,
    step=10
)

if st.button("▶ Generate Network Traffic"):

    rng = np.random.default_rng()

    traffic = rng.normal(
        loc=demand,
        scale=12,
        size=simulation_steps
    )

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
        traffic_df,
        x="Time Step",
        y=["Demand (Mbps)", "Capacity (Mbps)"],
        markers=True,
        title="Simulated Hospital Network Traffic"
    )

    st.plotly_chart(
        traffic_chart,
        use_container_width=True
    )

    congestion_count = int(
        traffic_df["Congested"].sum()
    )

    total_steps = len(traffic_df)

    congestion_percentage = (
        congestion_count / total_steps * 100
        if total_steps else 0
    )

    a, b, c = st.columns(3)

    a.metric(
        "Simulation Steps",
        total_steps
    )

    b.metric(
        "Congested Steps",
        congestion_count
    )

    c.metric(
        "Congestion Frequency",
        f"{congestion_percentage:.1f}%"
    )

    if congestion_count > 0:
        st.warning(
            f"⚠️ Congestion occurred during "
            f"{congestion_count} of {total_steps} "
            f"simulated time steps."
        )
    else:
        st.success(
            "✅ No congestion detected during "
            "this simulation."
        )

st.caption(
    "Synthetic traffic simulation only. "
    "This graph does not represent live hospital "
    "network measurements."
)

# --------------------------------------------------
# STEP 11: DYNAMIC BANDWIDTH ALLOCATION
# --------------------------------------------------
st.subheader("⚡ Dynamic Bandwidth Allocation Simulation")

if "traffic_df" in st.session_state:

    traffic_data = st.session_state["traffic_df"]

    dynamic_results = []

    for _, traffic_row in traffic_data.iterrows():

        current_demand = float(
            traffic_row["Demand (Mbps)"]
        )

        current_capacity = float(
            traffic_row["Capacity (Mbps)"]
        )

        # Scale application demands proportionally
        demand_scale = (
            current_demand / demand if demand > 0 else 0
        )

        current_profile = {
            name: {
                "demand": item["demand"] * demand_scale,
                "weight": item["weight"]
            }
            for name, item in (
                EMERGENCY if emergency else NORMAL
            ).items()
        }

        baseline_profile_step = {
            name: {
                "demand": item["demand"],
                "weight": weight
            }
            for (name, item), weight in zip(
                current_profile.items(),
                [5, 4, 3, 2, 1]
            )
        }

        if current_demand > current_capacity or emergency:
            for name, weight in zip(
                current_profile,
                [10, 8, 5, 3, 1]
            ):
                current_profile[name]["weight"] = weight

        baseline_step = allocate(
            baseline_profile_step,
            current_capacity
        )

        adaptive_step = allocate(
            current_profile,
            current_capacity
        )

        baseline_critical_step = sum(
            baseline_step[name]
            for name in critical_apps
        )

        adaptive_critical_step = sum(
            adaptive_step[name]
            for name in critical_apps
        )

        dynamic_results.append({
            "Time Step": int(traffic_row["Time Step"]),
            "Baseline Critical (Mbps)":
                baseline_critical_step,
            "Adaptive Critical (Mbps)":
                adaptive_critical_step
        })

    dynamic_df = pd.DataFrame(dynamic_results)

    dynamic_chart = px.line(
        dynamic_df,
        x="Time Step",
        y=[
            "Baseline Critical (Mbps)",
            "Adaptive Critical (Mbps)"
        ],
        markers=True,
        title="Critical Bandwidth Allocation Over Time"
    )

    st.plotly_chart(
        dynamic_chart,
        use_container_width=True
    )

    total_baseline = dynamic_df[
        "Baseline Critical (Mbps)"
    ].sum()

    total_adaptive = dynamic_df[
        "Adaptive Critical (Mbps)"
    ].sum()

    gain = total_adaptive - total_baseline

    gain_percent = (
        gain / total_baseline * 100
        if total_baseline > 0 else 0
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Average Baseline Critical Bandwidth",
        f"{dynamic_df['Baseline Critical (Mbps)'].mean():.2f} Mbps"
    )

    col2.metric(
        "Average Adaptive Critical Bandwidth",
        f"{dynamic_df['Adaptive Critical (Mbps)'].mean():.2f} Mbps"
    )

    col3.metric(
        "Relative Critical Bandwidth Gain",
        f"{gain_percent:+.1f}%"
    )

    st.caption(
        "Application demands are proportionally scaled "
        "using synthetic traffic. Adaptive allocation "
        "changes priority weights during congestion "
        "or emergency mode. These are simulated results."
    )

else:
    st.info(
        "Generate network traffic in Step 10 first "
        "to see dynamic bandwidth allocation."
    )

# --------------------------------------------------
# STEP 12: TEJAS LIVE DEMONSTRATION PANEL
# --------------------------------------------------
st.divider()
st.subheader("🎯 TEJAS Live Demonstration Panel")

st.write(
    "This panel summarizes the current simulated "
    "hospital network conditions and allocation decisions."
)

# Determine network status
if emergency:
    demo_status = "🚨 Emergency Mode"
elif congested:
    demo_status = "🔴 Network Congested"
elif utilization >= 80:
    demo_status = "🟡 High Network Load"
else:
    demo_status = "🟢 Network Normal"

# Display summary metrics
demo1, demo2, demo3 = st.columns(3)

demo1.metric(
    "Network Utilization",
    f"{utilization:.1f}%"
)

demo2.metric(
    "Critical Bandwidth Gain",
    f"{improvement_percent:+.1f}%"
)

demo3.metric(
    "Congestion Probability",
    f"{probability:.1%}"
)

st.write(f"**Current Network Status:** {demo_status}")

# Explain the current decision
if emergency:
    st.error(
        "Emergency mode is active. The allocation policy "
        "prioritizes Remote Surgery, Patient Monitoring, "
        "and Medical Telemetry."
    )
elif congested:
    st.warning(
        "Network demand exceeds capacity. Adaptive "
        "prioritization is active to protect critical "
        "application bandwidth."
    )
else:
    st.success(
        "Network demand is within available capacity. "
        "Normal allocation priorities are active."
    )

st.caption(
    "TEJAS demonstration uses synthetic network data. "
    "Congestion probability is estimated by a Random Forest "
    "model, while allocation decisions follow a "
    "rule-based priority policy."
)

# --------------------------------------------------
# STEP 13: AUTOMATED PROTOTYPE VALIDATION
# --------------------------------------------------
st.divider()
st.subheader("🧪 Prototype Validation Tests")

st.write(
    "Automated checks for simulated bandwidth "
    "allocation and emergency prioritization."
)

validation_results = []

# Test 1: Total allocation must not exceed capacity
total_allocated = sum(allocation.values())

validation_results.append({
    "Test": "Capacity Constraint",
    "Result": "PASS" if total_allocated <= capacity + 1e-6 else "FAIL",
    "Details": f"Allocated {total_allocated:.2f} of {capacity} Mbps"
})

# Test 2: No application exceeds its requested demand
demand_limits_valid = all(
    allocation[name] <= profile[name]["demand"] + 1e-6
    for name in profile
)

validation_results.append({
    "Test": "Application Demand Limits",
    "Result": "PASS" if demand_limits_valid else "FAIL",
    "Details": "Allocated bandwidth does not exceed application demand"
})

# Test 3: Allocation values must be nonnegative
nonnegative_valid = all(
    value >= -1e-6 for value in allocation.values()
)

validation_results.append({
    "Test": "Nonnegative Allocation",
    "Result": "PASS" if nonnegative_valid else "FAIL",
    "Details": "All allocated bandwidth values are nonnegative"
})

# Test 4: Adaptive policy should not reduce combined
# critical bandwidth under the current simulated profile
critical_priority_valid = adaptive_critical >= baseline_critical - 1e-6

validation_results.append({
    "Test": "Critical Application Prioritization",
    "Result": "PASS" if critical_priority_valid else "FAIL",
    "Details": (
        f"Baseline: {baseline_critical:.2f} Mbps; "
        f"Adaptive: {adaptive_critical:.2f} Mbps"
    )
})

# Test 5: Total unmet demand should be consistent
expected_unmet = max(0, demand - capacity)

unmet_valid = (
    abs(adaptive_unmet - expected_unmet) <= 1e-6
)

validation_results.append({
    "Test": "Unmet Demand Consistency",
    "Result": "PASS" if unmet_valid else "FAIL",
    "Details": (
        f"Expected: {expected_unmet:.2f} Mbps; "
        f"Actual: {adaptive_unmet:.2f} Mbps"
    )
})

validation_df = pd.DataFrame(validation_results)

st.dataframe(
    validation_df,
    use_container_width=True,
    hide_index=True
)

passed_tests = int(
    (validation_df["Result"] == "PASS").sum()
)

total_tests = len(validation_df)

st.metric(
    "Validation Tests Passed",
    f"{passed_tests}/{total_tests}"
)

if passed_tests == total_tests:
    st.success(
        "All allocation validation checks passed "
        "for the current simulation settings."
    )
else:
    st.error(
        "One or more allocation checks failed. "
        "Review the allocation logic before presenting."
    )

st.caption(
    "These checks verify basic allocation constraints "
    "for the selected scenario. They do not validate "
    "real-world hospital network reliability or "
    "machine-learning prediction accuracy."
)

# --------------------------------------------------
# STEP 14: RANDOM FOREST MODEL EVALUATION
# --------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

st.divider()
st.subheader("🤖 AI Model Performance Evaluation")

st.write(
    "Evaluate a Random Forest classifier on held-out "
    "synthetic network scenarios."
)

@st.cache_data
def evaluate_synthetic_model():
    rng = np.random.default_rng(2026)
    n = 3000

    capacities = rng.integers(10, 101, n)
    emergencies = rng.integers(0, 2, n)

    demands = np.array([
        sum(
            item["demand"] * rng.uniform(0.6, 1.4)
            for item in (
                EMERGENCY if e else NORMAL
            ).values()
        )
        for e in emergencies
    ])

    utilizations = 100 * demands / capacities

    unmet = np.maximum(0, demands - capacities)
    unmet_fraction = unmet / demands

    latencies = np.clip(
        15 + 150 * unmet_fraction
        + rng.normal(0, 5, n),
        1, 300
    )

    losses = np.clip(
        15 * unmet_fraction
        + rng.normal(0, 0.5, n),
        0, 30
    )

    X_eval = pd.DataFrame({
        "Capacity_Mbps": capacities,
        "Demand_Mbps": demands,
        "Utilization_Percent": utilizations,
        "Latency_ms": latencies,
        "Packet_Loss_Percent": losses,
        "Emergency": emergencies
    })

    y_eval = (
        (demands > capacities)
        | (latencies > 100)
        | (losses > 5)
    ).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_eval,
        y_eval,
        test_size=0.25,
        random_state=42,
        stratify=y_eval
    )

    evaluation_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight="balanced"
    )

    evaluation_model.fit(X_train, y_train)
    predictions = evaluation_model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1-score": f1_score(
            y_test, predictions, zero_division=0
        )
    }

    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    return metrics, matrix, len(y_test)


metrics, matrix, test_size = evaluate_synthetic_model()

m1, m2, m3, m4 = st.columns(4)

m1.metric("Accuracy", f"{metrics['Accuracy']:.1%}")
m2.metric("Precision", f"{metrics['Precision']:.1%}")
m3.metric("Recall", f"{metrics['Recall']:.1%}")
m4.metric("F1-score", f"{metrics['F1-score']:.1%}")

st.caption(
    f"Evaluation uses {test_size} held-out synthetic test cases. "
    "These metrics evaluate a separately trained model using "
    "the same synthetic-data generation approach; they are "
    "not measurements of real hospital network performance."
)

st.markdown("#### Confusion Matrix")

confusion_df = pd.DataFrame(
    matrix,
    index=["Actual: No Congestion", "Actual: Congestion"],
    columns=["Predicted: No Congestion", "Predicted: Congestion"]
)

st.dataframe(
    confusion_df,
    use_container_width=True
)

st.caption(
    "Synthetic labels are generated from simulated demand, "
    "latency, and packet-loss rules. High scores do not "
    "establish real-world predictive accuracy."
)

# ==========================================================
# STEP 15: NETWORK FAILURE AND RECOVERY SIMULATION
# ==========================================================

st.divider()
st.header("🚨 Step 15: Network Failure and Recovery Simulation")

st.write(
    "Simulate a sudden hospital network failure and observe "
    "how priority-based bandwidth allocation responds."
)

failure_capacity = st.slider(
    "Normal Network Capacity (Mbps)",
    min_value=20,
    max_value=100,
    value=60,
    step=10,
    key="failure_normal_capacity"
)

failure_drop = st.slider(
    "Capacity Reduction During Failure (%)",
    min_value=10,
    max_value=90,
    value=60,
    step=10,
    key="failure_capacity_drop"
)

normal_capacity = float(failure_capacity)
failed_capacity = normal_capacity * (1 - failure_drop / 100)
recovered_capacity = normal_capacity

# Use the current hospital traffic profile
failure_demand = float(demand)

failure_stages = [
    ("Normal", normal_capacity),
    ("Network Failure", failed_capacity),
    ("Recovery", recovered_capacity),
]

failure_results = []

critical_apps = ["Remote Surgery", "Patient Monitoring"]

for stage, stage_capacity in failure_stages:
    stage_baseline = allocate(profile, stage_capacity)

    stage_profile = {
        app: {
            "demand": details["demand"],
            "weight": (
                10 if app == "Remote Surgery"
                else 8 if app == "Patient Monitoring"
                else details["weight"]
            )
        }
        for app, details in profile.items()
    }

    stage_adaptive = allocate(stage_profile, stage_capacity)

    baseline_critical = sum(
        stage_baseline.get(app, 0)
        for app in critical_apps
    )

    adaptive_critical = sum(
        stage_adaptive.get(app, 0)
        for app in critical_apps
    )

    failure_results.append({
        "Stage": stage,
        "Capacity (Mbps)": round(stage_capacity, 2),
        "Demand (Mbps)": round(failure_demand, 2),
        "Baseline Critical (Mbps)": round(baseline_critical, 2),
        "Priority Critical (Mbps)": round(adaptive_critical, 2),
        "Unmet Demand (Mbps)": round(
            max(0, failure_demand - stage_capacity), 2
        ),
    })

failure_df = pd.DataFrame(failure_results)

st.subheader("Network Failure Timeline")
st.dataframe(failure_df, use_container_width=True, hide_index=True)

chart_df = failure_df.melt(
    id_vars="Stage",
    value_vars=[
        "Baseline Critical (Mbps)",
        "Priority Critical (Mbps)"
    ],
    var_name="Allocation Policy",
    value_name="Critical Bandwidth (Mbps)"
)

fig_failure = px.bar(
    chart_df,
    x="Stage",
    y="Critical Bandwidth (Mbps)",
    color="Allocation Policy",
    barmode="group",
    title="Critical Hospital Bandwidth During Failure and Recovery"
)

st.plotly_chart(fig_failure, use_container_width=True)

st.info(
    "Simulation only: this demonstrates a software allocation policy. "
    "It does not control real hospital routers or guarantee real-time "
    "network recovery."
)

# ==========================================================
# STEP 16: EMERGENCY ALERT AND RECOVERY RECOMMENDATIONS
# ==========================================================

st.divider()
st.header("🚨 Step 16: Emergency Alert and Recovery Recommendations")

st.write(
    "TEJAS analyses the simulated network failure and recommends "
    "actions to protect critical hospital applications."
)

# Read the failure results generated in Step 15
failure_row = failure_df[
    failure_df["Stage"] == "Network Failure"
].iloc[0]

normal_row = failure_df[
    failure_df["Stage"] == "Normal"
].iloc[0]

recovery_row = failure_df[
    failure_df["Stage"] == "Recovery"
].iloc[0]

failure_capacity_now = float(failure_row["Capacity (Mbps)"])
failure_demand_now = float(failure_row["Demand (Mbps)"])
failure_unmet_now = float(failure_row["Unmet Demand (Mbps)"])

baseline_critical_now = float(
    failure_row["Baseline Critical (Mbps)"]
)

priority_critical_now = float(
    failure_row["Priority Critical (Mbps)"]
)

# Display current simulated failure status
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Failure Capacity",
        f"{failure_capacity_now:.1f} Mbps"
    )

with col2:
    st.metric(
        "Unmet Demand",
        f"{failure_unmet_now:.1f} Mbps"
    )

with col3:
    st.metric(
        "Critical Bandwidth Gain",
        f"{priority_critical_now - baseline_critical_now:.1f} Mbps"
    )

# Generate a rule-based recommendation
if failure_unmet_now > 0:
    st.error("🚨 SIMULATED NETWORK CONGESTION DETECTED")

    st.subheader("Recommended Actions")

    st.markdown(
        """
        1. Prioritize Remote Surgery and Patient Monitoring traffic.
        2. Reduce or defer non-critical traffic where operationally appropriate.
        3. Ask the network administrator to investigate the capacity reduction.
        4. Restore available network capacity and verify critical application connectivity.
        """
    )

    if priority_critical_now > baseline_critical_now:
        st.success(
            "The simulated priority policy allocates more bandwidth "
            "to critical applications than the baseline."
        )
    else:
        st.warning(
            "The priority policy has not increased critical bandwidth "
            "in this scenario. Review the allocation policy before claiming "
            "an improvement."
        )

else:
    st.success(
        "No unmet bandwidth demand is detected in the simulated "
        "failure scenario. Continue monitoring network capacity."
    )

# Compare the simulated recovery with normal operation
st.subheader("Recovery Verification")

normal_capacity_now = float(normal_row["Capacity (Mbps)"])
recovered_capacity_now = float(recovery_row["Capacity (Mbps)"])

if recovered_capacity_now >= normal_capacity_now:
    st.success(
        f"Simulated recovery capacity is {recovered_capacity_now:.1f} Mbps. "
        "It has returned to the normal capacity setting."
    )
else:
    st.warning(
        "Simulated recovery capacity is below the normal capacity setting. "
        "Further investigation is recommended."
    )

st.info(
    "This is a rule-based software demonstration using simulated data. "
    "It does not send real emergency notifications, change router settings, "
    "or restore an actual hospital network."
)

# ==========================================================
# STEP 17: NETWORK PERFORMANCE REPORT
# ==========================================================

st.divider()
st.header("📊 Step 17: Network Performance Report")

st.write(
    "Review the current simulated hospital network performance "
    "and download a report for analysis."
)

# Collect current network metrics
report_metrics = {
    "Metric": [
        "Operating Mode",
        "Total Bandwidth Demand (Mbps)",
        "Available Bandwidth (Mbps)",
        "Unmet Demand (Mbps)",
        "Network Utilization (%)",
        "Estimated Latency (ms)",
        "Estimated Packet Loss (%)",
        "ML Congestion Probability (%)",
        "Baseline Critical Bandwidth (Mbps)",
        "Priority Critical Bandwidth (Mbps)",
        "Critical Bandwidth Improvement (Mbps)",
        "Network Status",
    ],
    "Value": [
        "Emergency" if emergency else "Normal",
        round(float(demand), 2),
        round(float(capacity), 2),
        round(float(unmet_demand), 2),
        round(float(utilization), 2),
        round(float(latency), 2),
        round(float(loss), 2),
        round(float(probability) * 100, 2),
        round(float(baseline_critical), 2),
        round(float(adaptive_critical), 2),
        round(float(adaptive_critical - baseline_critical), 2),
        "Congested" if congested else "Within Capacity",
    ],
}

report_df = pd.DataFrame(report_metrics)

st.subheader("Current Network Summary")
st.dataframe(report_df, use_container_width=True, hide_index=True)

# Create a downloadable CSV report
report_csv = report_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="📥 Download Network Performance Report (CSV)",
    data=report_csv,
    file_name="tejas_network_performance_report.csv",
    mime="text/csv",
    key="download_tejas_report",
)
st.caption(
    "Report values are generated from the current prototype simulation. "
    "ML congestion probability is a model output, not a guarantee. "
    "Latency and packet loss are estimated values, not live measurements."
)

# ==========================================================
# STEP 18: NETWORK EVENT HISTORY AND AUDIT LOG
# ==========================================================

st.divider()
st.header("📝 Step 18: Network Event History and Audit Log")

st.write(
    "Review a timestamped audit log of the current simulated "
    "hospital network state."
)

from datetime import datetime

# Initialize the event history for this browser session
if "tejas_event_log" not in st.session_state:
    st.session_state["tejas_event_log"] = []

# Build a signature for the current network state
current_signature = (
    bool(emergency),
    round(float(capacity), 2),
    round(float(demand), 2),
    bool(congested),
)

previous_signature = st.session_state.get(
    "tejas_previous_signature"
)

# Record events only when the state changes
if previous_signature is None:
    st.session_state["tejas_event_log"].append({
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Event": "Monitoring Started",
        "Details": (
            f"Mode: {'Emergency' if emergency else 'Normal'}; "
            f"Capacity: {capacity:.1f} Mbps; Demand: {demand:.1f} Mbps"
        ),
    })

else:
    old_emergency, old_capacity, old_demand, old_congested = (
        previous_signature
    )

    if bool(emergency) != old_emergency:
        st.session_state["tejas_event_log"].append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Event": "Operating Mode Changed",
            "Details": (
                f"{'Emergency' if emergency else 'Normal'} mode enabled"
            ),
        })

    if round(float(capacity), 2) != old_capacity:
        st.session_state["tejas_event_log"].append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Event": "Capacity Changed",
            "Details": (
                f"Capacity changed from {old_capacity:.1f} "
                f"to {capacity:.1f} Mbps"
            ),
        })

    if bool(congested) != old_congested:
        event_name = (
            "Network Congestion Detected"
            if congested
            else "Network Congestion Cleared"
        )

        st.session_state["tejas_event_log"].append({
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Event": event_name,
            "Details": (
                f"Demand: {demand:.1f} Mbps; "
                f"Capacity: {capacity:.1f} Mbps"
            ),
        })

# Store the current state for comparison on the next rerun
st.session_state["tejas_previous_signature"] = current_signature

# Keep the most recent 100 events
st.session_state["tejas_event_log"] = (
    st.session_state["tejas_event_log"][-100:]
)

event_df = pd.DataFrame(
    st.session_state["tejas_event_log"]
)

st.subheader("Recent Network Events")

if not event_df.empty:
    st.dataframe(
        event_df.iloc[::-1],
        use_container_width=True,
        hide_index=True,
    )

    event_csv = event_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Event Audit Log (CSV)",
        data=event_csv,
        file_name="tejas_network_event_audit.csv",
        mime="text/csv",
        key="download_tejas_event_log",
    )
else:
    st.info("No network events have been recorded yet.")

if st.button("Clear Event History", key="clear_tejas_event_log"):
    st.session_state["tejas_event_log"] = []
    st.session_state.pop("tejas_previous_signature", None)
    st.rerun()

st.caption(
    "This audit log records observed application-state changes "
    "during the current Streamlit session. It is not a persistent "
    "or tamper-proof production audit system."
)

# ==========================================================
# STEP 19: LIVE NETWORK HEALTH SCORE
# ==========================================================

st.divider()
st.header("💚 Step 19: Live Network Health Score")

st.write(
    "A rule-based score summarizing the current simulated "
    "hospital network conditions."
)

# Calculate individual health factors (0 to 100)
utilization_health = max(
    0.0,
    min(100.0, 100.0 - max(0.0, utilization - 70.0) * 2.0)
)

latency_health = max(
    0.0,
    min(100.0, 100.0 - max(0.0, latency - 20.0) * 0.8)
)

packet_loss_health = max(
    0.0,
    min(100.0, 100.0 - loss * 10.0)
)

# Weighted score: congestion has the greatest influence
network_health_score = round(
    0.50 * utilization_health
    + 0.30 * latency_health
    + 0.20 * packet_loss_health
)

network_health_score = int(
    max(0, min(100, network_health_score))
)

if network_health_score >= 80:
    health_status = "Healthy"
    st.success("Network Health: HEALTHY")
elif network_health_score >= 50:
    health_status = "Degraded"
    st.warning("Network Health: DEGRADED")
else:
    health_status = "Critical"
    st.error("Network Health: CRITICAL")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Network Health Score",
        f"{network_health_score}/100"
    )
    st.progress(network_health_score / 100)

with col2:
    st.metric(
        "Network Status",
        health_status
    )
    st.metric(
        "Current Utilization",
        f"{utilization:.1f}%"
    )

health_factors_df = pd.DataFrame({
    "Health Factor": [
        "Utilization",
        "Estimated Latency",
        "Estimated Packet Loss",
    ],
    "Factor Score": [
        round(utilization_health, 1),
        round(latency_health, 1),
        round(packet_loss_health, 1),
    ],
})

st.subheader("Health Factor Breakdown")

fig_health = px.bar(
    health_factors_df,
    x="Health Factor",
    y="Factor Score",
    range_y=[0, 100],
    text="Factor Score",
    title="Contribution Factors (Individual Scores)",
)

st.plotly_chart(fig_health, use_container_width=True)

st.caption(
    "The overall score is a custom weighted indicator based on "
    "simulated utilization, estimated latency, and estimated packet loss. "
    "It is not a standardized network SLA or a real-time measurement."
)
# --------------------------------------------------
# PROTOTYPE DISCLAIMER
# --------------------------------------------------
st.divider()

st.info(
    "HospitalNet AI is a simulation prototype. "
    "It does not monitor or control a real hospital "
    "network and does not guarantee medical-service "
    "performance."
)
