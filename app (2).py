import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="HospitalNet AI",
    page_icon="🏥",
    layout="wide"
)

# Professional HospitalNet AI Dashboard Styling
st.markdown("""
<style>
/* Main dashboard background */
.stApp {
    background-color: #0B1730;
    color: #EAF2FF;
}

/* Main content area */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1250px;
}

/* Dashboard headings */
h1, h2, h3 {
    color: #59C7FA !important;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #172A49;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #29496B;
}

/* Metric values */
[data-testid="stMetricValue"] {
    color: #FFFFFF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #101F39;
}

/* Buttons */
.stButton > button {
    background-color: #1683C5;
    color: white;
    border-radius: 8px;
    border: none;
}

/* Dataframe container */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


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

@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 3000
    capacity = rng.integers(10, 101, n)
    emergency = rng.integers(0, 2, n)

    demand = np.array([
        sum(x["demand"] * rng.uniform(.6, 1.4)
            for x in (EMERGENCY if e else NORMAL).values())
        for e in emergency
    ])

    utilization = 100 * demand / capacity
    latency = np.clip(
        15 + 1.2 * utilization + rng.normal(0, 8, n), 1, 300
    )
    loss = np.clip(
        np.maximum(0, utilization - 75) * .12
        + rng.normal(0, .8, n), 0, 30
    )

    X = pd.DataFrame({
        "Capacity_Mbps": capacity,
        "Demand_Mbps": demand,
        "Utilization_Percent": utilization,
        "Latency_ms": latency,
        "Packet_Loss_Percent": loss,
        "Emergency": emergency
    })

    y = ((demand > capacity) | (latency > 100) | (loss > 5)).astype(int)

    model = RandomForestClassifier(
        n_estimators=100, max_depth=10,
        random_state=42, class_weight="balanced"
    )
    model.fit(X, y)
    return model

model = train_model()

def allocate(profile, capacity):
    allocation = {name: 0.0 for name in profile}
    remaining = float(capacity)
    active = list(profile)

    while remaining > 1e-9 and active:
        total_weight = sum(profile[n]["weight"] for n in active)
        used = 0.0
        next_active = []

        for name in active:
            share = remaining * profile[name]["weight"] / total_weight
            grant = min(
                share, profile[name]["demand"] - allocation[name]
            )
            allocation[name] += grant
            used += grant

            if allocation[name] < profile[name]["demand"] - 1e-9:
                next_active.append(name)

        remaining -= used
        if used <= 1e-9:
            break
        active = next_active

    return allocation

st.title("🏥 HospitalNet AI")
st.markdown(
    "**Intelligent Hospital Network Management | "
    "AI-Assisted Bandwidth Allocation**"
)
st.caption(
    "TEJAS 2026 | 5G/6G Healthcare Networking Prototype"
)
st.divider()

st.caption("AI-Assisted Hospital Network Resource Management")

st.sidebar.header("Network Controls")
capacity = st.sidebar.slider(
    "Available bandwidth (Mbps)", 10, 100, 30, 5
)
emergency = st.sidebar.toggle("Activate emergency mode", value=False)

profile = {
    name: item.copy()
    for name, item in (EMERGENCY if emergency else NORMAL).items()
}

demand = sum(x["demand"] for x in profile.values())
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

# Operational simulation status based on demand and capacity
congested = demand > capacity
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
        "Allocated (Mbps)": round(allocation[name], 2),
        "Unmet Demand (Mbps)": round(
            max(0, item["demand"] - allocation[name]), 2
        )
    }
    for name, item in profile.items()
])

a, b, c, d = st.columns(4)
a.metric("Network Capacity", f"{capacity} Mbps")
b.metric("Total Demand", f"{demand} Mbps")
c.metric("Estimated Latency", f"{latency:.1f} ms")
d.metric("Estimated Packet Loss", f"{loss:.2f}%")

st.subheader("AI Congestion Prediction")
st.write("**Status:**", "⚠️ Possible congestion" if congested
         else "✅ No congestion predicted")
st.write(f"Model congestion probability: {probability:.1%}")
st.caption("Predictions use a synthetic training dataset; they are not validated clinical-network predictions.")

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

# Baseline vs Adaptive Allocation Comparison
st.subheader("Baseline vs AI-Assisted Allocation")

# Keep the same demand for a fair comparison.
# Baseline uses fixed priorities; adaptive uses the priorities above.
baseline_profile = {
    name: {
        "demand": item["demand"],
        "weight": weight
    }
    for (name, item), weight in zip(
        profile.items(), [5, 4, 3, 2, 1]
    )
}

baseline_allocation = allocate(baseline_profile, capacity)

critical_apps = ["Remote Surgery", "Patient Monitoring"]

baseline_critical = sum(
    baseline_allocation[name] for name in critical_apps
)
adaptive_critical = sum(
    allocation[name] for name in critical_apps
)

baseline_unmet = sum(
    max(0, item["demand"] - baseline_allocation[name])
    for name, item in baseline_profile.items()
)
adaptive_unmet = sum(
    max(0, item["demand"] - allocation[name])
    for name, item in profile.items()
)

comparison = pd.DataFrame([
    {
        "Method": "Baseline",
        "Critical Apps Bandwidth (Mbps)": round(baseline_critical, 2),
        "Total Unmet Demand (Mbps)": round(baseline_unmet, 2)
    },
    {
        "Method": "AI-Assisted",
        "Critical Apps Bandwidth (Mbps)": round(adaptive_critical, 2),
        "Total Unmet Demand (Mbps)": round(adaptive_unmet, 2)
    }
])

st.dataframe(comparison, use_container_width=True, hide_index=True)

comparison_chart = px.bar(
    comparison,
    x="Method",
    y=["Critical Apps Bandwidth (Mbps)", "Total Unmet Demand (Mbps)"],
    barmode="group",
    title="Baseline vs AI-Assisted Performance"
)

st.plotly_chart(comparison_chart, use_container_width=True)

# Per-Application Bandwidth Comparison
st.subheader("Bandwidth Allocation by Hospital Application")

application_comparison = pd.DataFrame([
    {
        "Application": name,
        "Baseline (Mbps)": round(baseline_allocation[name], 2),
        "AI-Assisted (Mbps)": round(allocation[name], 2)
    }
    for name in profile
])

application_comparison_long = application_comparison.melt(
    id_vars="Application",
    value_vars=["Baseline (Mbps)", "AI-Assisted (Mbps)"],
    var_name="Allocation Method",
    value_name="Bandwidth (Mbps)"
)

application_chart = px.bar(
    application_comparison_long,
    x="Application",
    y="Bandwidth (Mbps)",
    color="Allocation Method",
    barmode="group",
    title="Baseline vs AI-Assisted Bandwidth per Application",
    labels={
        "Application": "Hospital Application",
        "Bandwidth (Mbps)": "Allocated Bandwidth (Mbps)"
    }
)

st.plotly_chart(application_chart, use_container_width=True)

st.caption(
    "The chart compares simulated allocations under the same capacity "
    "and demand. AI-assisted allocation changes priority weights when "
    "congestion is predicted or emergency mode is enabled."
)

st.info(
    "Prototype simulation only. This website does not control a real "
    "hospital network or guarantee medical-service performance."
)
