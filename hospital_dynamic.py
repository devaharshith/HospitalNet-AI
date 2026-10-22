
import streamlit as st
import pandas as pd

from hospital_database import (
    get_hospital_devices,
    update_device_count,
    get_application_demands,
    get_total_demand
)


def show_dynamic_hospital_dashboard():

    st.header("🏥 Dynamic Hospital Device Monitoring")

    st.info(
        "Simulation mode: Synthetic hospital device data."
    )

    devices = get_hospital_devices()

    st.subheader("Connected Medical Devices")

    with st.form("device_update_form"):

        new_counts = {}

        for device in devices:

            application = device["application"]

            new_counts[application] = st.number_input(
                f"{application} - Active Devices",
                min_value=0,
                max_value=10000,
                value=int(device["active_devices"]),
                step=1,
                key=f"count_{application}"
            )

        submitted = st.form_submit_button(
            "Update Network Demand"
        )

    if submitted:

        for application, count in new_counts.items():
            update_device_count(application, count)

        st.success("Device counts updated!")
        st.rerun()

    demands = get_application_demands()
    total_demand = get_total_demand()

    total_devices = sum(
        d["active_devices"]
        for d in get_hospital_devices()
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Active Connected Devices",
            total_devices
        )

    with col2:
        st.metric(
            "Total Bandwidth Demand",
            f"{total_demand:.1f} Mbps"
        )

    df = pd.DataFrame([
        {
            "Application": d["application"],
            "Active Devices": d["active_devices"],
            "Mbps per Device": d["bandwidth_per_device"],
            "Total Demand (Mbps)":
                demands[d["application"]]
        }
        for d in get_hospital_devices()
    ])

    st.subheader("Application Bandwidth Demand")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.bar_chart(
        df.set_index("Application")[
            "Total Demand (Mbps)"
        ]
    )

    return demands
