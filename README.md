# HospitalNet AI

AI-assisted hospital network management prototype that predicts potential network congestion and simulates adaptive bandwidth allocation for critical healthcare applications.

## Features
- Random Forest congestion prediction
- Adaptive bandwidth allocation simulation
- Emergency-mode prioritization
- Interactive Streamlit dashboard
- Bandwidth demand and allocation charts

## Technology Stack
- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly

## Getting Started

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## How It Works
Users can adjust the available network bandwidth and activate emergency mode. The application predicts potential congestion and simulates bandwidth allocation based on application priorities.

## Project Status
Prototype developed for the TEJAS project. The current implementation uses synthetic data and simulated network conditions.

## Disclaimer
This is an experimental simulation, not a validated hospital network controller. It is not intended for clinical use.
