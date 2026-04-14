import streamlit as st
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.io as pio

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Rethinking Generalization: Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to increase sidebar width and multiselect chip visibility
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        min-width: 400px;
        max-width: 400px;
    }
    .stMultiSelect div div div div {
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Set dark theme for Plotly
pio.templates.default = "plotly_dark"

# --- HELPER FUNCTIONS ---
def load_all_histories(models_root):
    histories = {}
    models_path = Path(models_root)
    json_files = list(models_path.rglob("*_history.json"))
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            rel_path = json_file.relative_to(models_path)
            # Format name: "condition | model" for better readability
            parts = rel_path.parts
            condition = parts[0].replace('_', ' ').title()
            model_base = rel_path.stem.replace('_history', '')
            display_name = f"{condition} | {model_base}"
            
            histories[display_name] = data
        except Exception as e:
            st.error(f"Error loading {json_file}: {e}")
    return histories

# --- DATA DISCOVERY ---
current_file = Path(__file__).resolve()
# Assuming project structure: root/RETHINKING_GENERALIZATION/modeling/plots.py
project_root = current_file.parent.parent.parent
models_dir = project_root / "models"
report_dir = project_root / "reports"

all_histories = load_all_histories(models_dir)

# --- SIDEBAR ---
st.sidebar.title("🛠️ Settings")

if not all_histories:
    st.sidebar.warning("No history JSONs found in the models folder.")
    st.stop()

st.sidebar.subheader("Select Models")
selected_names = st.sidebar.multiselect(
    "Choose models to visualize",
    options=list(all_histories.keys()),
    default=list(all_histories.keys())
)

metric_option = st.sidebar.selectbox(
    "Select Metric",
    options=["train_acc", "train_loss"],
    format_func=lambda x: "Accuracy" if x == "train_acc" else "Loss"
)

st.sidebar.divider()
save_btn = st.sidebar.button("💾 Save Current View to Reports", use_container_width=True)

# --- MAIN CONTENT ---
st.title("📊 Training History Analysis")
st.markdown("""
Interactive dashboard to visualize training histories from different conditions 
(Normal vs Random Labels vs Shuffled Pixels).
""")

if not selected_names:
    st.info("Please select at least one model from the sidebar.")
else:
    # --- PLOT GENERATION ---
    fig = go.Figure()
    
    for name in selected_names:
        data = all_histories[name]
        if metric_option in data:
            y = data[metric_option]
            x = list(range(1, len(y) + 1))
            fig.add_trace(go.Scatter(
                x=x, y=y, 
                mode='lines+markers', 
                name=name,
                hovertemplate=f"<b>{name}</b><br>Epoch: %{{x}}<br>Val: %{{y:.4f}}<extra></extra>"
            ))
    
    label = "Accuracy" if metric_option == "train_acc" else "Loss"
    fig.update_layout(
        title=f"Comparison: {label}",
        xaxis_title="Epoch",
        yaxis_title=label,
        hovermode="x unified",
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics Table
    with st.expander("📝 Summary Statistics"):
        stats = []
        for name in selected_names:
            vals = all_histories[name][metric_option]
            stats.append({
                "Model": name,
                "Final Value": f"{vals[-1]:.4f}",
                "Min": f"{min(vals):.4f}",
                "Max": f"{max(vals):.4f}",
                "Epochs": len(vals)
            })
        st.table(stats)

    # --- SAVE LOGIC ---
    if save_btn:
        report_dir.mkdir(exist_ok=True)
        timestamp = "" # Could add timestamp if needed
        filename = f"web_report_{metric_option}.html"
        save_path = report_dir / filename
        fig.write_html(str(save_path))
        st.sidebar.success(f"Saved to: reports/{filename}")
        st.balloons()
