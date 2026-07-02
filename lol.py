"""
Sodium-Ion Cathode Screener
A robust data mining and 3D visualization dashboard targeting Sodium-ion battery materials.
Powered by Streamlit, Materials Project API (mp-api), pymatgen, and py3Dmol.
"""

import streamlit as st
import py3Dmol
from stmol import showmol
from mp_api.client import MPRester
from pymatgen.core import Composition
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import pandas as pd
import plotly.express as px

# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Sodium-Ion Cathode Screener",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar entirely
)

# Enforce a polished, professional Light Mode UI using custom CSS
st.markdown("""
<style>
    /* Global Theme Adjustments */
    :root {
        --bg-color: #f8f9fa;
        --text-color: #2c3e50;
        --card-bg: #ffffff;
        --border-color: #e9ecef;
        --accent-color: #3498db;
    }
    
    .stApp {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Card Styles for Metrics */
    .metric-card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 24px;
        text-align: center;
        border: 1px solid var(--border-color);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-color);
        margin-top: 8px;
    }
    .metric-label {
        font-size: 13px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #1a252f;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Initialize Session State Variables
# -----------------------------------------------------------------------------
if "selected_elements" not in st.session_state:
    st.session_state.selected_elements = ["Fe", "P", "O"]
if "search_performed" not in st.session_state:
    st.session_state.search_performed = False
if "battery_data" not in st.session_state:
    st.session_state.battery_data = []
if "comparison_basket" not in st.session_state:
    st.session_state.comparison_basket = []
if "active_material" not in st.session_state:
    st.session_state.active_material = None


# -----------------------------------------------------------------------------
# 2. Advanced ML & DFT Helper Functions
# -----------------------------------------------------------------------------
def get_atomic_mix(structure):
    """Calculates the atomic fraction breakdown of the crystal structure."""
    comp = structure.composition
    total_atoms = comp.num_atoms
    return {str(el): round(count / total_atoms, 4) for el, count in comp.items()}

def calculate_packing_fraction(structure):
    """Estimates the lattice packing fraction using atomic/ionic radii."""
    total_atomic_vol = 0.0
    cell_volume = structure.volume
    for site in structure:
        try:
            specie = site.specie
            r = getattr(specie, "atomic_radius", None)
            if r is None:
                r = getattr(specie, "covalent_radius", 1.0)
            if r is None:
                r = 1.0
            total_atomic_vol += (4.0 / 3.0) * 3.1415926535 * (r ** 3)
        except Exception:
            total_atomic_vol += (4.0 / 3.0) * 3.1415926535 * (1.0 ** 3)
    return min(total_atomic_vol / cell_volume, 1.0)

def estimate_kpoints(structure, spacing=0.04):
    """Estimates the recommended k-point grid dimensions for a DFT simulation."""
    a, b, c = structure.lattice.abc
    nk_a = max(1, round(1.0 / (a * spacing)))
    nk_b = max(1, round(1.0 / (b * spacing)))
    nk_c = max(1, round(1.0 / (c * spacing)))
    return nk_a, nk_b, nk_c

def check_magnetic_moments(structure):
    """Scans for transition metals and issues warnings regarding spin polarization."""
    magnetic_transition_metals = {"Fe", "Mn", "Co", "Ni", "V", "Cr"}
    comp_elements = {el.symbol for el in structure.composition.elements}
    found_magnetic = comp_elements.intersection(magnetic_transition_metals)
    if found_magnetic:
        st.info(
            f"⚠️ **DFT Spin Initialization Warning:** This material contains transition metals ({', '.join(found_magnetic)}) "
            f"which may have unpaired d-electrons. For DFT simulations, make sure to initialize magnetic moments."
        )

def safe_extract(doc, attribute, default=0.0):
    """Safely extracts attributes from API Document classes or fallback sub-entries."""
    val = getattr(doc, attribute, None)
    if val is None and hasattr(doc, "entries") and doc.entries:
        primary_entry = list(doc.entries.values())[0]
        val = getattr(primary_entry, attribute, None)
    return val if val is not None else default


# -----------------------------------------------------------------------------
# 3. Authentication (Secrets Management)
# -----------------------------------------------------------------------------
@st.cache_resource
def get_api_key():
    """Securely load the API key directly from Streamlit secrets."""
    try:
        return st.secrets["MP_API_KEY"]
    except KeyError:
        st.error("🚨 MP_API_KEY not found. Please add it to your `.streamlit/secrets.toml` file.")
        st.stop()

API_KEY = get_api_key()


# -----------------------------------------------------------------------------
# 4. Main App Layout: Input & Visualization Split
# -----------------------------------------------------------------------------
st.title("🔋 Sodium-Ion Cathode Screener")
st.markdown("A robust data mining and interactive 3D visualization dashboard for Na-ion battery materials.")

col_input, col_display = st.columns([2, 3], gap="large")

with col_input:
    st.markdown("### 🧬 1. Define Cathode")
    
    elements_list = [
        'H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Be', 'Mg', 'Ca', 'Sr', 'Ba',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
        'B', 'Al', 'Ga', 'In', 'C', 'Si', 'Ge', 'Sn',
        'N', 'P', 'As', 'Sb', 'O', 'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I'
    ]

    selected_elements = st.multiselect(
        "Select Framework Elements:",
        options=elements_list,
        default=st.session_state.selected_elements
    )
    
    st.markdown("**Current Selection:**")
    if selected_elements:
        st.pills(
            label="Selected", 
            options=selected_elements, 
            selection_mode="multi", 
            default=selected_elements,
            disabled=True, 
            label_visibility="collapsed"
        )
    else:
        st.info("No elements selected yet.")
        
    st.markdown("---")
    search_clicked = st.button("Search Battery Database", type="primary", use_container_width=True)
    
    # Execution Logic for Search
    if search_clicked:
        if not selected_elements:
            st.warning("Please select at least one framework element.")
        else:
            st.session_state.selected_elements = selected_elements
            with st.spinner("Mining Materials Project Database..."):
                try:
                    with MPRester(API_KEY) as mpr:
                        docs = mpr.insertion_electrodes.search(working_ion=["Na"])
                        filtered_docs = []
                        selected_set = set(selected_elements)
                        
                        for doc in docs:
                            try:
                                formula_str = safe_extract(doc, "formula_discharge", "")
                                if not formula_str:
                                    formula_str = getattr(doc, "battery_id", "")
                                if formula_str:
                                    comp = Composition(formula_str)
                                    comp_elements = {el.symbol for el in comp.elements}
                                    if selected_set.issubset(comp_elements):
                                        filtered_docs.append(doc)
                            except Exception:
                                continue
                        
                        st.session_state.battery_data = filtered_docs
                        st.session_state.search_performed = True
                        st.session_state.active_material = None # Reset active material on new search
                        st.rerun()
                except Exception as e:
                    st.error(f"API Connection Error: {e}")

with col_display:
    st.markdown("### 📈 Electrochemical Explorer")
    if not st.session_state.search_performed:
        st.info("👈 Please define your framework on the left and click **'Search'** to begin.")
    else:
        data = st.session_state.battery_data
        if not data:
            st.warning(f"No Sodium-ion capable materials found containing: {', '.join(st.session_state.selected_elements)}")
        else:
            # Build Dataframe for Plotly
            plot_df_list = []
            options_dict = {} # Map labels to documents
            
            for doc in data:
                b_id = getattr(doc, "battery_id", "Unknown")
                form = safe_extract(doc, "formula_discharge", b_id)
                v_avg = safe_extract(doc, "average_voltage", 0.0)
                c_grav = safe_extract(doc, "capacity_grav", 0.0)
                d_vol = safe_extract(doc, "max_delta_volume", 0.0)
                
                label = f"{form} - Average Voltage: {v_avg:.2f} V ({b_id})"
                options_dict[label] = doc
                
                plot_df_list.append({
                    "Label": label,
                    "Formula": form,
                    "Average Voltage (V)": float(v_avg),
                    "Gravimetric Capacity (mAh/g)": float(c_grav),
                    "Max Delta Volume (%)": float(d_vol) * 100.0,
                    "Battery ID": b_id
                })
                
            plot_df = pd.DataFrame(plot_df_list)
            
            # Default to first material if none selected
            if st.session_state.active_material not in options_dict:
                st.session_state.active_material = plot_df["Label"].iloc[0]

            # Interactive Plotly Chart
            fig = px.scatter(
                plot_df,
                x="Gravimetric Capacity (mAh/g)",
                y="Average Voltage (V)",
                color="Max Delta Volume (%)",
                color_continuous_scale="Viridis",
                hover_data={"Label": False, "Formula": True, "Battery ID": True},
                custom_data=["Label"], # Pass the precise label to CustomData for exact mapping
                template="plotly_white",
                labels={"Max Delta Volume (%)": "Max ΔV (%)"},
            )
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))
            
            st.caption("✨ **Click on any point in the plot** to analyze that specific material below.")
            plot_event = st.plotly_chart(
                fig, 
                on_select="rerun", 
                selection_mode="points",
                use_container_width=True,
                key="interactive_plot"
            )

            # Process Click Event
            if plot_event and plot_event.selection.points:
                # Extract the hidden 'Label' from customdata array
                clicked_label = plot_event.selection.points[0]["customdata"][0]
                if clicked_label != st.session_state.active_material:
                    st.session_state.active_material = clicked_label
                    st.rerun()

# -----------------------------------------------------------------------------
# 5. Full Screen Dashboard Details (Only shows after search)
# -----------------------------------------------------------------------------
if st.session_state.search_performed and st.session_state.battery_data:
    st.markdown("---")
    
    # 5a. Material Selection Sync
    options_list = list(options_dict.keys())
    current_index = options_list.index(st.session_state.active_material)
    
    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        selected_option = st.selectbox(
            "2. Selected Battery Material to Analyze:", 
            options_list,
            index=current_index
        )
        # Allow manual dropdown override
        if selected_option != st.session_state.active_material:
            st.session_state.active_material = selected_option
            st.rerun()
            
    selected_doc = options_dict[st.session_state.active_material]
            
    with col_btn:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        add_to_basket = st.button("➕ Add to Comparison", use_container_width=True, type="secondary")
        
    target_mp_id = safe_extract(selected_doc, "id_discharge", getattr(selected_doc, "battery_id", ""))
    materials_project_url = f"https://next-gen.materialsproject.org/materials/{target_mp_id}"
    
    st.markdown(f"""
    <div class="metric-card" style="border-left: 5px solid #3B82F6; margin-top: 10px; padding: 12px 20px; text-align: left;">
        <div class="metric-label" style="font-weight: bold; margin-bottom: 2px;">External Database Verification</div>
        <a href="{materials_project_url}" target="_blank" style="text-decoration: none; color: #3B82F6; font-weight: 600; font-size: 0.95rem;">
            🔗 View full thermodynamic & electronic profile for {target_mp_id} on Materials Project ↗
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    # 5b. Fetching Crystal Structure Data
    current_view_state = st.session_state.get("view_state_toggle", "Discharged (Sodiated)")
    target_struct_id = safe_extract(selected_doc, "id_charge", "") if current_view_state == "Charged (Desodiated)" else safe_extract(selected_doc, "id_discharge", "")
    if not target_struct_id:
        target_struct_id = getattr(selected_doc, "battery_id", "")

    structure = None
    crys_system = "Unknown"
    sg_symbol = "Unknown"
    lattice_html = "<div class='metric-value' style='font-size: 18px;'>N/A</div>"
    angles_html = "<div class='metric-value' style='font-size: 18px;'>N/A</div>"
    
    with st.spinner("Fetching Crystal Structure & Symmetry Data..."):
        try:
            with MPRester(API_KEY) as mpr:
                structure = mpr.get_structure_by_material_id(target_struct_id)
                if structure:
                    sga = SpacegroupAnalyzer(structure)
                    crys_system = sga.get_crystal_system().capitalize()
                    sg_symbol = sga.get_space_group_symbol()
                    lattice = structure.lattice
                    lattice_html = f"""
                    <div class="metric-value" style="font-size: 18px;">
                        a = {lattice.a:.3f}, b = {lattice.b:.3f}, c = {lattice.c:.3f}
                    </div>
                    """
                    angles_html = f"""
                    <div class="metric-value" style="font-size: 18px;">
                        α = {lattice.alpha:.2f}, β = {lattice.beta:.2f}, γ = {lattice.gamma:.2f}
                    </div>
                    """
        except Exception as e:
            st.error(f"Could not automatically resolve structural unit cell metadata for ID {target_struct_id}: {e}")

    # 5c. Basket Logic
    avg_voltage = float(safe_extract(selected_doc, "average_voltage", 0.0))
    mdv = safe_extract(selected_doc, "max_delta_volume", None)
    capacity = safe_extract(selected_doc, "capacity_grav", None)
    
    if add_to_basket:
        basket_item = {
            "Battery ID": getattr(selected_doc, "battery_id", "Unknown"),
            "Formula": safe_extract(selected_doc, "formula_discharge", "Unknown"),
            "Average Voltage (V)": round(avg_voltage, 2),
            "Max Delta Volume (%)": f"{float(mdv) * 100:.2f}%" if mdv is not None else "N/A",
            "Crystal System": crys_system
        }
        if basket_item["Battery ID"] not in [item["Battery ID"] for item in st.session_state.comparison_basket]:
            st.session_state.comparison_basket.append(basket_item)
            st.toast(f"Added {basket_item['Formula']} to comparison basket!")
        else:
            st.toast(f"{basket_item['Formula']} is already in your comparison basket.")

    # 5d. Tabbed Views
    tab1, tab2, tab3 = st.tabs(["📊 Main Dashboard", "🤖 ML & DFT Pipeline", "🛒 Comparison Basket"])
    
    with tab1:
        st.markdown("### 📊 DFT Structural & Electrochemical Parameters")
        col_metrics1, col_metrics2 = st.columns(2)
        
        mdv_text = f"{float(mdv) * 100:.2f}%" if mdv is not None else "N/A"
        capacity_text = f"{float(capacity):.2f}" if capacity is not None else "N/A"
        
        with col_metrics1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Average Voltage (V)</div>
                <div class="metric-value">{avg_voltage:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Max Delta Volume</div>
                <div class="metric-value">{mdv_text}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Gravimetric Capacity (mAh/g)</div>
                <div class="metric-value">{capacity_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_metrics2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Crystal System & Space Group</div>
                <div class="metric-value">{crys_system} ({sg_symbol})</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Lattice Parameters (Å)</div>
                {lattice_html}
            </div>
            <div class="metric-card">
                <div class="metric-label">Lattice Angles (°)</div>
                {angles_html}
            </div>
            """, unsafe_allow_html=True)
            
        if structure:
            st.markdown("---")
            st.markdown("### ⚛️ Interactive 3D Crystal Structure")
            
            st.radio(
                "View State: Discharged (Sodiated) vs. Charged (Desodiated)",
                options=["Discharged (Sodiated)", "Charged (Desodiated)"],
                key="view_state_toggle",
                horizontal=True
            )
            
            cif_writer = CifWriter(structure)
            cif_str = str(cif_writer)
            
            view = py3Dmol.view(width=800, height=500)
            view.addModel(cif_str, 'cif')
            view.setStyle({'sphere': {'scale': 0.3}, 'stick': {'radius': 0.15}})
            view.addUnitCell()
            view.zoomTo()
            
            showmol(view, height=500, width=800)
            st.markdown("<br>", unsafe_allow_html=True)
            
            formula_name = safe_extract(selected_doc, "formula_discharge", "Material")
            file_name = f"{formula_name}_{getattr(selected_doc, 'battery_id', 'struct')}.cif"
            
            st.download_button(
                label="⬇️ Download .cif File for QuantumATK",
                data=cif_str,
                file_name=file_name,
                mime="text/plain",
                use_container_width=True,
                type="primary"
            )

    with tab2:
        if structure:
            st.markdown("### 🤖 Materials Informatics Featurizer")
            
            density = structure.density
            f_capacity = float(capacity) if capacity is not None else 0.0
            vol_capacity = (f_capacity * density)
            vol_energy_density = (vol_capacity / 1000.0) * avg_voltage if capacity is not None else 0.0
            packing_fraction = calculate_packing_fraction(structure)
            atomic_mix = get_atomic_mix(structure)
            
            ml_records = [
                {"Property": "Crystal Density (g/cm³)", "Value": f"{density:.3f}"},
                {"Property": "Calculated Volumetric Capacity (mAh/cm³)", "Value": f"{vol_capacity:.2f}" if capacity is not None else "N/A"},
                {"Property": "Volumetric Energy Density (Wh/cm³)", "Value": f"{vol_energy_density:.4f}" if capacity is not None else "N/A"},
                {"Property": "Lattice Packing Fraction", "Value": f"{packing_fraction:.2%}"}
            ]
            
            st.dataframe(pd.DataFrame(ml_records), use_container_width=True, hide_index=True)
            
            st.markdown("#### Atomic Mix Breakdown (Fractional)")
            st.json(atomic_mix)
            
            st.markdown("---")
            st.markdown("### 🔬 DFT & QuantumATK Pre-Processor Setup")
            
            nk_a, nk_b, nk_c = estimate_kpoints(structure, spacing=0.04)
            st.markdown(f"**Suggested Reciprocal k-point Grid:** `{nk_a} × {nk_b} × {nk_c}` (aiming for reciprocal spacing of ~0.04 Å⁻¹)")
            
            check_magnetic_moments(structure)
        else:
            st.info("Please select a material on the Main Dashboard to enable ML/DFT calculations.")

    with tab3:
        st.markdown("### 🛒 Materials Comparison Basket")
        
        if not st.session_state.comparison_basket:
            st.info("Your comparison basket is empty. Add materials using the '➕ Add to Comparison' button above.")
        else:
            comparison_df = pd.DataFrame(st.session_state.comparison_basket)
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
            
            if st.button("🗑️ Clear Basket", type="secondary", use_container_width=True):
                st.session_state.comparison_basket = []
                st.toast("Comparison basket cleared!")
                st.rerun()