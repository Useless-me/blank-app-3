import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Page configuration
st.set_page_config(page_title="Composite Material Analysis", page_icon="📊", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    .main {padding: 2rem;}
    .stButton>button {width: 100%;}
    .stDownloadButton>button {width: 100%;}
    .stTextInput>div>div>input {text-align: center;}
    .stNumberInput>div>div>input {text-align: center;}
    .reportview-container .main .block-container {padding-top: 2rem;}
    h1 {color: #2a5c9a;}
    h2 {color: #3a7ca5;}
    .sidebar .sidebar-content {background-color: #f8f9fa;}
    .matrix-table {font-size: 0.85rem;}
    .stDataFrame {width: 100%;}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("Composite Material Analysis Tool")
st.markdown("""
This application performs various composite material analyses including:
•⁠  ⁠Micro-mechanics of composites
•⁠  ⁠Stiffness and compliance matrices
•⁠  ⁠Laminate analysis
•⁠  ⁠Failure prediction
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Select Analysis Mode", [
    "Material Properties",
    "Single Ply Analysis",
    "Laminate Analysis",
    "Failure Prediction"
])

# Initialize session state for storing data
if 'laminate_layers' not in st.session_state:
    st.session_state.laminate_layers = []

if 'composite_props' not in st.session_state:
    st.session_state.composite_props = None

# Helper functions (keep all the existing helper functions here)
def calculate_Q_matrix(E1, E2, nu12, G12):
    """Calculate stiffness matrix Q"""
    nu21 = nu12 * E2 / E1
    Q = np.array([
        [E1/(1-nu12*nu21), nu12*E2/(1-nu12*nu21), 0],
        [nu12*E2/(1-nu12*nu21), E2/(1-nu12*nu21), 0],
        [0, 0, G12]
    ])
    return Q

def calculate_S_matrix(E1, E2, nu12, G12):
    """Calculate compliance matrix S"""
    nu21 = nu12 * E2 / E1
    S = np.array([
        [1/E1, -nu21/E2, 0],
        [-nu12/E1, 1/E2, 0],
        [0, 0, 1/G12]
    ])
    return S

def calculate_Qbar_matrix(Q, angle):
    """Calculate transformed stiffness matrix Qbar"""
    theta = np.radians(angle)
    m = np.cos(theta)
    n = np.sin(theta)
    
    T = np.array([
        [m*2, n*2, 2*m*n],
        [n*2, m*2, -2*m*n],
        [-m*n, m*n, m*2-n*2]
    ])
    
    Tinv = np.array([
        [m*2, n*2, -2*m*n],
        [n*2, m*2, 2*m*n],
        [m*n, -m*n, m*2-n*2]
    ])
    
    Qbar = np.linalg.inv(T).dot(Q).dot(Tinv)
    return Qbar

def display_matrix(matrix, row_labels, col_labels, title):
    """Helper function to display matrices with custom formatting"""
    df = pd.DataFrame(matrix, columns=col_labels, index=row_labels)
    st.markdown(f"*{title}*")
    st.dataframe(df.style.format("{:.4e}"), use_container_width=True)

# Material Properties Module - Updated to include both options
def material_properties_module():
    st.header("1. Material Properties")
    
    input_method = st.radio("Select Input Method:", 
                           ("Fiber and Matrix Properties", "Direct Composite Properties"))
    
    if input_method == "Fiber and Matrix Properties":
        st.subheader("Input Fiber and Matrix Properties")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("*Fiber Properties*")
            Ef = st.number_input("Fiber Young's Modulus (GPa)", value=230.0, min_value=0.1)
            nu_f = st.number_input("Fiber Poisson's Ratio", value=0.2, min_value=0.0, max_value=0.5, step=0.01)
            Gf = st.number_input("Fiber Shear Modulus (GPa)", value=90.0, min_value=0.1)
        
        with col2:
            st.markdown("*Matrix Properties*")
            Em = st.number_input("Matrix Young's Modulus (GPa)", value=3.5, min_value=0.1)
            nu_m = st.number_input("Matrix Poisson's Ratio", value=0.35, min_value=0.0, max_value=0.5, step=0.01)
            Gm = st.number_input("Matrix Shear Modulus (GPa)", value=1.3, min_value=0.1)
        
        Vf = st.slider("Fiber Volume Fraction", 0.0, 1.0, 0.6, 0.01)
        
        if st.button("Calculate Composite Properties"):
            # Rule of mixtures calculations
            Vm = 1 - Vf
            E1 = Ef*Vf + Em*Vm
            E2 = (Ef*Em)/(Ef*Vm + Em*Vf)
            G12 = (Gf*Gm)/(Gf*Vm + Gm*Vf)
            nu12 = nu_f*Vf + nu_m*Vm
            nu21 = nu12 * E2 / E1
            
            # Display results
            st.subheader("Composite Properties")
            results = {
                "Property": ["E₁ (Longitudinal Modulus)", "E₂ (Transverse Modulus)", 
                            "G₁₂ (In-plane Shear Modulus)", "ν₁₂ (Major Poisson's Ratio)", 
                            "ν₂₁ (Minor Poisson's Ratio)"],
                "Value": [f"{E1:.2f} GPa", f"{E2:.2f} GPa", f"{G12:.2f} GPa", 
                         f"{nu12:.4f}", f"{nu21:.4f}"],
                "Formula": ["E₁ = Ef*Vf + Em*Vm", "E₂ = (Ef*Em)/(Ef*Vm + Em*Vf)", 
                           "G₁₂ = (Gf*Gm)/(Gf*Vm + Gm*Vf)", "ν₁₂ = νf*Vf + νm*Vm", 
                           "ν₂₁ = ν₁₂ * E₂/E₁"]
            }
            
            df = pd.DataFrame(results)
            st.table(df)
            
            # Store results in session state
            st.session_state.composite_props = {
                'E1': E1, 'E2': E2, 'G12': G12, 'nu12': nu12, 'nu21': nu21
            }
    
    else:  # Direct Composite Properties
        st.subheader("Input Composite Properties Directly")
        
        col1, col2 = st.columns(2)
        
        with col1:
            E1 = st.number_input("Longitudinal Modulus E₁ (GPa)", value=138.0, min_value=0.1)
            E2 = st.number_input("Transverse Modulus E₂ (GPa)", value=9.0, min_value=0.1)
        
        with col2:
            G12 = st.number_input("In-plane Shear Modulus G₁₂ (GPa)", value=4.8, min_value=0.1)
            nu12 = st.number_input("Major Poisson's Ratio ν₁₂", value=0.3, min_value=0.0, max_value=0.5, step=0.01)
        
        if st.button("Set Composite Properties"):
            nu21 = nu12 * E2 / E1
            
            # Display results
            st.subheader("Composite Properties")
            results = {
                "Property": ["E₁ (Longitudinal Modulus)", "E₂ (Transverse Modulus)", 
                            "G₁₂ (In-plane Shear Modulus)", "ν₁₂ (Major Poisson's Ratio)", 
                            "ν₂₁ (Minor Poisson's Ratio)"],
                "Value": [f"{E1:.2f} GPa", f"{E2:.2f} GPa", f"{G12:.2f} GPa", 
                         f"{nu12:.4f}", f"{nu21:.4f}"]
            }
            
            df = pd.DataFrame(results)
            st.table(df)
            
            # Store results in session state
            st.session_state.composite_props = {
                'E1': E1, 'E2': E2, 'G12': G12, 'nu12': nu12, 'nu21': nu21
            }
    
    # Display current properties if they exist
    if st.session_state.composite_props is not None:
        st.subheader("Current Composite Properties")
        props = st.session_state.composite_props
        current_props = pd.DataFrame({
            "Property": ["E₁", "E₂", "G₁₂", "ν₁₂", "ν₂₁"],
            "Value": [f"{props['E1']} GPa", f"{props['E2']} GPa", 
                     f"{props['G12']} GPa", props['nu12'], props['nu21']]
        })
        st.table(current_props)

# Keep all the other modules (Single Ply Analysis, Laminate Analysis, Failure Prediction) the same as before

# Single Ply Analysis Module (no changes needed)
def single_ply_analysis_module():
    st.header("2. Single Ply Analysis")
    
    if 'composite_props' not in st.session_state or st.session_state.composite_props is None:
        st.warning("Please calculate composite properties first in the Material Properties section.")
        return
    
    # Get properties from session state
    props = st.session_state.composite_props
    E1, E2, G12, nu12 = props['E1'], props['E2'], props['G12'], props['nu12']
    
    st.subheader("Stiffness and Compliance Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Input Parameters*")
        angle = st.number_input("Ply Orientation Angle (degrees)", -180, 180, 0)
        sigma_xx = st.number_input("Applied Stress σₓₓ (MPa)", value=100.0)
        sigma_yy = st.number_input("Applied Stress σᵧᵧ (MPa)", value=0.0)
        tau_xy = st.number_input("Applied Shear Stress τₓᵧ (MPa)", value=0.0)
    
    # Calculate Q and S matrices
    Q = calculate_Q_matrix(E1, E2, nu12, G12)
    S = calculate_S_matrix(E1, E2, nu12, G12)
    Qbar = calculate_Qbar_matrix(Q, angle)
    Sbar = np.linalg.inv(Qbar)
    
    st.subheader("Stiffness and Compliance Matrices")
    
    col3, col4 = st.columns(2)
    
    with col3:
        display_matrix(Q, ['ε₁', 'ε₂', 'γ₁₂'], ['σ₁', 'σ₂', 'τ₁₂'], "Q Matrix (Stiffness)")
        display_matrix(Qbar, ['εₓ', 'εᵧ', 'γₓᵧ'], ['σₓ', 'σᵧ', 'τₓᵧ'], "Q-bar Matrix (Transformed Stiffness)")
    
    with col4:
        display_matrix(S, ['σ₁', 'σ₂', 'τ₁₂'], ['ε₁', 'ε₂', 'γ₁₂'], "S Matrix (Compliance)")
        display_matrix(Sbar, ['σₓ', 'σᵧ', 'τₓᵧ'], ['εₓ', 'εᵧ', 'γₓᵧ'], "S-bar Matrix (Transformed Compliance)")
    # Calculate modulus in given direction
    Ex = 1/Sbar[0,0]
    Ey = 1/Sbar[1,1]
    Gxy = 1/Sbar[2,2]
    nuxy = -Sbar[0,1]/Sbar[0,0]
    
    st.subheader("Effective Properties in Loading Direction")
    mod_results = {
        "Property": ["Ex", "Ey", "Gxy", "νxy"],
        "Value": [f"{Ex:.2f} GPa", f"{Ey:.2f} GPa", f"{Gxy:.2f} GPa", f"{nuxy:.4f}"]
    }
    st.table(pd.DataFrame(mod_results))
    
    # Stress/strain transformations
    st.subheader("Stress/Strain Transformations")
    
    # Transform applied stresses to material coordinates
    theta = np.radians(angle)
    m = np.cos(theta)
    n = np.sin(theta)
    
    T = np.array([
        [m*2, n*2, 2*m*n],
        [n*2, m*2, -2*m*n],
        [-m*n, m*n, m*2-n*2]
    ])
    
    stress_xy = np.array([sigma_xx, sigma_yy, tau_xy])
    stress_12 = T.dot(stress_xy)
    
    # Calculate strains in x-y coordinates
    strain_xy = Sbar.dot(stress_xy)
    
    # Calculate strains in material coordinates
    strain_12 = S.dot(stress_12)
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("*Stresses*")
        stress_df = pd.DataFrame({
            "Coordinate System": ["x-y", "1-2 (material)"],
            "σ₁/σₓ (MPa)": [stress_xy[0], stress_12[0]],
            "σ₂/σᵧ (MPa)": [stress_xy[1], stress_12[1]],
            "τ₁₂/τₓᵧ (MPa)": [stress_xy[2], stress_12[2]]
        })
        numeric_cols = ["σ₁/σₓ (MPa)", "σ₂/σᵧ (MPa)", "τ₁₂/τₓᵧ (MPa)"]
        st.table(stress_df.style.format({col: "{:.2f}" for col in numeric_cols}))
    
    with col6:
        st.markdown("*Strains*")
        strain_df = pd.DataFrame({
            "Coordinate System": ["x-y", "1-2 (material)"],
            "ε₁/εₓ (μɛ)": [strain_xy[0]*1e6, strain_12[0]*1e6],
            "ε₂/εᵧ (μɛ)": [strain_xy[1]*1e6, strain_12[1]*1e6],
            "γ₁₂/γₓᵧ (μɛ)": [strain_xy[2]*1e6, strain_12[2]*1e6]
        })
        numeric_cols = ["ε₁/εₓ (μɛ)", "ε₂/εᵧ (μɛ)", "γ₁₂/γₓᵧ (μɛ)"]
        st.table(strain_df.style.format({col: "{:.2f}" for col in numeric_cols}))

# Laminate Analysis Module (no changes needed)
def laminate_analysis_module():
    st.header("3. Laminate Analysis")
    
    if 'composite_props' not in st.session_state or st.session_state.composite_props is None:
        st.warning("Please calculate composite properties first in the Material Properties section.")
        return
    
    props = st.session_state.composite_props
    E1, E2, G12, nu12 = props['E1'], props['E2'], props['G12'], props['nu12']
    
    st.subheader("Laminate Definition")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("*Add New Ply*")
        angle = st.number_input("Ply Angle (degrees)", -180, 180, 0, key="ply_angle")
        thickness = st.number_input("Ply Thickness (mm)", 0.001, 10.0, 0.125, 0.001, key="ply_thickness")
        
        if st.button("Add Ply to Laminate"):
            if 'laminate_layers' not in st.session_state:
                st.session_state.laminate_layers = []
            
            st.session_state.laminate_layers.append({
                'angle': angle,
                'thickness': thickness
            })
    
    with col2:
        st.markdown("*Current Laminate Stack*")
        if not st.session_state.laminate_layers:
            st.info("No plies added yet")
        else:
            laminate_df = pd.DataFrame(st.session_state.laminate_layers)
            st.dataframe(laminate_df)
            
            if st.button("Clear Laminate"):
                st.session_state.laminate_layers = []
                st.experimental_rerun()
    
    if not st.session_state.laminate_layers:
        return
    
    st.subheader("Laminate Analysis")
    
    # Calculate ABD matrices
    Q = calculate_Q_matrix(E1, E2, nu12, G12)
    
    # Calculate total thickness and ply positions
    total_thickness = sum(ply['thickness'] for ply in st.session_state.laminate_layers)
    z_locations = [ -total_thickness/2 ]  # Start at bottom
    
    for ply in st.session_state.laminate_layers:
        z_locations.append(z_locations[-1] + ply['thickness'])
    
    # Initialize ABD matrices
    A = np.zeros((3,3))
    B = np.zeros((3,3))
    D = np.zeros((3,3))
    
    for i, ply in enumerate(st.session_state.laminate_layers):
        angle = ply['angle']
        thickness = ply['thickness']
        z_bottom = z_locations[i]
        z_top = z_locations[i+1]
        z_mid = (z_bottom + z_top)/2
        
        Qbar = calculate_Qbar_matrix(Q, angle)
        
        # Add to ABD matrices
        A += Qbar * thickness
        B += Qbar * thickness * z_mid
        D += Qbar * (thickness * z_mid*2 + thickness*3 / 12)
    
    ABD = np.vstack([
        np.hstack([A, B]),
        np.hstack([B, D])
    ])
    
    ABD_inv = np.linalg.inv(ABD)
    
    # Display ABD matrices
    st.subheader("ABD Matrices")
    
    col3, col4 = st.columns(2)
    
    with col3:
        display_matrix(A, ['εx', 'εy', 'γxy'], ['Nx', 'Ny', 'Nxy'], "A Matrix (Extensional Stiffness)")
        display_matrix(B, ['εx', 'εy', 'γxy'], ['Mx', 'My', 'Mxy'], "B Matrix (Coupling Stiffness)")
    
    with col4:
        display_matrix(D, ['κx', 'κy', 'κxy'], ['Mx', 'My', 'Mxy'], "D Matrix (Bending Stiffness)")
        
    
    # Display ABD matrix
    st.subheader("ABD Matrix")
    display_matrix(ABD, 
                      ['εx', 'εy', 'γxy', 'κx', 'κy', 'κxy'], 
                      ['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy'], 
                      "ABD Matrix"
                      )
    # Display ABD inverse matrix
    st.subheader("ABD Inverse Matrix")
    display_matrix(ABD_inv,
                  ['Nx', 'Ny', 'Nxy', 'Mx', 'My', 'Mxy'],
                  ['εx', 'εy', 'γxy', 'κx', 'κy', 'κxy'],
                  "ABD Inverse Matrix"
                  )

    # Calculate apparent laminate stiffness coefficients
    h = total_thickness
    A_star = A / h  # Normalized extensional stiffness matrix
    B_star = B / (h**2)  # Normalized coupling stiffness matrix
    D_star = D / (h**3)  # Normalized bending stiffness matrix
    
    # st.subheader("Normalized Laminate Stiffness Coefficients")
    
    # col_norm1, col_norm2, col_norm3 = st.columns(3)
    
    # with col_norm1:
    #     display_matrix(A_star, 
    #                   ['εx', 'εy', 'γxy'], 
    #                   ['Nx/h', 'Ny/h', 'Nxy/h'], 
    #                   "A* Matrix (Normalized Extensional Stiffness)")
    
    # with col_norm2:
    #     display_matrix(B_star, 
    #                   ['εx', 'εy', 'γxy'], 
    #                   ['Mx/h²', 'My/h²', 'Mxy/h²'], 
    #                   "B* Matrix (Normalized Coupling Stiffness)")
    
    # with col_norm3:
    #     display_matrix(D_star, 
    #                   ['κx', 'κy', 'κxy'], 
    #                   ['Mx/h³', 'My/h³', 'Mxy/h³'], 
    #                   "D* Matrix (Normalized Bending Stiffness)")
    
    # Calculate apparent engineering constants
    st.subheader("Apparent Laminate Engineering Constants")
    
    # Extensional stiffness terms
    Ex_avg = 1/(h * ABD_inv[0,0])
    Ey_avg = 1/(h * ABD_inv[1,1])
    Gxy_avg = 1/(h * ABD_inv[2,2])
    nuxy_avg = -ABD_inv[0,1]/ABD_inv[0,0]
    nuyx_avg = -ABD_inv[1,0]/ABD_inv[1,1]
    
    # Coupling stiffness terms
    eta_x = ABD_inv[0,3]/ABD_inv[0,0]  # Extension-twist coupling
    eta_y = ABD_inv[1,3]/ABD_inv[1,1]  # Extension-twist coupling
    zeta_x = ABD_inv[0,4]/ABD_inv[0,0]  # Extension-bending coupling
    zeta_y = ABD_inv[1,5]/ABD_inv[1,1]  # Extension-bending coupling
    
    # Bending stiffness terms
    Dx_avg = 1/(h**3 * ABD_inv[3,3])
    Dy_avg = 1/(h**3 * ABD_inv[4,4])
    Dxy_avg = 1/(h**3 * ABD_inv[5,5])
    nuxy_bend = -ABD_inv[3,4]/ABD_inv[3,3]
    nuyx_bend = -ABD_inv[4,3]/ABD_inv[4,4]
    
    # Display results in tables
    st.markdown("*In-Plane Stiffness Properties*")
    in_plane_results = pd.DataFrame({
        "Property": ["Ex_avg", "Ey_avg", "Gxy_avg", "νxy_avg", "νyx_avg"],
        "Value": [f"{Ex_avg:.2f} GPa", f"{Ey_avg:.2f} GPa", f"{Gxy_avg:.2f} GPa", 
                 f"{nuxy_avg:.4f}", f"{nuyx_avg:.4f}"]
    })
    st.table(in_plane_results)
    
    st.markdown("*Coupling Stiffness Properties*")
    coupling_results = pd.DataFrame({
        "Property": ["ηx (Extension-Twist)", "ηy (Extension-Twist)", 
                    "ζx (Extension-Bending)", "ζy (Extension-Bending)"],
        "Value": [f"{eta_x:.4f} m⁻¹", f"{eta_y:.4f} m⁻¹", 
                 f"{zeta_x:.4f} m⁻¹", f"{zeta_y:.4f} m⁻¹"]
    })
    st.table(coupling_results)
    
    # st.markdown("*Bending Stiffness Properties*")
    # bending_results = pd.DataFrame({
    #     "Property": ["Dx_avg", "Dy_avg", "Dxy_avg", "νxy_bend", "νyx_bend"],
    #     "Value": [f"{Dx_avg:.2f} N·m", f"{Dy_avg:.2f} N·m", f"{Dxy_avg:.2f} N·m", 
    #              f"{nuxy_bend:.4f}", f"{nuyx_bend:.4f}"]
    # })
    # st.table(bending_results)
    
    st.subheader("Load Application and Response")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.markdown("*Mechanical Loads*")
        Nx = st.number_input("Nx (N/mm)", value=0.0)
        Ny = st.number_input("Ny (N/mm)", value=0.0)
        Nxy = st.number_input("Nxy (N/mm)", value=0.0)
    
    with col6:
        st.markdown("*Moment Loads*")
        Mx = st.number_input("Mx (N)", value=0.0)
        My = st.number_input("My (N)", value=0.0)
        Mxy = st.number_input("Mxy (N)", value=0.0)
    
    if st.button("Calculate Laminate Response"):
        forces_moments = np.array([Nx, Ny, Nxy, Mx, My, Mxy])
        result = ABD_inv.dot(forces_moments)
        
        epsilon0 = result[:3]
        kappa = result[3:]
        
        st.subheader("Mid-Plane Strains and Curvatures")
        midplane_results = {
            "Parameter": ["εx", "εy", "γxy", "κx", "κy", "κxy"],
            "Value": [f"{epsilon0[0]:.6f}", f"{epsilon0[1]:.6f}", f"{epsilon0[2]:.6f}", 
                     f"{kappa[0]:.6f}", f"{kappa[1]:.6f}", f"{kappa[2]:.6f}"]
        }
        st.table(pd.DataFrame(midplane_results))
        
        # Calculate ply stresses and strains
        st.subheader("Ply-by-Ply Analysis (Bottom to Top)")
        ply_results = []
        for i, ply in enumerate(st.session_state.laminate_layers):
            angle = ply['angle']
            z_bottom = z_locations[i]
            z_top = z_locations[i+1]
            
            Qbar = calculate_Qbar_matrix(Q, angle)
            
            # Calculate transformation matrix
            theta = np.radians(angle)
            m = np.cos(theta)
            n = np.sin(theta)
            
            T = np.array([
                [m*2, n*2, 2*m*n],
                [n*2, m*2, -2*m*n],
                [-m*n, m*n, m*2-n*2]
            ])
            
            # Calculate at three points through thickness
            for z_pos, location in zip(
                [z_bottom, (z_bottom+z_top)/2, z_top],
                ['Bottom', 'Middle', 'Top']
            ):
                # Global strains (x-y)
                strain_xy = epsilon0 + z_pos * kappa
                
                # Global stresses (x-y)
                stress_xy = Qbar.dot(strain_xy)
                
                # Material coordinates (1-2)
                stress_12 = T.dot(stress_xy)
                strain_12 = np.linalg.inv(Q).dot(stress_12)
                
                ply_results.append({
                    'Ply': i+1,
                    'Angle': f"{angle}°",
                    'Location': location,
                    'z (mm)': f"{z_pos:.3f}",
                    'εₓ (μɛ)': f"{strain_xy[0]*1e6:.2f}",
                    'εᵧ (μɛ)': f"{strain_xy[1]*1e6:.2f}",
                    'γₓᵧ (μrad)': f"{strain_xy[2]*1e6:.2f}",
                    'ε₁ (μɛ)': f"{strain_12[0]*1e6:.2f}",
                    'ε₂ (μɛ)': f"{strain_12[1]*1e6:.2f}",
                    'γ₁₂ (μrad)': f"{strain_12[2]*1e6:.2f}",
                    'σₓ (MPa)': f"{stress_xy[0]:.2f}",
                    'σᵧ (MPa)': f"{stress_xy[1]:.2f}",
                    'τₓᵧ (MPa)': f"{stress_xy[2]:.2f}",
                    'σ₁ (MPa)': f"{stress_12[0]:.2f}",
                    'σ₂ (MPa)': f"{stress_12[1]:.2f}",
                    'τ₁₂ (MPa)': f"{stress_12[2]:.2f}"
                })
        
        # Display results in tabs
        tab1, tab2, tab3 = st.tabs(["All Data", "Strains", "Stresses"])
        
        with tab1:
            st.dataframe(pd.DataFrame(ply_results))
        
        with tab2:
            strain_cols = ['Ply', 'Angle', 'Location', 'z (mm)',
                         'εₓ (μɛ)', 'εᵧ (μɛ)', 'γₓᵧ (μrad)',
                         'ε₁ (μɛ)', 'ε₂ (μɛ)', 'γ₁₂ (μrad)']
            st.dataframe(pd.DataFrame(ply_results)[strain_cols])
        
        with tab3:
            stress_cols = ['Ply', 'Angle', 'Location', 'z (mm)',
                         'σₓ (MPa)', 'σᵧ (MPa)', 'τₓᵧ (MPa)',
                         'σ₁ (MPa)', 'σ₂ (MPa)', 'τ₁₂ (MPa)']
            st.dataframe(pd.DataFrame(ply_results)[stress_cols])


# Failure Prediction Module (no changes needed)
def failure_prediction_module():
    st.header("4. Failure Prediction")
    
    if 'composite_props' not in st.session_state or st.session_state.composite_props is None or 'laminate_layers' not in st.session_state:
        st.warning("Please define material properties and laminate first.")
        return
    
    props = st.session_state.composite_props
    E1, E2, G12, nu12 = props['E1'], props['E2'], props['G12'], props['nu12']
    
    st.subheader("Material Strength Properties")
    
    col1, col2 = st.columns(2)
    
    with col1:
        Xt = st.number_input("Longitudinal Tensile Strength (MPa)", value=1500.0)
        Xc = st.number_input("Longitudinal Compressive Strength (MPa)", value=1200.0)
    
    with col2:
        Yt = st.number_input("Transverse Tensile Strength (MPa)", value=50.0)
        Yc = st.number_input("Transverse Compressive Strength (MPa)", value=200.0)
        S = st.number_input("In-plane Shear Strength (MPa)", value=70.0)
    
    st.subheader("Load Application")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("*Mechanical Loads*")
        Nx = st.number_input("Applied Nx (N/mm)", value=0.0, key="fail_Nx")
        Ny = st.number_input("Applied Ny (N/mm)", value=0.0, key="fail_Ny")
        Nxy = st.number_input("Applied Nxy (N/mm)", value=0.0, key="fail_Nxy")
    
    with col4:
        st.markdown("*Moment Loads*")
        Mx = st.number_input("Applied Mx (N)", value=0.0, key="fail_Mx")
        My = st.number_input("Applied My (N)", value=0.0, key="fail_My")
        Mxy = st.number_input("Applied Mxy (N)", value=0.0, key="fail_Mxy")
    
    if st.button("Predict Failure"):
        # First calculate ABD matrices
        Q = calculate_Q_matrix(E1, E2, nu12, G12)
        
        # Calculate total thickness and ply positions
        total_thickness = sum(ply['thickness'] for ply in st.session_state.laminate_layers)
        z_locations = [ -total_thickness/2 ]  # Start at bottom
        
        for ply in st.session_state.laminate_layers:
            z_locations.append(z_locations[-1] + ply['thickness'])
        
        # Initialize ABD matrices
        A = np.zeros((3,3))
        B = np.zeros((3,3))
        D = np.zeros((3,3))
        
        for i, ply in enumerate(st.session_state.laminate_layers):
            angle = ply['angle']
            thickness = ply['thickness']
            z_bottom = z_locations[i]
            z_top = z_locations[i+1]
            z_mid = (z_bottom + z_top)/2
            
            Qbar = calculate_Qbar_matrix(Q, angle)
            
            A += Qbar * thickness
            B += Qbar * thickness * z_mid
            D += Qbar * (thickness * z_mid*2 + thickness*3 / 12)
        
        ABD = np.vstack([
            np.hstack([A, B]),
            np.hstack([B, D])
        ])
        
        ABD_inv = np.linalg.inv(ABD)
        
        # Calculate mid-plane strains and curvatures
        forces_moments = np.array([Nx, Ny, Nxy, Mx, My, Mxy])
        result = ABD_inv.dot(forces_moments)
        epsilon0 = result[:3]
        kappa = result[3:]
        
        # Calculate failure indices for each ply
        failure_results = []
        failed_plies = []
        
        for i, ply in enumerate(st.session_state.laminate_layers):
            angle = ply['angle']
            thickness = ply['thickness']
            z_bottom = z_locations[i]
            z_top = z_locations[i+1]
            
            Qbar = calculate_Qbar_matrix(Q, angle)
            
            # Calculate transformation matrix
            theta = np.radians(angle)
            m = np.cos(theta)
            n = np.sin(theta)
            
            T = np.array([
                [m*2, n*2, 2*m*n],
                [n*2, m*2, -2*m*n],
                [-m*n, m*n, m*2-n*2]
            ])
            
            # Calculate at critical locations (top and bottom of each ply)
            for z_pos, location in zip([z_bottom, z_top], ['Bottom', 'Top']):
                # Strains in x-y coordinates
                strain_xy = epsilon0 + z_pos * kappa
                
                # Stresses in x-y coordinates
                stress_xy = Qbar.dot(strain_xy)
                
                # Stresses in material coordinates
                stress_12 = T.dot(stress_xy)
                sigma1, sigma2, tau12 = stress_12
                
                # Maximum Stress Criterion
                R1t = Xt/sigma1 if sigma1 > 0 else -Xc/sigma1
                R2t = Yt/sigma2 if sigma2 > 0 else -Yc/sigma2
                R12 = S/abs(tau12)
                max_stress_R = min(R1t, R2t, R12)
                max_stress_mode = ["Fiber", "Matrix", "Shear"][np.argmin([R1t, R2t, R12])]
                
                failure_status = "Will Fail" if max_stress_R < 1.0 else "Safe"
                
                if max_stress_R < 1.0:
                    failed_plies.append({
                        'Ply': i+1,
                        'Angle': angle,
                        'Location': location,
                        'Failure Mode': max_stress_mode,
                        'Safety Factor': f"{max_stress_R:.4f}",
                        'σ1 (MPa)': f"{sigma1:.2f}",
                        'σ2 (MPa)': f"{sigma2:.2f}",
                        'τ12 (MPa)': f"{tau12:.2f}"
                    })
                
                failure_results.append({
                    'Ply': i+1,
                    'Angle': angle,
                    'Location': location,
                    'σ1 (MPa)': f"{sigma1:.2f}",
                    'σ2 (MPa)': f"{sigma2:.2f}",
                    'τ12 (MPa)': f"{tau12:.2f}",
                    'Max Stress R': f"{max_stress_R:.4f}",
                    'Failure Mode': max_stress_mode,
                    'Status': failure_status
                })

        st.subheader("Failure Analysis Results (Maximum Stress Criterion)")
        st.dataframe(pd.DataFrame(failure_results))
        
        if failed_plies:
            st.subheader("All Plies That Will Fail")
            st.dataframe(pd.DataFrame(failed_plies))
            
            # Visualize failure locations
            st.subheader("Failure Visualization")
            
            # Create a plot showing the laminate stack with failed plies highlighted
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot laminate stack
            current_z = -total_thickness/2
            
            for i, ply in enumerate(st.session_state.laminate_layers):
                thickness = ply['thickness']
                angle = ply['angle']
                
                # Check if this ply has any failures
                ply_failures = [fp for fp in failed_plies if fp['Ply'] == i+1]
                
                # Color based on failure status
                color = 'red' if ply_failures else 'green'
                
                # Draw the ply
                rect = plt.Rectangle((0, current_z), 1, thickness, 
                                    linewidth=1, edgecolor='black', 
                                    facecolor=color, alpha=0.5)
                ax.add_patch(rect)
                
                # Add ply info
                plt.text(0.5, current_z + thickness/2, 
                        f"Ply {i+1}\n{angle}°\n{thickness}mm", 
                        ha='center', va='center', fontsize=8)
                
                current_z += thickness
            
            ax.set_xlim(0, 1)
            ax.set_ylim(-total_thickness/2, total_thickness/2)
            ax.set_title('Laminate Stack (Red = Failed, Green = Safe)')
            ax.set_ylabel('Position through thickness (mm)')
            ax.set_xticks([])
            st.pyplot(fig)
            
        else:
            st.success("All plies are safe according to Maximum Stress Criterion")

# Main app logic
if app_mode == "Material Properties":
    material_properties_module()
elif app_mode == "Single Ply Analysis":
    single_ply_analysis_module()
elif app_mode == "Laminate Analysis":
    laminate_analysis_module()
elif app_mode == "Failure Prediction":
    failure_prediction_module()

# Footer
st.markdown("---")
st.markdown("*Composite Material Analysis Tool* - Created for Composite Model Review")
