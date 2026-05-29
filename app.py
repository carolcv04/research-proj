#!/usr/bin/env python3
"""
Particle Packing Visualization App
A Streamlit web application for interactive 3D particle packing visualization and analysis
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
from math import pi
from datetime import datetime
from pathlib import Path
from io import BytesIO
import pandas as pd


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Particle Packing Visualization",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STATISTICS & CALCULATIONS
# ============================================================================

def calculate_metrics(packing, box_length):
    """Calculate all packing metrics"""
    diameters = packing[:, 3]
    positions = packing[:, :3]
    
    # Volume calculations
    particle_volumes = (4/3) * pi * (diameters/2)**3
    total_particle_volume = particle_volumes.sum()
    box_volume = box_length**3
    void_volume = box_volume - total_particle_volume
    porosity = void_volume / box_volume
    packing_fraction = 1 - porosity
    actual_density = total_particle_volume / box_volume
    
    # Distance metrics
    dm = pdist(positions)
    distances = squareform(dm)
    
    # Coordination number
    contact_distance = diameters.max() + 0.01
    coordination_number = (distances > 0) & (distances < contact_distance)
    avg_neighbors = coordination_number.sum(axis=1).mean()
    
    # Boundary checks
    boundary_particles = 0
    for i in range(len(packing)):
        x, y, z, d = packing[i]
        r = d / 2
        if x - r < 0 or x + r > box_length or \
           y - r < 0 or y + r > box_length or \
           z - r < 0 or z + r > box_length:
            boundary_particles += 1
    
    # Overlap detection
    overlapping_pairs = 0
    overlapping_indices = set()

    for i in range(len(packing)):
        for j in range(i+1, len(packing)):
            dist = np.linalg.norm(packing[i, :3] - packing[j, :3])
            min_dist = (packing[i, 3] + packing[j, 3]) / 2
            
            if dist < min_dist - 0.001:
                overlapping_pairs += 1
                overlapping_indices.add(i)
                overlapping_indices.add(j)
    # overlapping_pairs = 0
    # for i in range(len(packing)):
    #     for j in range(i+1, len(packing)):
    #         dist = np.linalg.norm(packing[i, :3] - packing[j, :3])
    #         min_dist = (packing[i, 3] + packing[j, 3]) / 2
    #         if dist < min_dist - 0.001:
    #             overlapping_pairs += 1
    
    return {
        'diameters': diameters,
        'distances': dm,
        'particle_volumes': particle_volumes,
        'total_particle_volume': total_particle_volume,
        'box_volume': box_volume,
        'void_volume': void_volume,
        'porosity': porosity,
        'packing_fraction': packing_fraction,
        'actual_density': actual_density,
        'avg_neighbors': avg_neighbors,
        'boundary_particles': boundary_particles,
        'total_particles': len(packing),
        'overlapping_pairs': overlapping_pairs,
        'overlapping_indices': list(overlapping_indices)
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_3d_plot(packing, box_length, num_particles):
    """Create interactive 3D scatter plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=packing[:, 0],
        y=packing[:, 1],
        z=packing[:, 2],
        mode='markers',
        marker=dict(
            size=packing[:, 3] * 3,
            color=packing[:, 3],
            colorscale='Turbo',
            showscale=True,
            colorbar=dict(
                title="Diameter",
                thickness=15,
                len=0.7,
            ),
            opacity=0.85,
            line=dict(width=0)
        ),
        text=[f"Particle {i}<br>Diameter: {packing[i,3]:.4f}" for i in range(len(packing))],
        hoverinfo='text',
        name='Particles'
    ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[0, box_length], title='X'),
            yaxis=dict(range=[0, box_length], title='Y'),
            zaxis=dict(range=[0, box_length], title='Z'),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        title=f"<b>3D Particle Packing ({num_particles} particles)</b>",
        height=700,
        width=1000,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_html_report(packing, box_length, num_particles, min_diameter, max_diameter, seed_val, fig_html, metrics):
    """Generate comprehensive HTML report"""
    
    status_color = "success" if metrics['overlapping_pairs'] == 0 else "danger"
    status_text = "✓ PASS" if metrics['overlapping_pairs'] == 0 else "✗ FAIL"
    
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Particle Packing Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
        <style>
            * {{
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }}
            
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 40px 20px;
                line-height: 1.6;
                color: #333;
            }}
            
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 12px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.15);
                overflow: hidden;
            }}
            
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }}
            
            .header h1 {{
                font-size: 2.5em;
                margin-bottom: 10px;
            }}
            
            .header p {{
                font-size: 1.1em;
                opacity: 0.9;
            }}
            
            .content {{
                padding: 40px;
            }}
            
            .section {{
                margin-bottom: 40px;
            }}
            
            .section h2 {{
                border-bottom: 3px solid #667eea;
                padding-bottom: 15px;
                margin-bottom: 25px;
                font-size: 1.8em;
                color: #667eea;
            }}
            
            .metrics-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            
            .metric-card {{
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            }}
            
            .metric-label {{
                font-weight: 600;
                color: #555;
                font-size: 0.95em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 8px;
            }}
            
            .metric-value {{
                font-size: 1.8em;
                color: #667eea;
                font-weight: bold;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                border-radius: 8px;
                overflow: hidden;
            }}
            
            th {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px;
                text-align: left;
                font-weight: 600;
                text-transform: uppercase;
                font-size: 0.9em;
                letter-spacing: 0.5px;
            }}
            
            td {{
                padding: 15px;
                border-bottom: 1px solid #eee;
            }}
            
            tr:hover {{
                background-color: #f8f9fa;
            }}
            
            .visualization {{
                margin: 30px 0;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                border: 2px dashed #667eea;
            }}
            
            .status-badge {{
                display: inline-block;
                padding: 15px 30px;
                border-radius: 8px;
                font-size: 1.2em;
                font-weight: bold;
                text-align: center;
                margin: 20px 0;
            }}
            
            .status-badge.success {{
                background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
                color: #2d5016;
            }}
            
            .status-badge.danger {{
                background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                color: #8B0000;
            }}
            
            .footer {{
                background: #f8f9fa;
                padding: 20px;
                text-align: center;
                color: #666;
                border-top: 1px solid #eee;
            }}
            
            .footer p {{
                margin: 5px 0;
                font-size: 0.9em;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Particle Packing Analysis Report</h1>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="content">
                <!-- CONFIGURATION -->
                <div class="section">
                    <h2>📋 Configuration Parameters</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Total Particles</div>
                            <div class="metric-value">{num_particles}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Random Seed</div>
                            <div class="metric-value">{seed_val}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Box Size</div>
                            <div class="metric-value">{box_length:.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Min Diameter</div>
                            <div class="metric-value">{min_diameter:.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Max Diameter</div>
                            <div class="metric-value">{max_diameter:.4f}</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Mean Diameter</div>
                            <div class="metric-value">{metrics['diameters'].mean():.4f}</div>
                        </div>
                    </div>
                </div>
                
                <!-- VISUALIZATION -->
                <div class="section">
                    <h2>🎨 3D Visualization</h2>
                    <div class="visualization">
                        {fig_html}
                    </div>
                </div>
                
                <!-- DIAMETER STATISTICS -->
                <div class="section">
                    <h2>📊 Diameter Statistics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Minimum Diameter</td>
                            <td>{metrics['diameters'].min():.6f}</td>
                        </tr>
                        <tr>
                            <td>Maximum Diameter</td>
                            <td>{metrics['diameters'].max():.6f}</td>
                        </tr>
                        <tr>
                            <td>Mean Diameter</td>
                            <td>{metrics['diameters'].mean():.6f}</td>
                        </tr>
                        <tr>
                            <td>Standard Deviation</td>
                            <td>{metrics['diameters'].std():.6f}</td>
                        </tr>
                        <tr>
                            <td>Coefficient of Variation</td>
                            <td>{metrics['diameters'].std() / metrics['diameters'].mean():.6f}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- POROSITY METRICS -->
                <div class="section">
                    <h2>🌊 Porosity & Density Metrics</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Porosity</td>
                            <td>{metrics['porosity']:.6f}</td>
                        </tr>
                        <tr>
                            <td>Packing Fraction</td>
                            <td>{metrics['packing_fraction']:.6f}</td>
                        </tr>
                        <tr>
                            <td>Actual Density</td>
                            <td>{metrics['actual_density']:.6f}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- VOLUME ANALYSIS -->
                <div class="section">
                    <h2>📏 Volume Analysis</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Particle Volume</td>
                            <td>{metrics['total_particle_volume']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Box Volume</td>
                            <td>{metrics['box_volume']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Void Volume</td>
                            <td>{metrics['void_volume']:.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- SPATIAL DISTRIBUTION -->
                <div class="section">
                    <h2>📍 Spatial Distribution</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Minimum Distance Between Particles</td>
                            <td>{metrics['distances'].min():.4f}</td>
                        </tr>
                        <tr>
                            <td>Maximum Distance Between Particles</td>
                            <td>{metrics['distances'].max():.4f}</td>
                        </tr>
                        <tr>
                            <td>Mean Distance Between Particles</td>
                            <td>{np.mean(metrics['distances']):.4f}</td>
                        </tr>
                        <tr>
                            <td>Distance Standard Deviation</td>
                            <td>{np.std(metrics['distances']):.4f}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- NEIGHBORHOOD -->
                <div class="section">
                    <h2>👥 Neighborhood Analysis</h2>
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Average Coordination Number</td>
                            <td>{metrics['avg_neighbors']:.2f}</td>
                        </tr>
                        <tr>
                            <td>Particles Touching Boundary</td>
                            <td>{metrics['boundary_particles']} / {metrics['total_particles']}</td>
                        </tr>
                    </table>
                </div>
                
                <!-- OVERLAP DETECTION -->
                <!-- 
                    <div class="section">
                        <h2>⚠️ Overlap Detection</h2>
                        <table>
                            <tr>
                                <th>Metric</th>
                                <th>Value</th>
                            </tr>
                            <tr>
                                <td>Overlapping Particle Pairs</td>
                                <td>{metrics['overlapping_pairs']}</td>
                            </tr>
                            <tr>
                                <td>Status</td>
                                <td>{'✓ PASS' if metrics['overlapping_pairs'] == 0 else '✗ FAIL'}</td>
                            </tr>
                        </table>
                        <div class="status-badge {status_color}">
                            {status_text} - {'No Overlaps Detected' if metrics['overlapping_pairs'] == 0 else f'{metrics["overlapping_pairs"]} Overlaps Found'}
                        </div>
                    </div>
                <!-- 
            </div>
            
            <div class="footer">
                <p><strong>Report Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Tool:</strong> Particle Packing Visualization System</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html


def save_report(html_content, num_particles):
    """Save report to local folder with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    folder_name = f"{timestamp}-{num_particles}"
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    
    report_path = folder_path / "report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    return folder_path, report_path


def upload_to_onedrive(html_content, csv_content, num_particles):
    """Upload report and CSV to OneDrive cloud"""
    try:
        from O365 import Account
        
        # Authenticate with Microsoft account
        credentials = ('caraballovlez@chapman.edu', 'Casper1108!')
        account = Account(credentials)
        
        if not account.authenticate():
            return False, "Authentication failed. Check your credentials."
        
        # Access OneDrive
        storage = account.storage()
        drive = storage.get_default_drive()
        
        # Create folder: Research Resources/Literature Review/Script Output/{timestamp}-{particles}
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        folder_name = f"{timestamp}-{num_particles}"
        
        # Navigate to target folder
        target_folder = drive.get_item_by_path("Research Resources/Literature Review/Script Output")
        if not target_folder:
            target_folder = drive.get_root_folder()
        
        # Create subfolder
        new_folder = target_folder.create_folder(folder_name)
        
        # Upload HTML report
        report_bytes = html_content.encode('utf-8')
        report_name = f"{timestamp}-{num_particles}.html"
        new_folder.upload_file(file_stream=BytesIO(report_bytes), file_name=report_name)
        
        # Upload CSV data
        csv_bytes = csv_content.encode('utf-8')
        csv_name = f"{timestamp}-packing-{num_particles}.csv"
        new_folder.upload_file(file_stream=BytesIO(csv_bytes), file_name=csv_name)
        
        return True, f"✓ Files uploaded to OneDrive: {folder_name}"
    
    except Exception as e:
        return False, f"Error: {str(e)}"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # HEADER
    st.title("🎯 Particle Packing Visualization")
    st.markdown("Interactive 3D visualization and analysis of particle packings")
    
    # SIDEBAR CONTROLS
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            num_particles = st.number_input(
                "Number of Particles",
                min_value=10,
                max_value=10000,
                value=50,
                step=10
            )
        with col2:
            seed_val = st.number_input(
                "Random Seed",
                value=42,
                min_value=0
            )
        
        col1, col2 = st.columns(2)
        with col1:
            min_diameter = st.number_input(
                "Min Diameter",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1
            )
        with col2:
            max_diameter = st.number_input(
                "Max Diameter",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.1
            )
    
    # GENERATE PACKING
    np.random.seed(int(seed_val))
    diameters = np.random.uniform(min_diameter, max_diameter, num_particles)
    particle_vol = np.sum((4/3) * pi * (diameters/2)**3)
    box_length = (particle_vol * 2)**(1/3)
    positions = np.random.uniform(0, box_length, (num_particles, 3))
    packing = np.column_stack([positions, diameters])
    
    # CALCULATE METRICS
    metrics = calculate_metrics(packing, box_length)

    # =========================================================
    # ⚠️ OVERLAP DEBUG SECTION (ADD THIS HERE)
    # =========================================================
    if metrics['overlapping_pairs'] > 0:
        st.subheader("⚠️ Overlapping Particles Breakdown")

        overlap_ids = metrics['overlapping_indices']
        
        st.write(f"Total overlapping particles: {len(overlap_ids)}")
        st.write("Particle indices involved:", overlap_ids)
        
        overlap_table = packing[overlap_ids]

        st.dataframe(
            pd.DataFrame({
                "Index": overlap_ids,
                "X": overlap_table[:, 0],
                "Y": overlap_table[:, 1],
                "Z": overlap_table[:, 2],
                "Diameter": overlap_table[:, 3]
            }).style.highlight_max(axis=0)
        )
        
    # DISPLAY PLOT
    fig = create_3d_plot(packing, box_length, num_particles)
    st.plotly_chart(fig, use_container_width=True)
    
    # KEY METRICS
    st.markdown("---")
    st.subheader("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Box Size", f"{box_length:.4f}")
    with col2:
        st.metric("Porosity", f"{metrics['porosity']:.4f}")
    with col3:
        st.metric("Packing Fraction", f"{metrics['packing_fraction']:.4f}")
    with col4:
        overlap_status = "✓ Pass" if metrics['overlapping_pairs'] == 0 else "✗ Fail"
        st.metric("Overlap Status", overlap_status)
    
    # DETAILED STATISTICS
    if st.checkbox("📈 Show Detailed Statistics"):
        st.subheader("Detailed Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Distance Statistics:**")
            st.write(f"Min: {metrics['distances'].min():.4f}")
            st.write(f"Max: {metrics['distances'].max():.4f}")
            st.write(f"Mean: {np.mean(metrics['distances']):.4f}")
        
        with col2:
            st.write("**Volume:**")
            st.write(f"Total Particle: {metrics['total_particle_volume']:.4f}")
            st.write(f"Box: {metrics['box_volume']:.4f}")
            st.write(f"Void: {metrics['void_volume']:.4f}")
        
        with col3:
            st.write("**Diameter:**")
            st.write(f"Min: {metrics['diameters'].min():.4f}")
            st.write(f"Max: {metrics['diameters'].max():.4f}")
            st.write(f"Mean: {metrics['diameters'].mean():.4f}")
    
    # EXPORT SECTION
    st.markdown("---")
    st.subheader("💾 Export & Download")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📥 Download Locally", use_container_width=True):
            fig_html = fig.to_html(include_plotlyjs='cdn')
            html_report = generate_html_report(
                packing, box_length, num_particles, 
                min_diameter, max_diameter, seed_val, 
                fig_html, metrics
            )
            
            folder_path, report_path = save_report(html_report, num_particles)
            
            st.success(f"✓ Report saved to: `{folder_path}`")
            
            st.download_button(
                label="📄 Download HTML Report",
                data=html_report,
                file_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{num_particles}.html",
                mime="text/html",
                use_container_width=True
            )
    
    with col2:
        if st.button("📊 Download CSV Data", use_container_width=True):
            csv_data = "x,y,z,diameter\n"
            for particle in packing:
                csv_data += f"{particle[0]:.6f},{particle[1]:.6f},{particle[2]:.6f},{particle[3]:.6f}\n"
            
            st.download_button(
                label="📥 Get CSV Data",
                data=csv_data,
                file_name=f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-packing-{num_particles}.csv",
                mime="text/csv",
                use_container_width=True
            )
    
    with col3:
        if st.button("☁️ Upload to OneDrive", use_container_width=True):
            st.info("🔐 Initializing OneDrive authentication...")
            
            # Generate reports
            fig_html = fig.to_html(include_plotlyjs='cdn')
            html_report = generate_html_report(
                packing, box_length, num_particles, 
                min_diameter, max_diameter, seed_val, 
                fig_html, metrics
            )
            
            csv_data = "x,y,z,diameter\n"
            for particle in packing:
                csv_data += f"{particle[0]:.6f},{particle[1]:.6f},{particle[2]:.6f},{particle[3]:.6f}\n"
            
            # Attempt upload
            success, message = upload_to_onedrive(html_report, csv_data, num_particles)
            
            if success:
                st.success(message)
            else:
                st.error(message)
                st.warning("⚠️ Make sure you have O365 library installed: `pip install O365`")
    
    # ONEDRIVE INFO
    if st.checkbox("☁️ OneDrive Integration Help"):
        st.info(
            """
            **OneDrive Cloud Upload:**
            
            To enable OneDrive upload:
            
            1. **Install library**: `pip install O365`
            
            2. **Get your credentials**:
               - Email: your Chapman email (xxx@chapman.edu)
               - Password: your Chapman password
            
            3. **Update app.py** (line with credentials):
               ```python
               credentials = ('your_email@chapman.edu', 'your_password')
               ```
            
            4. **Click "Upload to OneDrive"** button
            
            **Files will be saved to:**
            `Research Resources/Literature Review/Script Output/{YYYYMMDD-HHMMSS}-{particle_count}/`
            """
        )


if __name__ == "__main__":
    main()