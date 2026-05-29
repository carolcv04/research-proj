#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Particle Packing Visualization Script

Visualizes 3D particle packing configuration with detailed statistics.
"""

import plotly.graph_objects as go
import numpy as np
from scipy.spatial.distance import pdist, squareform
from math import pi
from tqdm import tqdm


def create_sphere(center, radius, color, name=None, resolution=10):
    """Create a sphere mesh for Plotly"""
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]

    return go.Surface(
        x=x, y=y, z=z,
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        hoverinfo='skip',
        opacity=0.8
    )


def generate_sample_packing(num_particles=100, box_length=None):
    """
    Generate a sample particle packing for visualization.
    
    Parameters:
    -----------
    num_particles : int
        Number of particles to generate
    box_length : float
        Size of the box (if None, calculated from particles)
        
    Returns:
    --------
    np.ndarray : Packing data (N, 4) with columns [x, y, z, diameter]
    float : Box length
    """
    np.random.seed(42)
    
    # Generate random diameters
    diameters = np.random.uniform(0.5, 3.0, num_particles)
    
    # Generate positions
    if box_length is None:
        # Calculate box length based on particles
        particle_vol = np.sum((4/3) * pi * (diameters/2)**3)
        box_length = (particle_vol * 2)**(1/3)
    
    positions = np.random.uniform(0, box_length, (num_particles, 3))
    
    # Combine into packing
    packing = np.column_stack([positions, diameters])
    
    return packing, box_length


def calculate_porosity_metrics(packing, box_length):
    """
    Calculate porosity and packing metrics.
    
    Returns:
    --------
    dict : Dictionary with porosity metrics
    """
    particle_volumes = (4/3) * pi * (packing[:, 3] / 2)**3
    total_particle_volume = particle_volumes.sum()
    box_volume = box_length ** 3
    void_volume = box_volume - total_particle_volume
    
    theoretical_porosity = void_volume / box_volume
    final_porosity = theoretical_porosity  # In this demo, they're the same
    packing_fraction = 1 - final_porosity
    actual_density = np.sum(particle_volumes) / box_volume
    
    return {
        'theoretical_porosity': theoretical_porosity,
        'final_porosity': final_porosity,
        'packing_fraction': packing_fraction,
        'actual_density': actual_density,
        'total_particle_volume': total_particle_volume,
        'box_volume': box_volume,
        'void_volume': void_volume
    }


def visualize_packing(packing, box_length, metrics=None):
    """
    Create interactive 3D visualization of particle packing.
    
    Parameters:
    -----------
    packing : np.ndarray
        Particle data (N, 4) with columns [x, y, z, diameter]
    box_length : float
        Size of simulation domain
    metrics : dict
        Optional metrics dictionary
    """
    fig = go.Figure()
    
    # First view: Fast scatter plot
    print("\n📊 Creating fast scatter plot view...")
    fig.add_trace(go.Scatter3d(
        x=packing[:, 0],
        y=packing[:, 1],
        z=packing[:, 2],
        mode='markers',
        marker=dict(
            size=packing[:, 3] * 3,
            color=packing[:, 3],
            colorscale='Turbo',
            opacity=0.8,
            showscale=True,
            colorbar=dict(title="Diameter"),
            line=dict(width=0)
        ),
        hoverinfo='skip',
        name='Particles'
    ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        title="Fast Particle Packing View",
        width=1000,
        height=800
    )
    
    fig.show()
    
    # Second view: Detailed sphere visualization
    print("🎯 Creating detailed sphere visualization...")
    fig2 = go.Figure()
    
    # Color palette
    colors_map = [
        'rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)',
        'rgb(214, 39, 40)', 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
        'rgb(227, 119, 194)', 'rgb(127, 127, 127)', 'rgb(188, 143, 143)',
        'rgb(255, 152, 150)'
    ]
    
    # Add spheres
    for i in range(len(packing)):
        center = packing[i, :3]
        radius = packing[i, 3] / 2
        color = colors_map[i % len(colors_map)]
        fig2.add_trace(create_sphere(center, radius, color, name=f"Particle {i}"))
    
    # Draw box edges
    max_radius = (packing[:, 3] / 2).max()
    box_min = 0 - max_radius
    box_max = box_length + max_radius
    
    corners = np.array([
        [box_min, box_min, box_min],
        [box_max, box_min, box_min],
        [box_max, box_max, box_min],
        [box_min, box_max, box_min],
        [box_min, box_min, box_max],
        [box_max, box_min, box_max],
        [box_max, box_max, box_max],
        [box_min, box_max, box_max],
    ])
    
    edges = [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ]
    
    for edge in edges:
        points = corners[edge]
        fig2.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='lines',
            line=dict(color='red', width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    fig2.update_layout(
        width=1000,
        height=800,
        scene=dict(
            xaxis=dict(range=[box_min, box_max]),
            yaxis=dict(range=[box_min, box_max]),
            zaxis=dict(range=[box_min, box_max]),
            aspectmode='data',
        ),
        title=f'Particle Packing with Boundaries ({len(packing)} particles)',
        showlegend=False
    )
    
    fig2.show()


def print_statistics(packing, box_length, metrics=None):
    """
    Print detailed statistics about the packing.
    
    Parameters:
    -----------
    packing : np.ndarray
        Particle data
    box_length : float
        Box size
    metrics : dict
        Pre-calculated metrics (optional)
    """
    if metrics is None:
        metrics = calculate_porosity_metrics(packing, box_length)
    
    # Calculate distance matrix
    distances = squareform(pdist(packing[:, :3]))
    dm = pdist(packing[:, :3])
    
    print("\n" + "="*100)
    print("PARTICLE PACKING STATISTICS")
    print("="*100)
    
    # PARAMETER INFO
    print(f"\n{'PARAMETER INFO':^100}")
    print("-" * 100)
    print(f"Box size: {box_length:.4f}")
    print(f"Number of particles: {len(packing)}")
    unique_diameters, counts = np.unique(packing[:, 3], return_counts=True)
    print(f"Particle diameter distribution: {dict(zip(unique_diameters, counts))}")
    
    # DIAMETER STATISTICS
    print(f"\n{'DIAMETER STATISTICS':^100}")
    print("-" * 100)
    print(f"  Min diameter: {packing[:, 3].min():.4f}")
    print(f"  Max diameter: {packing[:, 3].max():.4f}")
    print(f"  Mean diameter: {packing[:, 3].mean():.4f}")
    print(f"  Std dev: {packing[:, 3].std():.4f}")
    if packing[:, 3].mean() > 0:
        print(f"  Coefficient of variation: {packing[:, 3].std() / packing[:, 3].mean():.4f}")
    
    # POROSITY METRICS
    print(f"\n{'POROSITY METRICS':^100}")
    print("-" * 100)
    print(f"  Theoretical porosity: {metrics['theoretical_porosity']:.6f}")
    print(f"  Final porosity: {metrics['final_porosity']:.6f}")
    porosity_error = abs(metrics['final_porosity'] - metrics['theoretical_porosity'])
    print(f"  Porosity error: {porosity_error:.6f}")
    
    # PACKING EFFICIENCY
    print(f"\n{'PACKING EFFICIENCY':^100}")
    print("-" * 100)
    print(f"  Packing fraction: {metrics['packing_fraction']:.6f}")
    print(f"  Actual density: {metrics['actual_density']:.6f}")
    
    # DENSITY & VOLUME
    print(f"\n{'VOLUME ANALYSIS':^100}")
    print("-" * 100)
    print(f"  Total particle volume: {metrics['total_particle_volume']:.4f}")
    print(f"  Box volume: {metrics['box_volume']:.4f}")
    print(f"  Void volume: {metrics['void_volume']:.4f}")
    
    # SPATIAL DISTRIBUTION
    print(f"\n{'SPATIAL DISTRIBUTION':^100}")
    print("-" * 100)
    print(f"  Min distance between particles: {min(dm):.4f}")
    print(f"  Max distance between particles: {max(dm):.4f}")
    print(f"  Mean distance between particles: {np.mean(dm):.4f}")
    print(f"  Std dev distance: {np.std(dm):.4f}")
    
    # COORDINATION NUMBER
    contact_distance = packing[:, 3].max() + 0.01
    coordination_number = (distances > 0) & (distances < contact_distance)
    avg_neighbors = coordination_number.sum(axis=1).mean()
    print(f"\n{'NEIGHBORHOOD ANALYSIS':^100}")
    print("-" * 100)
    print(f"  Average coordination number: {avg_neighbors:.2f}")
    
    # BOUNDARY CHECKS
    print(f"\n{'BOUNDARY CHECKS':^100}")
    print("-" * 100)
    particles_touching_boundary = 0
    
    for i in range(len(packing)):
        x, y, z, d = packing[i]
        r = d / 2
        if x - r < 0 or x + r > box_length or \
           y - r < 0 or y + r > box_length or \
           z - r < 0 or z + r > box_length:
            particles_touching_boundary += 1
    print(f"  Particles touching boundary: {particles_touching_boundary}/{len(packing)}")
    
    # OVERLAP DETECTION
    print(f"\n{'OVERLAP DETECTION':^100}")
    print("-" * 100)
    overlapping_pairs = 0

    # for i in tqdm(range(len(packing)), desc="Overlap check"):
    #     for j in range(i+1, len(packing)):
    #         dist = np.linalg.norm(packing[i, :3] - packing[j, :3])
    #         min_dist = (packing[i, 3] + packing[j, 3]) / 2
    #         if dist < min_dist - 0.001:
    #             overlapping_pairs += 1

    for i in range(len(packing)):
    for j in range(i+1, len(packing)):
        dist = np.linalg.norm(packing[i, :3] - packing[j, :3])
        min_dist = (packing[i, 3] + packing[j, 3]) / 2

        overlap_amount = min_dist - dist

        if overlap_amount > 0.001:
            overlaps.append((i, j, overlap_amount))

    # for i in range(len(packing)):
    #     for j in range(i+1, len(packing)):
    #         dist = np.linalg.norm(packing[i, :3] - packing[j, :3])
    #         min_dist = (packing[i, 3] + packing[j, 3]) / 2
    #         if dist < min_dist - 0.001:
    #             overlapping_pairs += 1
    print(f"  Overlapping particle pairs: {overlapping_pairs}")
    status = 'PASS ✓' if overlapping_pairs == 0 else 'FAIL ✗ - OVERLAPS DETECTED'
    print(f"  Status: {status}")
    
    # SUMMARY
    print(f"\n" + "="*100)
    success = overlapping_pairs == 0 and metrics['final_porosity'] <= metrics['theoretical_porosity'] * 1.1
    status_str = 'SUCCESS ✓' if success else 'FAILED ✗'
    print(f"SIMULATION STATUS: {status_str}".center(100))
    print("="*100 + "\n")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("PARTICLE PACKING VISUALIZATION".center(100))
    print("="*100)
    
    # Generate sample packing (or load your own)
    print("\n🔄 Generating sample particle packing...")
    packing, box_length = generate_sample_packing(num_particles=50)
    
    # Calculate metrics
    print("📈 Calculating metrics...")
    metrics = calculate_porosity_metrics(packing, box_length)
    
    # Print statistics
    print_statistics(packing, box_length, metrics)
    
    # Visualize
    print("🎨 Creating visualizations...")
    visualize_packing(packing, box_length, metrics)
    
    print("✓ Visualization complete!")