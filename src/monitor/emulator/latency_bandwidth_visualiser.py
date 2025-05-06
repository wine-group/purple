#!/usr/bin/env python3
"""
Latency-Bandwidth Visualiser

This script analyses the relationship between bandwidth settings and observed latency,
with proper timestamp handling and dual y-axis visualisation.

Usage:
  python3 latency_bandwidth_visualiser.py --data trajectory_file.json
  or
  python3 latency_bandwidth_visualiser.py --csv scenario_stats.csv
"""

import argparse
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
from datetime import datetime
from scipy.stats import linregress
import pandas as pd
import re
import os
from typing import Dict, List, Optional, Tuple, Union

# Set better default styles for plots
plt.style.use('seaborn-v0_8')
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'sans-serif']
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['legend.fontsize'] = 10
rcParams['legend.frameon'] = True
rcParams['figure.titlesize'] = 16
rcParams['figure.dpi'] = 100

def load_trajectory_data(file_path: str) -> List[Dict]:
    """Load trajectory data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading trajectory data: {e}")
        return []

def load_csv_stats(file_path: str) -> List[Dict]:
    """Load statistics from a CSV file and convert to dictionary format"""
    try:
        data = []
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert values to appropriate types
                processed_row = {}
                for key, value in row.items():
                    if key == 'timestamp':
                        processed_row[key] = float(value)
                    elif key in ['latency_ms', 'throughput_mbps']:
                        try:
                            processed_row[key] = float(value)
                        except ValueError:
                            processed_row[key] = 0.0
                    elif key in ['bottleneck_bw', 'cake_bw']:
                        # Extract Mbps value from string like '100Mbps'
                        match = re.search(r'(\d+\.?\d*)(?:Mbps|mbit)', value)
                        if match:
                            processed_row[key] = float(match.group(1))
                        else:
                            processed_row[key] = 0.0
                    else:
                        try:
                            processed_row[key] = int(value)
                        except ValueError:
                            processed_row[key] = value
                
                # Add the processed row
                data.append(processed_row)
        return data
    except Exception as e:
        print(f"Error loading CSV stats: {e}")
        return []

def create_dual_axis_plot(data: Union[List[Dict], pd.DataFrame], 
                         is_trajectory: bool = True,
                         title_prefix: str = "") -> plt.Figure:
    """
    Create a plot with dual y-axes showing both latency and bandwidth over time
    
    Args:
        data: List of data points or DataFrame
        is_trajectory: Whether the data is from trajectory (True) or CSV stats (False)
        title_prefix: Optional prefix for the plot title
        
    Returns:
        Figure object with dual y-axis visualisation
    """
    if not data:
        return None
    
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Extract data based on source type
    if is_trajectory:
        if 'timestamp' in df.columns:
            # Ensure timestamps are float type
            df['timestamp'] = df['timestamp'].astype(float)
            # Convert Unix timestamps to datetime objects
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            # If no timestamp, create a sequence
            df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
        
        bandwidth_col = 'rate'
        latency_col = 'latency'
        data_source = "PURPLE-AIMD"
        # data_source = "Controller Trajectory"
        
        # Try to get target latency if available
        target_latency = None
        if 'objective' in df.columns and not df['objective'].empty:
            target_latency = df['objective'].iloc[0]
    else:
        # For CSV stats
        if 'timestamp' in df.columns:
            # Ensure timestamps are float type
            df['timestamp'] = df['timestamp'].astype(float)
            # Convert Unix timestamps to datetime objects
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        else:
            # If no timestamp, create a sequence
            df['datetime'] = pd.date_range(start='2023-01-01', periods=len(df), freq='S')
        
        bandwidth_col = 'cake_bw'
        latency_col = 'latency_ms'
        data_source = "Bottleneck Scenario Test"
        target_latency = None
        
        # Check if we have bottleneck_bw for comparison
        has_bottleneck = 'bottleneck_bw' in df.columns
    
    # Create figure with dual y-axes
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot bandwidth on the primary y-axis
    bandwidth_line, = ax1.plot(df['datetime'], df[bandwidth_col], 
                            color='#1f77b4',  # Blue
                            linestyle='-', 
                            marker='o', 
                            markersize=4, 
                            alpha=0.8,
                            linewidth=1.5,
                            markerfacecolor='white',
                            markeredgewidth=0.5,
                            label='Bandwidth Rate (Mbps)')
    
    # Set primary y-axis properties
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Bandwidth Rate (Mbps)', fontsize=12, fontweight='bold', color='#1f77b4')
    ax1.tick_params(axis='y', colors='#1f77b4')
    
    # Calculate appropriate y-axis limits with padding
    min_bw = df[bandwidth_col].min() if not df[bandwidth_col].empty else 0
    max_bw = df[bandwidth_col].max() if not df[bandwidth_col].empty else 100
    padding_bw = (max_bw - min_bw) * 0.1 if max_bw > min_bw else 10
    ax1.set_ylim(max(0, min_bw - padding_bw), max_bw + padding_bw)
    
    # Calculate and plot rolling average for bandwidth
    if len(df) > 2:
        window = min(10, len(df))
        df['rolling_bw'] = df[bandwidth_col].rolling(window=window, center=True).mean()
        rolling_line, = ax1.plot(df['datetime'], df['rolling_bw'], 
                           color='#1f77b4',  # Blue
                           linewidth=2.5, 
                           alpha=0.5, 
                           linestyle='--',
                           label=f'Bandwidth {window}-point Moving Average')
    
    # Create secondary y-axis for latency
    ax2 = ax1.twinx()
    
    # Plot latency on the secondary y-axis
    latency_line, = ax2.plot(df['datetime'], df[latency_col], 
                         color='#d62728',  # Red
                         linestyle='-', 
                         marker='s', 
                         markersize=4, 
                         alpha=0.8,
                         linewidth=1.5,
                         markerfacecolor='white',
                         markeredgewidth=0.5,
                         label='Latency (ms)')
    
    # Set secondary y-axis properties
    ax2.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold', color='#d62728')
    ax2.tick_params(axis='y', colors='#d62728')
    
    # Calculate appropriate y-axis limits with padding
    min_lat = df[latency_col].min() if not df[latency_col].empty else 0
    max_lat = df[latency_col].max() if not df[latency_col].empty else 100
    padding_lat = (max_lat - min_lat) * 0.1 if max_lat > min_lat else 5
    ax2.set_ylim(max(0, min_lat - padding_lat), max_lat + padding_lat)
    
    # Calculate and plot rolling average for latency
    if len(df) > 2:
        window = min(10, len(df))
        df['rolling_lat'] = df[latency_col].rolling(window=window, center=True).mean()
        rolling_lat_line, = ax2.plot(df['datetime'], df['rolling_lat'], 
                                color='#d62728',  # Red
                                linewidth=2.5, 
                                alpha=0.5, 
                                linestyle='--',
                                label=f'Latency {window}-point Moving Average')
    
    # Add target latency line if available - FIXED: removed the comma after target_line
    target_line = None
    if target_latency:
        target_line = ax2.axhline(y=target_latency, 
                              color='green', 
                              linestyle='--', 
                              linewidth=2,
                              alpha=0.8,
                              label=f'Target Latency ({target_latency}ms)')
    
    # Format x-axis for datetime
    time_format = mdates.DateFormatter('%H:%M:%S')  # Hours:Minutes:Seconds
    ax1.xaxis.set_major_formatter(time_format)
    
    # Determine appropriate locator based on time span
    time_span = (df['datetime'].max() - df['datetime'].min()).total_seconds()
    if time_span > 86400:  # More than a day
        ax1.xaxis.set_major_locator(mdates.DayLocator())
        time_format = mdates.DateFormatter('%Y-%m-%d')
        ax1.xaxis.set_major_formatter(time_format)
    elif time_span > 3600:  # More than an hour
        ax1.xaxis.set_major_locator(mdates.HourLocator())
    else:
        ax1.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
    
    # Set title with improved styling
    if title_prefix:
        title = f"{title_prefix} - Bandwidth and Latency over Time"
    else:
        title = f"{data_source} - Bandwidth and Latency over Time"
    
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add grid with better styling
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Collect handles and labels for the legend from both axes
    lines = [bandwidth_line, latency_line]
    labels = [l.get_label() for l in lines]
    
    if 'rolling_bw' in df.columns:
        lines.append(rolling_line)
        labels.append(rolling_line.get_label())
        
    if 'rolling_lat' in df.columns:
        lines.append(rolling_lat_line)
        labels.append(rolling_lat_line.get_label())
        
    if target_line:
        lines.append(target_line)
        labels.append(target_line.get_label())
    
    # Create unified legend for both axes
    ax1.legend(lines, labels, loc='upper left', frameon=True, 
              fancybox=True, framealpha=0.8, fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add annotations for extreme points
    if not df[bandwidth_col].empty and not df[latency_col].empty:
        # Find max bandwidth and latency points
        max_bw_idx = df[bandwidth_col].idxmax()
        max_lat_idx = df[latency_col].idxmax()
        
        # Annotate max bandwidth point
        ax1.annotate(f'Max BW: {df[bandwidth_col].iloc[max_bw_idx]:.1f} Mbps',
                    xy=(df['datetime'].iloc[max_bw_idx], df[bandwidth_col].iloc[max_bw_idx]),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#1f77b4'),
                    color='#1f77b4',
                    fontsize=9,
                    fontweight='bold')
        
        # Annotate max latency point
        ax2.annotate(f'Max Latency: {df[latency_col].iloc[max_lat_idx]:.1f} ms',
                    xy=(df['datetime'].iloc[max_lat_idx], df[latency_col].iloc[max_lat_idx]),
                    xytext=(10, -20),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', color='#d62728'),
                    color='#d62728',
                    fontsize=9,
                    fontweight='bold')
    
    # Adjust layout to make room for everything
    plt.tight_layout()
    fig.autofmt_xdate()  # Rotate dates for better fit
    
    return fig

def analyze_latency_bandwidth_relationship(data: Union[List[Dict], pd.DataFrame], 
                                           is_trajectory: bool = True,
                                           title_prefix: str = "") -> Tuple[plt.Figure, Dict]:
    """
    Analyse the relationship between bandwidth settings and observed latency
    
    Args:
        data: List of data points or DataFrame
        is_trajectory: Whether the data is from trajectory (True) or CSV stats (False)
        title_prefix: Optional prefix for the plot title
        
    Returns:
        Figure object and dictionary of analysis results
    """
    if not data:
        print("No data to analyse")
        return None, {}
    
    # Convert to DataFrame if it's not already
    if not isinstance(data, pd.DataFrame):
        df = pd.DataFrame(data)
    else:
        df = data.copy()
    
    # Extract bandwidth and latency data
    if is_trajectory:
        bandwidth_col = 'rate'
        latency_col = 'latency'
        data_source = "Controller Trajectory"
    else:
        # For CSV stats, use the cake_bw and latency_ms columns
        bandwidth_col = 'cake_bw'
        latency_col = 'latency_ms'
        data_source = "Bottleneck Scenario Test"
    
    # Get the data arrays
    bandwidth = np.array(df[bandwidth_col].tolist())
    latency = np.array(df[latency_col].tolist())
    
    # Remove invalid data points (where latency is 0 or negative)
    valid_idx = latency > 0
    bandwidth = bandwidth[valid_idx]
    latency = latency[valid_idx]
    
    if len(bandwidth) < 2:
        print("Not enough valid data points for analysis")
        return None, {}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create scatter plot with time progression color coding
    scatter = ax.scatter(bandwidth, latency, 
                       alpha=0.7, 
                       c=range(len(bandwidth)), 
                       cmap='viridis', 
                       s=60,  # Larger points
                       edgecolors='w',
                       linewidths=0.5,
                       label='Measured Points')
    
    # Add colorbar for time progression
    cbar = plt.colorbar(scatter, ax=ax, pad=0.01)
    cbar.set_label('Time Progression', fontsize=11)
    
    # Fit linear model
    results = {}
    try:
        slope, intercept, r_value, p_value, std_err = linregress(bandwidth, latency)
        linear_fit = slope * bandwidth + intercept
        r_squared = r_value**2
        
        ax.plot(bandwidth, linear_fit, 'r-', 
               linewidth=2, 
               alpha=0.7, 
               label=f'Linear Fit (R²={r_squared:.3f})')
        
        # Store basic results
        results['linear_slope'] = slope
        results['linear_intercept'] = intercept
        results['linear_r_squared'] = r_squared
    except Exception as e:
        print(f"Error fitting linear model: {e}")
    
    # Try polynomial fits if we have enough data points
    if len(bandwidth) > 5:
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score
            
            # Reshape for sklearn
            X = bandwidth.reshape(-1, 1)
            
            # Try different polynomial degrees
            best_r2 = 0
            best_degree = 0
            best_model = None
            best_poly = None
            
            for degree in [2, 3]:
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X)
                
                model = LinearRegression()
                model.fit(X_poly, latency)
                
                # Generate points for plotting the curve
                X_range = np.linspace(min(bandwidth), max(bandwidth), 100).reshape(-1, 1)
                X_range_poly = poly.transform(X_range)
                y_pred = model.predict(X_range_poly)
                
                # Calculate R-squared
                y_pred_data = model.predict(X_poly)
                poly_r2 = r2_score(latency, y_pred_data)
                
                # Plot the curve
                ax.plot(X_range, y_pred, 
                       linestyle='-', 
                       linewidth=2.5, 
                       alpha=0.7,
                       label=f'Polynomial Fit (degree={degree}, R²={poly_r2:.3f})')
                
                # Track the best model
                if poly_r2 > best_r2:
                    best_r2 = poly_r2
                    best_degree = degree
                    best_model = model
                    best_poly = poly
            
            if best_model is not None:
                # Generate smooth curve from the best model
                X_smooth = np.linspace(min(bandwidth), max(bandwidth), 1000).reshape(-1, 1)
                X_smooth_poly = best_poly.transform(X_smooth)
                y_smooth = best_model.predict(X_smooth_poly)
                
                # Calculate the derivative (rate of change)
                dy = np.gradient(y_smooth.ravel())
                dx = np.gradient(X_smooth.ravel())
                derivative = dy / dx
                
                # Use a percentile threshold to find the knee point
                threshold_percentile = 75
                threshold = np.percentile(derivative, threshold_percentile)
                
                # Find where the derivative exceeds the threshold
                knee_indices = np.where(derivative > threshold)[0]
                
                if len(knee_indices) > 0:
                    # Take the first point that exceeds the threshold
                    knee_index = knee_indices[0]
                    knee_bandwidth = X_smooth[knee_index, 0]
                    knee_latency = y_smooth[knee_index]
                    
                    # Mark the knee point with improved styling
                    ax.scatter([knee_bandwidth], [knee_latency], 
                              color='red', 
                              s=150,  # Larger marker
                              marker='*', 
                              edgecolors='white',
                              linewidths=1,
                              zorder=10,  # Ensure it's on top
                              label='Estimated Knee Point')
                    
                    # Add annotation for knee point
                    ax.annotate(f'Knee: ({knee_bandwidth:.1f} Mbps, {knee_latency:.1f} ms)',
                              xy=(knee_bandwidth, knee_latency),
                              xytext=(knee_bandwidth+10, knee_latency+5),
                              fontsize=11,
                              fontweight='bold',
                              arrowprops=dict(
                                  facecolor='black', 
                                  shrink=0.05, 
                                  width=1.5,
                                  alpha=0.7
                              ))
                
                    # Calculate optimal operating point - slightly before the knee
                    optimal_bandwidth = knee_bandwidth * 0.9  # 90% of knee point as a heuristic
                    
                    # Calculate expected latency at optimal point
                    optimal_index = np.argmin(np.abs(X_smooth.ravel() - optimal_bandwidth))
                    optimal_latency = y_smooth[optimal_index]
                    
                    # Mark optimal operating point
                    ax.scatter([optimal_bandwidth], [optimal_latency], 
                              color='green', 
                              s=150,  # Larger marker
                              marker='o', 
                              edgecolors='white',
                              linewidths=1,
                              zorder=10,  # Ensure it's on top
                              label='Suggested Operating Point')
                    
                    # Add annotation for optimal point
                    ax.annotate(f'Optimal: ({optimal_bandwidth:.1f} Mbps, {optimal_latency:.1f} ms)',
                              xy=(optimal_bandwidth, optimal_latency),
                              xytext=(optimal_bandwidth-40, optimal_latency+10),
                              fontsize=11,
                              fontweight='bold',
                              arrowprops=dict(
                                  facecolor='green', 
                                  shrink=0.05, 
                                  width=1.5,
                                  alpha=0.7
                              ))
                    
                    # Save analytical results
                    results.update({
                        'knee_bandwidth': float(knee_bandwidth),
                        'knee_latency': float(knee_latency),
                        'optimal_bandwidth': float(optimal_bandwidth),
                        'optimal_latency': float(optimal_latency),
                        'best_poly_degree': best_degree,
                        'best_poly_r_squared': best_r2
                    })
                else:
                    results['error'] = 'Could not identify knee point'
        except Exception as e:
            print(f"Error in curve fitting: {e}")
            import traceback
            traceback.print_exc()
            results['error'] = str(e)
    
    # Set labels and title
    ax.set_xlabel('Bandwidth Rate (Mbps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latency (ms)', fontsize=12, fontweight='bold')
    
    # Construct the title
    if title_prefix:
        title = f"{title_prefix} - Latency vs. Bandwidth Analysis"
    else:
        title = f"{data_source} - Latency vs. Bandwidth Analysis"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Add grid with better styling
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Ensure y-axis starts near zero but with some padding
    ylim = ax.get_ylim()
    ax.set_ylim(max(0, ylim[0] * 0.9), ylim[1] * 1.05)  # 5% extra space at top
    
    # Set x-axis to start at 0 or slightly below the minimum
    x_min = min(0, min(bandwidth) * 0.95)
    x_max = max(bandwidth) * 1.05  # 5% extra space
    ax.set_xlim(x_min, x_max)
    
    # Better legend - place outside the plot area for more room
    leg = ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.01, 1),
        fontsize=11,
        frameon=True,
        fancybox=True,
        framealpha=0.8,
        borderpad=1,
        title="Legend",
        title_fontsize=12
    )
    
    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(right=0.85)  # Make room for legend
    
    return fig, results

def main():
    parser = argparse.ArgumentParser(description='Analyse latency-bandwidth relationship with improved visualisation')
    parser.add_argument('--data', help='Path to trajectory data JSON file')
    parser.add_argument('--csv', help='Path to CSV stats file')
    parser.add_argument('--output-prefix', default='latency_analysis', 
                      help='Prefix for output files')
    parser.add_argument('--title', default='',
                      help='Custom title prefix for plots')
    parser.add_argument('--dpi', type=int, default=300,
                      help='DPI for output images (default: 300)')
    args = parser.parse_args()
    
    if not args.data and not args.csv:
        print("Error: You must specify either --data or --csv")
        parser.print_help()
        return
    
    # Determine data source and load data
    if args.data:
        data = load_trajectory_data(args.data)
        is_trajectory = True
        source_file = args.data
    else:
        data = load_csv_stats(args.csv)
        is_trajectory = False
        source_file = args.csv
    
    if not data:
        print(f"No data available from {source_file}. Exiting.")
        return
    
    # Set output prefix based on input file if not specified
    if args.output_prefix == 'latency_analysis':
        args.output_prefix = os.path.splitext(os.path.basename(source_file))[0]
    
    # Create dual y-axis plot (new functionality)
    fig_dual = create_dual_axis_plot(data, is_trajectory, title_prefix=args.title)
    if fig_dual:
        output_file = f"{args.output_prefix}_dual_axis.png"
        fig_dual.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved dual axis plot to {output_file}")
    
    # Analyse latency-bandwidth relationship
    fig_scatter, results = analyze_latency_bandwidth_relationship(
        data, is_trajectory, title_prefix=args.title
    )
    
    if fig_scatter:
        output_file = f"{args.output_prefix}_relationship.png"
        fig_scatter.savefig(output_file, dpi=args.dpi, bbox_inches='tight')
        print(f"Saved relationship analysis to {output_file}")
        
        # Save analytical results
        results_file = f"{args.output_prefix}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved analytical results to {results_file}")
    
    # Print key findings
    if 'optimal_bandwidth' in results:
        print("\nKey Findings:")
        print(f"Estimated knee point: {results['knee_bandwidth']:.2f} Mbps at {results['knee_latency']:.2f} ms latency")
        print(f"Suggested optimal operating point: {results['optimal_bandwidth']:.2f} Mbps at {results['optimal_latency']:.2f} ms latency")
        print(f"This should be used as your target bandwidth for optimal performance.")
    elif 'error' in results:
        print(f"\nAnalysis note: {results['error']}")

if __name__ == "__main__":
    main()