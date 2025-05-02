"""
AIMD Stability Analysis Module (Improved)

This module provides utilities to analyse and verify the stability of 
Additive Increase/Multiplicative Decrease (AIMD) controllers based on
the theoretical framework established by Chiu & Jain, adapted for
single-parameter control systems like bandwidth controllers.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import matplotlib.dates as mdates
from datetime import datetime, timedelta

class AIMDStabilityAnalyser:
    """
    Analyser for AIMD stability properties based on Chiu & Jain paper,
    adapted for single-parameter control scenarios.
    """
    
    @staticmethod
    def verify_stability_parameters(
        additive_increase: float, 
        multiplicative_decrease: float,
        min_rate: float,
        max_rate: float
    ) -> Tuple[bool, str]:
        """
        Verify that controller parameters satisfy Chiu & Jain stability conditions.
        
        Args:
            additive_increase: The additive increase parameter (a_I)
            multiplicative_decrease: The multiplicative decrease factor (b_D)
            min_rate: Minimum possible rate (X_min)
            max_rate: Maximum possible rate (X_max)
            
        Returns:
            Tuple containing:
            - Boolean indicating if parameters meet stability criteria
            - String with explanation/diagnostics
        """
        messages = []
        is_stable = True
        
        # Check basic constraints from Chiu & Jain paper
        if not (additive_increase > 0):
            messages.append(f"UNSTABLE: Additive increase parameter ({additive_increase}) must be > 0")
            is_stable = False
            
        if not (0 < multiplicative_decrease < 1):
            messages.append(f"UNSTABLE: Multiplicative decrease factor ({multiplicative_decrease}) must be between 0 and 1")
            is_stable = False
            
        # Calculate theoretical equilibrium properties
        if is_stable:
            # Calculate expected oscillation amplitude at equilibrium 
            # Based on AIMD theory: amplitude ≈ additive_increase / (1 - multiplicative_decrease)
            equilibrium_amplitude = additive_increase / (1 - multiplicative_decrease)
            
            # Estimate convergence time from max_rate to min_rate
            # Using multiplicative decrease: min_rate = max_rate * (multiplicative_decrease)^steps
            worst_case_steps = np.log(min_rate / max_rate) / np.log(multiplicative_decrease)
            
            # Equilibrium rate (theoretical center of oscillation)
            equilibrium_rate = (max_rate + min_rate) / 2
            
            messages.append(f"STABLE: Parameters satisfy Chiu & Jain stability criteria")
            messages.append(f"Expected equilibrium bandwidth oscillation: ±{equilibrium_amplitude:.2f} Mbps")
            messages.append(f"Worst-case convergence steps: {abs(worst_case_steps):.1f} feedback cycles")
            messages.append(f"Theoretical equilibrium rate: ~{equilibrium_rate:.2f} Mbps")
            
            # Analysis of convergence speed vs smoothness tradeoff
            if multiplicative_decrease < 0.5:
                messages.append("CAUTION: Low multiplicative decrease factor may cause aggressive rate reductions")
            if additive_increase > 10:
                messages.append("CAUTION: High additive increase value may cause large oscillations")
        
        return is_stable, "\n".join(messages)
    
    @staticmethod
    def plot_trajectory(trajectory_data: List[Dict], title: str = "AIMD Controller Trajectory"):
        """
        Plot the trajectory of the system, showing bandwidth adjustments over time.
        
        Args:
            trajectory_data: List of dictionaries with 'rate', 'latency', and 'feedback' keys
            title: Plot title
        """
        if not trajectory_data:
            return None
            
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        # Extract data
        timestamps = [point.get('timestamp', i) for i, point in enumerate(trajectory_data)]
        use_datetime = isinstance(timestamps[0], (datetime, np.datetime64))
        
        # Convert timestamps to matplotlib format for plotting
        if use_datetime:
            if isinstance(timestamps[0], float):
                # Convert float timestamps to datetime
                timestamps = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Format the x-axis for datetime
            time_format = mdates.DateFormatter('%H:%M:%S')
            ax1.xaxis.set_major_formatter(time_format)
            ax2.xaxis.set_major_formatter(time_format)
            
            # Convert to matplotlib dates for arithmetic operations
            plot_timestamps = mdates.date2num(timestamps)
        else:
            # Use the timestamps as is if they're already numeric
            plot_timestamps = timestamps
        
        rates = [point.get('rate', 0) for point in trajectory_data]
        latencies = [point.get('latency', 0) for point in trajectory_data]
        feedback_points = [i for i, point in enumerate(trajectory_data) if point.get('feedback', 0) == 1]
        
        # Extract AIMD parameters if available
        ai_value = trajectory_data[0].get('ai_factor', 0.5)
        md_value = trajectory_data[0].get('md_factor', 0.9)
        target_latency = None
        
        # Plot bandwidth rate over time
        ax1.plot(timestamps, rates, 'b-', marker='o', markersize=3, alpha=0.7, label='Bandwidth Rate')
        
        # Highlight decrease events
        if feedback_points:
            decrease_times = [timestamps[i] for i in feedback_points]
            decrease_rates = [rates[i] for i in feedback_points]
            ax1.scatter(decrease_times, decrease_rates, color='red', s=50, 
                      marker='x', label='Decrease Events', zorder=5)
        
        # Calculate and plot theoretical AIMD envelopes
        if feedback_points and len(feedback_points) > 1:
            # Find min/max points to estimate equilibrium range
            min_rate = min(rates)
            max_rate = max(rates)
            
            # Create a model of the expected AIMD behavior
            # This includes both multiplicative decrease and additive increase
            
            # Calculate the expected oscillation period based on AIMD parameters
            ai = ai_value  # Additive increase
            md = md_value  # Multiplicative decrease factor
            
            # Theoretical amplitude of oscillation
            amplitude = ai / (1 - md)
            
            # Calculate midpoint of oscillation (theoretical equilibrium)
            midpoint = (max_rate + min_rate) / 2
            
            # Model a simple oscillation pattern around the midpoint
            # Generate time points that extend from the first timestamp
            if use_datetime:
                # If using datetime, we need to use timedelta to create the theoretical model
                # Calculate average time delta between points
                if len(timestamps) > 1:
                    avg_delta = (plot_timestamps[-1] - plot_timestamps[0]) / len(plot_timestamps)
                else:
                    avg_delta = 0.01  # Default if we only have one point
                
                # Create numeric time points for the model
                numeric_time = np.linspace(0, 5*avg_delta, 1000)
                
                # Create the model rate values
                model_rate = np.zeros_like(numeric_time)
                period = 2 * amplitude / ai  # Approximate period based on increase/decrease cycle
                
                for i, t in enumerate(numeric_time):
                    cycle_position = (t % period) / period
                    if cycle_position < 0.7:  # Additive increase phase (longer)
                        model_rate[i] = midpoint - amplitude/2 + (ai * t % (amplitude))
                    else:  # Multiplicative decrease phase (shorter)
                        model_rate[i] = midpoint + amplitude/2 * (1 - (cycle_position - 0.7)/0.3)
                
                # Convert back to datetime for plotting
                model_time_numeric = plot_timestamps[0] + numeric_time
                model_time = mdates.num2date(model_time_numeric[:500])
                
                # Plot the theoretical AIMD behavior envelope
                ax1.plot(model_time, model_rate[:500], 'g--', 
                       alpha=0.5, label='Theoretical AIMD Behavior')
            else:
                # For numeric timestamps, we can use direct addition
                time_span = timestamps[-1] - timestamps[0]
                model_time = np.linspace(timestamps[0], timestamps[0] + time_span, 1000)
                model_rate = np.zeros_like(model_time)
                period = 2 * amplitude / ai
                
                for i, t in enumerate(model_time):
                    relative_t = t - timestamps[0]
                    cycle_position = (relative_t % period) / period
                    if cycle_position < 0.7:
                        model_rate[i] = midpoint - amplitude/2 + (ai * relative_t % (amplitude))
                    else:
                        model_rate[i] = midpoint + amplitude/2 * (1 - (cycle_position - 0.7)/0.3)
                
                # Plot the theoretical curve
                ax1.plot(model_time[:500], model_rate[:500], 'g--', 
                       alpha=0.5, label='Theoretical AIMD Behavior')
            
            # Draw the theoretical equilibrium point
            ax1.axhline(y=midpoint, color='g', linestyle='-.', 
                      alpha=0.5, label='Theoretical Equilibrium')
        
        # Plot latency over time on second subplot
        ax2.plot(timestamps, latencies, 'r-', alpha=0.7, label='Latency')
        
        # Add target latency line if available
        for point in trajectory_data:
            if 'objective' in point:
                target_latency = point.get('objective')
                break
                
        if target_latency:
            ax2.axhline(y=target_latency, color='g', linestyle='--', 
                      label=f'Target Latency ({target_latency}ms)')
        
        # Format the plots
        ax1.set_ylabel('Bandwidth Rate (Mbps)')
        ax1.set_title(f'{title} - AIMD Parameters: AI={ai_value}, MD={md_value}')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Latency (ms)')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        fig.autofmt_xdate()
        
        return fig

    @staticmethod
    def create_single_parameter_visualization(
        min_rate: float,
        max_rate: float,
        target_rate: float,
        additive_increase: float,
        multiplicative_decrease: float,
        trajectory_data: Optional[List[Dict]] = None,
        title: str = "AIMD Controller Behavior"
    ):
        """
        Create a visualisation of AIMD behavior for a single-parameter controller.
        
        Args:
            min_rate: Minimum rate
            max_rate: Maximum rate
            target_rate: Target operating rate
            additive_increase: The additive increase parameter
            multiplicative_decrease: The multiplicative decrease factor
            trajectory_data: Optional list of points from actual execution
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Set up the range for visualisation
        rate_range = max_rate - min_rate
        x_min = max(0, min_rate - rate_range * 0.1)
        x_max = max_rate + rate_range * 0.1
        
        # Plot the optimal operating point (target rate)
        ax.axvline(x=target_rate, color='g', linestyle='--', label='Target Operating Point')
        
        # Annotate regions
        ax.axvspan(x_min, target_rate, alpha=0.2, color='blue', label='Underutilised Region')
        ax.axvspan(target_rate, x_max, alpha=0.2, color='red', label='Overutilised Region')
        
        # Plot arrows showing the AIMD behavior
        arrow_y = 0.5
        arrow_props = dict(arrowstyle='->', lw=2, color='black')
        
        # Additive increase arrows (in underutilised region)
        ai_points = np.linspace(x_min, target_rate * 0.9, 4)
        for x in ai_points:
            ax.annotate('', xy=(x + additive_increase, arrow_y), 
                      xytext=(x, arrow_y), arrowprops=arrow_props)
        
        # Multiplicative decrease arrows (in overutilised region)
        md_points = np.linspace(target_rate * 1.1, x_max * 0.95, 4)
        for x in md_points:
            ax.annotate('', xy=(x * multiplicative_decrease, arrow_y), 
                      xytext=(x, arrow_y), arrowprops=arrow_props)
        
        # Add parameter information
        param_text = (
            f"AIMD Parameters:\n"
            f"Additive Increase = {additive_increase} Mbps\n"
            f"Multiplicative Decrease Factor = {multiplicative_decrease}\n"
            f"Expected Equilibrium Oscillation: ±{additive_increase/(1-multiplicative_decrease):.2f} Mbps"
        )
        
        # Place text box in top left corner
        ax.text(0.05, 0.95, param_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot actual trajectory if provided
        if trajectory_data:
            rates = [point.get('rate', 0) for point in trajectory_data]
            # Plot as points on x-axis
            y_positions = np.ones_like(rates) * 0.3
            ax.scatter(rates, y_positions, color='blue', s=20, alpha=0.5, label='Actual Rate Points')
        
        # Set plot limits and remove y-axis
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        
        # Add labels and legend
        ax.set_xlabel('Bandwidth Rate (Mbps)')
        ax.set_title(title)
        ax.legend(loc='lower right')
        
        # Add grid on x-axis only
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        return fig