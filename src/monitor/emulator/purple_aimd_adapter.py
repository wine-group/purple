#!/usr/bin/env python3
"""
PURPLE-AIMD Latency Controller Adapter

This script adapts the PURPLE-AIMD Controller to work with the simplified bottleneck
environment. It uses *only* latency measurements to control the CAKE bandwidth,
making it equipment-agnostic. This version includes (broken) stability analysis
based on Chiu & Jain's paper.

Usage: sudo python3 purple_aimd_adapter.py [options]
"""

import argparse
import csv
import logging
import subprocess
import sys
import time
import os
import signal
import re
import json
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, List, Any

# Import our controller - adjust path as needed
sys.path.append('.')
try:
    from purple_aimd_controller import PurpleAIMDController, PurpleAIMDConfig
    controller_import_success = True
except ImportError:
    controller_import_success = False
    print("Warning: Could not import PURPLE-AIMD Controller. Make sure it's in the current directory.")

# Import our stability analyzer
try:
    from stability_analysis import AIMDStabilityAnalyser
    stability_analyzer_available = True
except ImportError:
    stability_analyzer_available = False
    print("Warning: Could not import AIMM Stability Analyzer. Some features will be disabled.")

# Constants
CLIENT_NS = "client_ns"
CLIENT_VETH = "veth0"
DEFAULT_SAMPLE_INTERVAL = 0.1  # 100ms
DEFAULT_LOG_FILE = "pure_latency_control.csv"
DEFAULT_TRAJECTORY_FILE = "aimd_trajectory.json"

class NetworkMonitor:
    """
    Minimal network monitor that only measures latency and updates CAKE
    """
    
    def __init__(self):
        self.logger = logging.getLogger('NetworkMonitor')
    
    def run_command(self, cmd: str, namespace: Optional[str] = None) -> str:
        """Run a shell command, optionally in a network namespace"""
        try:
            if namespace:
                full_cmd = f"ip netns exec {namespace} {cmd}"
            else:
                full_cmd = cmd
            
            result = subprocess.run(full_cmd, shell=True, check=True, 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True)
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed: {e}")
            return ""
    
    def measure_latency(self) -> float:
        """Measure ping latency from client to server"""
        try:
            cmd = "ping -c 1 -w 1 -q 192.168.200.2"
            output = self.run_command(cmd, CLIENT_NS)
                
            # Try the rtt statistics pattern
            rtt_match = re.search(r'rtt min/avg/max/mdev = (\d+\.?\d*)/(\d+\.?\d*)/(\d+\.?\d*)/(\d+\.?\d*)', output)
            if rtt_match:
                # Use the average value (second group)
                latency = float(rtt_match.group(2))
                return latency
                
            # If we get here, no pattern matched
            return 5.0  # Default fallback
                
        except Exception as e:
            self.logger.error(f"Error measuring latency: {e}")
            return 5.0
    
    def update_cake_bandwidth(self, bandwidth_mbps: float) -> bool:
        """Update CAKE bandwidth on client interface"""
        try:
            # Ensure bandwidth is within reasonable limits
            bandwidth_mbps = max(10.0, min(1000.0, bandwidth_mbps))
            
            # Convert Mbps to bits per second
            bandwidth_bps = int(bandwidth_mbps * 1_000_000)
            
            # Build tc command - CRITICAL: no change threshold applied
            cmd = f"tc qdisc change dev {CLIENT_VETH} root handle 1: cake bandwidth {bandwidth_bps}bit"
            self.run_command(cmd, CLIENT_NS)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating CAKE bandwidth: {e}")
            return False

class PurpleAIMD:
    """
    Enhanced Adapter for the PURPLE-AIMD Controller with stability analysis
    """
    
    def __init__(self, 
                 config: Optional[PurpleAIMDConfig] = None,
                 log_file: str = DEFAULT_LOG_FILE,
                 trajectory_file: str = DEFAULT_TRAJECTORY_FILE,
                 sample_interval: float = DEFAULT_SAMPLE_INTERVAL):
        
        # Initialize network monitor
        self.network_monitor = NetworkMonitor()
        
        # Controller configuration
        self.config = config or PurpleAIMDConfig()
        self.sample_interval = sample_interval
        self.log_file = log_file
        self.trajectory_file = trajectory_file
        
        # Initialise controller
        if not controller_import_success:
            raise ImportError("Could not import PURPLE-AIMD Controller module")
        
        self.controller = PurpleAIMDController(self.config)
        
        # Internal state
        self.running = False
        self.latency = 5.0
        self.bandwidth_rate = self.config.initial_rate
        self.objective = 0.0
        self.trajectory_data = []
        
        # Setup logging
        self.logger = logging.getLogger('PURPLE-AIMD')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Setup CSV logging
        self.setup_logging()
        
        # Verify stability of controller parameters
        self.verify_stability()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def setup_logging(self):
        """Setup CSV logging for controller data"""
        file_exists = os.path.exists(self.log_file)
        
        self.log_file_handle = open(self.log_file, 'a', newline='')
        self.csv_writer = csv.writer(self.log_file_handle)
        
        if not file_exists:
            self.csv_writer.writerow([
                'timestamp', 'latency', 'bandwidth_rate', 'objective',
                'good_latency_count', 'latency_trend', 'recovery_phase',
                'feedback_state', 'stability_metric'
            ])
    
    def verify_stability(self):
        """Verify the stability of controller parameters using Chiu & Jain criteria"""
        if not stability_analyzer_available:
            self.logger.warning("Stability analysis not available, skipping verification")
            return
        
        # For AIMD, the key parameters are:
        # - additive_increase (recovery_rate in config)
        # - multiplicative_decrease (decay_factor in config)
        
        is_stable, message = AIMDStabilityAnalyser.verify_stability_parameters(
            additive_increase=self.config.recovery_rate,
            multiplicative_decrease=self.config.decay_factor,
            min_rate=self.config.min_rate,
            max_rate=self.config.max_rate
        )
        
        # Log the stability analysis
        for line in message.split('\n'):
            if line.startswith('UNSTABLE'):
                self.logger.error(line)
            elif line.startswith('STABLE'):
                self.logger.info(line)
            else:
                self.logger.info(line)
        
        if not is_stable:
            self.logger.warning("Controller parameters may lead to unstable behavior!")
    
    def signal_handler(self, sig, frame):
        """Handle termination signals"""
        print(f"Received signal {sig}, shutting down...")
        self.running = False
        
        # Save trajectory data
        if self.trajectory_data:
            with open(self.trajectory_file, 'w') as f:
                json.dump(self.trajectory_data, f, indent=2)
            print(f"Saved trajectory data to {self.trajectory_file}")
            
            # Generate trajectory plot if stability analyser is available
            if stability_analyzer_available:
                try:
                    plot = AIMDStabilityAnalyser.plot_trajectory(
                        self.trajectory_data, 
                        title="PURPLE-AIMD Controller AIMD Trajectory"
                    )
                    plot_file = os.path.splitext(self.trajectory_file)[0] + '.png'
                    plot.savefig(plot_file)
                    print(f"Saved trajectory plot to {plot_file}")
                except Exception as e:
                    print(f"Error generating trajectory plot: {e}")
                    print("Consider using the enhanced visualization script instead.")
            else:
                # Suggest using our enhanced visualization script
                print("Stability analyzer not available for plotting.")
                print("Use the enhanced visualization script to generate plots from trajectory data:")
                print(f"python3 enhanced_visualization.py --data {self.trajectory_file}")
        
        if hasattr(self, 'log_file_handle') and self.log_file_handle:
            self.log_file_handle.close()
        sys.exit(0)
        
    def update_controller(self):
        """Update controller with latest latency measurement"""
        # Get current time - Unix timestamp
        current_time = time.time()
        
        # Measure latency - ONLY measurement we need!
        self.latency = self.network_monitor.measure_latency()
        
        # Record previous rate for trajectory analysis
        prev_rate = self.bandwidth_rate
        
        # Update controller with latency
        self.bandwidth_rate, self.objective = self.controller.update(self.latency, current_time)
        
        # Get diagnostics
        diag = self.controller.get_diagnostics()
        
        # Determine if this is a decrease event (feedback = 1)
        feedback = 1 if self.bandwidth_rate < prev_rate else 0
        
        # Calculate stability metric (ratio of current rate to optimal rate)
        # For simplicity, we're using target latency as our proxy for optimal operating point
        stability_metric = abs(self.latency - self.config.target_latency) / self.config.target_latency
        
        # Record trajectory data for convergence analysis
        # Note: Using Unix timestamp (seconds since epoch) for compatibility with our visualization script
        self.trajectory_data.append({
            'timestamp': current_time,
            'rate': self.bandwidth_rate,
            'latency': self.latency,
            'feedback': feedback,
            'md_factor': self.config.decay_factor,
            'ai_factor': self.config.recovery_rate,
            'good_latency_count': diag.get('good_latency_count', 0),
            'recovery_phase': diag.get('recovery_phase', False),
            'objective': self.config.target_latency  # Include target for reference
        })
        
        # Update CAKE bandwidth with new rate
        success = self.network_monitor.update_cake_bandwidth(self.bandwidth_rate)
        
        # Log to CSV - use ISO format for timestamps in the CSV for readability
        self.log_stats(diag, feedback, stability_metric)
        
        return success, diag
    
    def log_stats(self, diag, feedback, stability_metric):
        """Log statistics to CSV file"""
        if hasattr(self, 'csv_writer'):
            self.csv_writer.writerow([
                datetime.now().isoformat(),  # Human-readable timestamp for CSV
                self.latency,
                self.bandwidth_rate,
                self.objective,
                diag.get('good_latency_count', 0),
                diag.get('latency_trend', 0.0),
                diag.get('recovery_phase', False),
                feedback,
                stability_metric
            ])
            self.log_file_handle.flush()
    
    def run(self):
        """Main loop to run the controller"""
        self.running = True
        
        print(f"Starting PURPLE-AIMDLatency Controller with target latency {self.config.target_latency}ms")
        print(f"Sample interval: {self.sample_interval}s, Log file: {self.log_file}")
        print(f"AIMD Parameters: AI={self.config.recovery_rate}, MD={self.config.decay_factor}")
        print(f"Trajectory file will be saved to: {self.trajectory_file}")
        
        try:
            last_log_time = 0
            while self.running:
                # Update controller with latest latency
                success, diag = self.update_controller()
                
                # Calculate stability metrics based on Chiu & Jain
                feedback = 1 if self.latency > self.config.target_latency else 0
                
                # Log status regularly
                current_time = time.time()
                if current_time - last_log_time >= 2:  # Every 2 seconds
                    last_log_time = current_time
                    
                    state = "RECOVERY" if diag.get('recovery_phase', False) else "NORMAL"
                    if self.latency > self.config.critical_latency:
                        state = "CRITICAL"
                    elif self.latency > self.config.target_latency + self.config.latency_tolerance:
                        state = "HIGH"
                    
                    # Calculate oscillation magnitude based on recent data points
                    if len(self.trajectory_data) > 10:
                        recent_rates = [point['rate'] for point in self.trajectory_data[-10:]]
                        oscillation = max(recent_rates) - min(recent_rates)
                    else:
                        oscillation = 0
                    
                    self.logger.info(
                        f"Latency: {self.latency:.1f}ms, Rate: {self.bandwidth_rate:.1f}Mbps, "
                        f"State: {state}, Trend: {diag.get('latency_trend', 0.0):.3f}, "
                        f"Feedback: {feedback}, Oscillation: {oscillation:.2f}Mbps"
                    )
                
                # Sleep for sample interval
                time.sleep(self.sample_interval)
                
        except KeyboardInterrupt:
            print("Interrupted by user")
        finally:
            # Make sure trajectory data is saved even on unexpected exit
            if self.trajectory_data:
                try:
                    with open(self.trajectory_file, 'w') as f:
                        json.dump(self.trajectory_data, f, indent=2)
                    print(f"Saved trajectory data to {self.trajectory_file}")
                    print("Use the enhanced visualization script to analyze the trajectory:")
                    print(f"python3 enhanced_visualization.py --data {self.trajectory_file}")
                except Exception as e:
                    print(f"Error saving trajectory data: {e}")
            
            if hasattr(self, 'log_file_handle') and self.log_file_handle:
                self.log_file_handle.close()
            print("Controller stopped")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PURPLE-AIMDLatency Controller Adapter')
    parser.add_argument('--interval', type=float, default=DEFAULT_SAMPLE_INTERVAL,
                        help=f'Sample interval in seconds (default: {DEFAULT_SAMPLE_INTERVAL})')
    parser.add_argument('--logfile', default=DEFAULT_LOG_FILE,
                        help=f'Log file path (default: {DEFAULT_LOG_FILE})')
    parser.add_argument('--trajectory', default=DEFAULT_TRAJECTORY_FILE,
                        help=f'Trajectory data file (default: {DEFAULT_TRAJECTORY_FILE})')
    parser.add_argument('--target', type=float, default=5.5,
                        help='Target latency in ms (default: 5.5)')
    parser.add_argument('--min-rate', type=float, default=10.0,
                        help='Minimum bandwidth rate in Mbps (default: 10.0)')
    parser.add_argument('--max-rate', type=float, default=120.0,
                        help='Maximum bandwidth rate in Mbps (default: 120.0)')
    parser.add_argument('--initial-rate', type=float, default=95.0,
                        help='Initial bandwidth rate in Mbps (default: 95.0)')
    parser.add_argument('--decay', type=float, default=0.05,
                        help='Decay factor for high latency (default: 0.05)')
    parser.add_argument('--recovery', type=float, default=0.25,
                        help='Recovery rate in Mbps per second (default: 1)')
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='Latency tolerance in ms (default: 2.0)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Check if we're running in the correct environment
    if not os.path.exists(f"/var/run/netns/{CLIENT_NS}"):
        print(f"Error: Network namespace {CLIENT_NS} not found.")
        print("Please run 'sudo ./simple_bottleneck_emulator.sh setup' first.")
        sys.exit(1)
    
    # Create controller configuration
    config = PurpleAIMDConfig(
        target_latency=args.target,
        min_rate=args.min_rate,
        max_rate=args.max_rate,
        initial_rate=args.initial_rate,
        decay_factor=args.decay,
        recovery_rate=args.recovery,
        latency_tolerance=args.tolerance
    )
    
    # Create and run adapter
    adapter = PurpleAIMD(
        config=config,
        log_file=args.logfile,
        trajectory_file=args.trajectory,
        sample_interval=args.interval
    )
    
    adapter.run()

if __name__ == "__main__":
    main()