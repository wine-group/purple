#!/usr/bin/env python3
"""
PURPLE-AIMD Latency Controller Adapter for Ubiquiti Radios

This script adapts the PURPLE-AIMD Controller to work with Ubiquiti wireless radios.
It collects radio statistics via the Web API and SSH, and uses latency measurements
to control CAKE bandwidth settings on a gateway system.

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
import requests
import paramiko
import pytz
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
import urllib3

# Disable HTTPS verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Import our controller - adjust path as needed
sys.path.append('.')
try:
    from purple_aimd_controller import PurpleAIMDController, PurpleAIMDConfig
    controller_import_success = True
except ImportError:
    controller_import_success = False
    print("Warning: Could not import PURPLE-AIMD Controller. Make sure it's in the current directory.")

# Import our stability analyser
try:
    from stability_analysis import AIMDStabilityAnalyser
    stability_analyzer_available = True
except ImportError:
    stability_analyzer_available = False
    print("Warning: Could not import AIMM Stability Analyser. Some features will be disabled.")

# Constants
DEFAULT_SAMPLE_INTERVAL = 0.1  # 1 second by default, not 0.1s for testing
DEFAULT_LOG_FILE = "radio_stats.csv"
DEFAULT_TRAJECTORY_FILE = "aimd_trajectory.json"
SSH_TIMEOUT = 10  # seconds


@dataclass
class RadioStats:
    """Data class to hold radio statistics"""
    timestamp: datetime
    tx_rate: float
    rx_rate: float
    radio_capacity: float
    backlog: int
    drops: int
    useful_capacity: float
    state: str
    latency: float = 0.0


class RadioMonitor:
    """Monitor radio statistics using Web API and SSH"""
    
    def __init__(self, 
                 radio_hostname: str,
                 radio_web_username: str,
                 radio_web_password: str,
                 radio_ssh_username: str,
                 radio_ssh_password: str = None,
                 radio_ssh_key: str = None,
                 radio_interface: str = "ath0",
                 gateway_hostname: str = None,
                 gateway_username: str = None,
                 gateway_password: str = None,
                 gateway_key_path: str = None,
                 gateway_interface: str = "eth0",
                 ping_target: str = "10.0.10.250", # Could be a public host too, but be careful.
                 radio_ssh_port: int = 22,
                 gateway_ssh_port: int = 22,
                 csv_path: Optional[str] = None):
        
        # Radio parameters
        self.radio_hostname = radio_hostname
        self.radio_interface = radio_interface
        self.radio_ssh_port = radio_ssh_port
        
        # Gateway parameters
        self.gateway_hostname = gateway_hostname
        self.gateway_username = gateway_username
        self.gateway_password = gateway_password
        self.gateway_key_path = gateway_key_path
        self.gateway_interface = gateway_interface
        self.gateway_ssh_port = gateway_ssh_port
        
        # Web API setup
        self.base_url = f"https://{radio_hostname}"
        self.web_username = radio_web_username
        self.web_password = radio_web_password
        self.session = requests.Session()
        
        # SSH setup for radio
        self.ssh_username = radio_ssh_username
        self.ssh_password = radio_ssh_password
        self.ssh_key = radio_ssh_key
        self.radio_ssh_client = None
        
        # SSH setup for gateway
        self.gateway_ssh_client = None
        
        # Ping target for latency measurement
        self.ping_target = ping_target
        
        # CSV logging setup
        self.csv_path = csv_path
        if csv_path:
            self._setup_csv_logging()
        
        # Rate limiting for TC updates
        self.last_tc_update = 0.0
        self.tc_update_interval = 1.0
        
        # Setup logging
        self.logger = logging.getLogger('RadioMonitor')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Connect to radio on initialization
        self._web_login()
        self._connect_radio_ssh()
        
        # Connect to gateway if provided
        if self.gateway_hostname:
            self._connect_gateway_ssh()
    
    def _web_login(self) -> None:
        """Login to radio web interface"""
        try:
            login_url = f"{self.base_url}/api/auth"
            data = {
                'username': self.web_username,
                'password': self.web_password
            }
            
            response = self.session.post(
                login_url,
                data=data,
                verify=False,
                timeout=10
            )
            
            if response.status_code == 200:
                self.logger.info("Successfully logged into radio web interface")
            else:
                raise Exception(f"Web login failed with status code: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Web login failed: {str(e)}")
            raise

    def _connect_radio_ssh(self) -> None:
        """Establish SSH connection with radio"""
        try:
            self.radio_ssh_client = paramiko.SSHClient()
            self.radio_ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_args = {
                'hostname': self.radio_hostname,
                'username': self.ssh_username,
                'port': self.radio_ssh_port,
                'timeout': SSH_TIMEOUT,
                'allow_agent': False,
                'look_for_keys': False
            }
            
            # Add authentication method
            if self.ssh_key:
                connect_args['key_filename'] = self.ssh_key
            else:
                connect_args['password'] = self.ssh_password
            
            # Some Ubiquiti devices have compatibility issues with newer SSH algorithms
            connect_args['disabled_algorithms'] = {'keys': ['rsa-sha2-256', 'rsa-sha2-512']}
            
            self.radio_ssh_client.connect(**connect_args)
            
            self.logger.info(f"Successfully connected to radio via SSH")
            
        except Exception as e:
            self.logger.error(f"Radio SSH connection failed: {str(e)}")
            raise
    
    def _connect_gateway_ssh(self) -> None:
        """Establish SSH connection with gateway system"""
        if not self.gateway_hostname:
            return
            
        try:
            self.gateway_ssh_client = paramiko.SSHClient()
            self.gateway_ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            connect_args = {
                'hostname': self.gateway_hostname,
                'username': self.gateway_username,
                'port': self.gateway_ssh_port,
                'timeout': SSH_TIMEOUT,
                'allow_agent': False,
                'look_for_keys': False
            }
            
            # Add authentication method
            if self.gateway_key_path:
                connect_args['key_filename'] = self.gateway_key_path
            else:
                connect_args['password'] = self.gateway_password
            
            self.gateway_ssh_client.connect(**connect_args)
            
            self.logger.info(f"Successfully connected to gateway via SSH")
            
        except Exception as e:
            self.logger.error(f"Gateway SSH connection failed: {str(e)}")
            # Don't raise here, so we can still function without gateway connection
            # Just log the error and continue
    
    def _get_web_status(self) -> Dict[str, Any]:
        """Get status from radio web API"""
        try:
            timestamp = int(time.time() * 1000)
            status_url = f"{self.base_url}/status.cgi?_={timestamp}"
            
            response = self.session.get(
                status_url,
                verify=False,
                timeout=10
            )
            
            if response.status_code != 200:
                raise Exception(f"Status request failed with code: {response.status_code}")
            
            return response.json()
            
        except Exception as e:
            self.logger.error(f"Failed to get web status: {str(e)}")
            return {}
    
    def _get_queue_stats(self) -> Tuple[int, int]:
        """Get queue backlog and drops via SSH"""
        try:
            cmd = f"tc -s qdisc show dev {self.radio_interface}"
            stdin, stdout, stderr = self.radio_ssh_client.exec_command(cmd)
            
            output = stdout.read().decode()
            errors = stderr.read().decode()
            
            if errors:
                self.logger.warning(f"Errors while getting queue stats: {errors}")
            
            # Parse backlog
            backlog_match = re.search(r"backlog\s+(\d+)b", output)
            backlog = int(backlog_match.group(1)) if backlog_match else 0
            
            # Parse drops
            drops_match = re.search(r"dropped\s+(\d+)", output)
            drops = int(drops_match.group(1)) if drops_match else 0
            
            return backlog, drops
            
        except Exception as e:
            self.logger.error(f"Error getting queue stats: {str(e)}")
            return 0, 0
    
    def _parse_web_stats(self, status: Dict[str, Any]) -> Tuple[float, float, float]:
        """Parse radio statistics from web API response"""
        try:
            wireless = status.get('wireless', {})
            throughput = wireless.get('throughput', {})
            polling = wireless.get('polling', {})
            
            tx_rate = float(throughput.get('tx', 0) / 1_000)
            rx_rate = float(throughput.get('rx', 0) / 1_000)
            radio_capacity = float(polling.get('dl_capacity', 0) / 1_000)
            
            return tx_rate, rx_rate, radio_capacity
            
        except Exception as e:
            self.logger.error(f"Error parsing web stats: {str(e)}")
            return 0.0, 0.0, 100.0  # Default to 100 Mbps capacity if parsing fails
    
    def _setup_csv_logging(self) -> None:
        """Setup CSV logging with headers"""
        if self.csv_path:
            csv_file = Path(self.csv_path)
            file_exists = csv_file.exists()
            
            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        'timestamp', 'tx_rate', 'rx_rate', 'radio_capacity',
                        'backlog', 'drops', 'useful_capacity', 'state', 'latency'
                    ])
    
    def _log_stats(self, stats: RadioStats) -> None:
        """Log statistics to CSV file"""
        if self.csv_path:
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    stats.timestamp.isoformat(),
                    stats.tx_rate,
                    stats.rx_rate,
                    stats.radio_capacity,
                    stats.backlog,
                    stats.drops,
                    stats.useful_capacity,
                    stats.state,
                    stats.latency
                ])
    
    def measure_latency(self) -> float:
        """Measure ping latency to target"""
        try:
            if self.radio_ssh_client:
                # Use radio for latency measurement
                # cmd = f"ping -c 3 -q {self.ping_target}"
                # stdin, stdout, stderr = self.radio_ssh_client.exec_command(cmd)
                # output = stdout.read().decode()

                result = subprocess.run(['ping', '-c', '1', '10.0.10.250'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if result.returncode == 0:
                    output = result.stdout.decode()
                    ping_time = output.split('time=')[-1].split(' ms')[0]
                    return float(ping_time)
                else:
                    return None
                
                # # Parse RTT statistics
                # rtt_match = re.search(r'rtt min/avg/max/mdev = (\d+\.?\d*)/(\d+\.?\d*)/(\d+\.?\d*)/(\d+\.?\d*)', output)
                # if rtt_match:
                #     # Use the average value (second group)
                #     latency = float(rtt_match.group(2))
                #     return latency
                
                # # Alternative parsing for different ping output formats
                # alt_match = re.search(r'min/avg/max\s*=\s*(\d+\.?\d*)/(\d+\.?\d*)/(\d+\.?\d*)', output)
                # if alt_match:
                #     latency = float(alt_match.group(2))
                #     return latency
            
            # Default fallback
            self.logger.warning("Could not measure latency, using default value")
            return 10.0
                
        except Exception as e:
            self.logger.error(f"Error measuring latency: {e}")
            return 10.0  # Default fallback
    
    def update_cake_bandwidth(self, bandwidth_mbps: float) -> bool:
        """Update CAKE bandwidth on gateway interface"""
        if not self.gateway_ssh_client:
            self.logger.warning("No gateway SSH connection, can't update CAKE bandwidth")
            return False
            
        try:
            current_time = time.time()
            
            # Only update if enough time has passed since last update
            if current_time - self.last_tc_update >= self.tc_update_interval:
                # Get current rate by executing tc command
                cmd = f"tc qdisc show dev {self.gateway_interface} | grep cake"
                stdin, stdout, stderr = self.gateway_ssh_client.exec_command(cmd)
                output = stdout.read().decode().strip()
                
                # Extract current rate in Mbps
                current_rate_mbps = 0.0
                if 'bandwidth' in output:
                    rate_match = re.search(r'bandwidth (\d+)([kmg])?bit', output, re.IGNORECASE)
                    if rate_match:
                        value = float(rate_match.group(1))
                        unit = rate_match.group(2).lower() if rate_match.group(2) else ''
                        
                        # Convert to Mbps
                        if unit == 'k':
                            current_rate_mbps = value / 1000
                        elif unit == 'g':
                            current_rate_mbps = value * 1000
                        else:  # mbit or bit
                            current_rate_mbps = value / 1000000 if not unit else value

                # Calculate absolute and percentage differences
                abs_diff = abs(bandwidth_mbps - current_rate_mbps)
                percent_diff = (abs_diff / current_rate_mbps) * 100 if current_rate_mbps > 0 else float('inf')
                
                # Only update if difference exceeds thresholds
                if abs_diff >= 2.5 or percent_diff >= 1.0:
                    # Convert Mbps to bps for tc
                    bandwidth_bps = int(bandwidth_mbps * 1_000_000)
                    
                    # Build tc command based on existing configuration
                    if 'handle' in output:
                        # Extract handle
                        handle_match = re.search(r'handle (\w+):', output)
                        handle = handle_match.group(1) if handle_match else "8001"
                        cmd = f"sudo tc qdisc change dev {self.gateway_interface} handle {handle}: cake bandwidth {bandwidth_bps}bit"
                    else:
                        # No existing config or handle not found, use replace
                        cmd = f"sudo tc qdisc replace dev {self.gateway_interface} root cake bandwidth {bandwidth_bps}bit"
                    
                    # Execute command
                    stdin, stdout, stderr = self.gateway_ssh_client.exec_command(cmd)
                    
                    # Check for errors
                    error_msg = stderr.read().decode().strip()
                    if error_msg:
                        self.logger.warning(f"TC command warning: {error_msg}")
                    
                    self.logger.info(f"Updated CAKE bandwidth from {current_rate_mbps:.2f} to {bandwidth_mbps:.2f} Mbps (diff: {abs_diff:.2f} Mbps, {percent_diff:.1f}%)")
                    self.last_tc_update = current_time
                else:
                    self.logger.debug(f"Skipping TC update - difference too small (abs: {abs_diff:.2f} Mbps, {percent_diff:.1f}%)")
                
            return True
                
        except Exception as e:
            self.logger.error(f"Error updating CAKE bandwidth: {str(e)}")
            return False
    
    def get_stats(self, useful_capacity: float = None) -> RadioStats:
        """
        Get all radio statistics
        
        Args:
            useful_capacity: Current useful capacity value (if None, use radio capacity)
            
        Returns:
            RadioStats object with current stats
        """
        try:
            # Get current timestamp
            timestamp = datetime.now(pytz.UTC)
            
            # Get web API stats
            status = self._get_web_status()
            tx_rate, rx_rate, radio_capacity = self._parse_web_stats(status)
            
            # Get queue stats via SSH
            backlog, drops = self._get_queue_stats()
            
            # Measure latency
            latency = self.measure_latency()
            
            # Create stats object
            stats = RadioStats(
                timestamp=timestamp,
                tx_rate=tx_rate,
                rx_rate=rx_rate,
                radio_capacity=radio_capacity,
                backlog=backlog,
                drops=drops,
                useful_capacity=useful_capacity if useful_capacity is not None else radio_capacity,
                state="NORMAL",  # Will be updated by controller
                latency=latency
            )
            
            # Log stats to CSV if useful_capacity is provided
            # This prevents logging during initial data collection
            if useful_capacity is not None:
                self._log_stats(stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting radio stats: {str(e)}")
            # Return default stats with timestamp in case of error
            return RadioStats(
                timestamp=datetime.now(pytz.UTC),
                tx_rate=0.0,
                rx_rate=0.0,
                radio_capacity=100.0,
                backlog=0,
                drops=0,
                useful_capacity=useful_capacity if useful_capacity is not None else 100.0,
                state="ERROR",
                latency=10.0
            )
    
    def close(self) -> None:
        """Close all connections"""
        try:
            if self.radio_ssh_client:
                self.radio_ssh_client.close()
                self.logger.info("Radio SSH connection closed")
            
            if self.gateway_ssh_client:
                self.gateway_ssh_client.close()
                self.logger.info("Gateway SSH connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing connections: {str(e)}")


class PurpleAIMD:
    """
    Enhanced Adapter for the PURPLE-AIMD Controller with Ubiquiti radio support
    """
    
    def __init__(self, 
                 config: Optional[PurpleAIMDConfig] = None,
                 radio_monitor: Optional[RadioMonitor] = None,
                 log_file: str = DEFAULT_LOG_FILE,
                 trajectory_file: str = DEFAULT_TRAJECTORY_FILE,
                 sample_interval: float = DEFAULT_SAMPLE_INTERVAL):
        
        # Initialize radio monitor
        if radio_monitor is None:
            raise ValueError("RadioMonitor instance must be provided")
        
        self.radio_monitor = radio_monitor
        
        # Controller configuration
        self.config = config or PurpleAIMDConfig()
        self.sample_interval = sample_interval
        self.log_file = log_file
        self.trajectory_file = trajectory_file
        
        # Initialize controller
        if not controller_import_success:
            raise ImportError("Could not import PURPLE-AIMD Controller module")
        
        self.controller = PurpleAIMDController(self.config)
        
        # Internal state
        self.running = False
        self.latency = 10.0
        self.bandwidth_rate = self.config.initial_rate
        self.objective = 0.0
        self.trajectory_data = []
        
        # Radio statistics
        self.tx_rate = 0.0
        self.rx_rate = 0.0
        self.radio_capacity = 100.0
        self.backlog = 0
        self.drops = 0
        self.last_drops = 0
        self.state = "NORMAL"
        
        # Setup logging
        self.logger = logging.getLogger('PURPLE-AIMD')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
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
                    print("Consider using the visualisation script instead.")
            else:
                # Suggest using our visualisation script
                print("Stability analyser not available for plotting.")
                print("Use the visualisation script to generate plots from trajectory data:")
                print(f"python3 visualisation.py --data {self.trajectory_file}")
            
        # Close radio monitor connections
        if hasattr(self.radio_monitor, 'close'):
            self.radio_monitor.close()
            
        sys.exit(0)
        
    def update_controller(self):
        """Update controller with latest measurements"""
        # Get current time - Unix timestamp
        current_time = time.time()
        
        # Get radio statistics
        radio_stats = self.radio_monitor.get_stats(self.bandwidth_rate)
        
        # Update internal state
        self.latency = radio_stats.latency
        self.tx_rate = radio_stats.tx_rate
        self.rx_rate = radio_stats.rx_rate
        self.radio_capacity = radio_stats.radio_capacity
        self.backlog = radio_stats.backlog
        self.drops = radio_stats.drops
        
        # Calculate delta drops
        delta_drops = self.drops - self.last_drops
        self.last_drops = self.drops
        
        # Record previous rate for trajectory analysis
        prev_rate = self.bandwidth_rate
        
        # Update controller with latency
        self.bandwidth_rate, self.objective = self.controller.update(self.latency, current_time)
        
        # Get diagnostics
        diag = self.controller.get_diagnostics()
        
        # Determine if this is a decrease event (feedback = 1)
        feedback = 1 if self.bandwidth_rate < prev_rate else 0
        
        # Update state for logging
        if diag.get('recovery_phase', False):
            self.state = "RECOVERY"
        elif self.latency > self.config.critical_latency:
            self.state = "CRITICAL"
        elif self.latency > self.config.target_latency + self.config.latency_tolerance:
            self.state = "HIGH"
        else:
            self.state = "NORMAL"
            
        # Update radio_stats with latest state and useful_capacity
        radio_stats.state = self.state
        radio_stats.useful_capacity = self.bandwidth_rate
        
        # Log stats to CSV
        self.radio_monitor._log_stats(radio_stats)
        
        # Calculate stability metric (ratio of current rate to optimal rate)
        stability_metric = abs(self.latency - self.config.target_latency) / self.config.target_latency
        
        # Record trajectory data for convergence analysis
        self.trajectory_data.append({
            'timestamp': current_time,
            'rate': self.bandwidth_rate,
            'latency': self.latency,
            'tx_rate': self.tx_rate,
            'rx_rate': self.rx_rate,
            'radio_capacity': self.radio_capacity,
            'backlog': self.backlog,
            'drops': self.drops,
            'delta_drops': delta_drops,
            'feedback': feedback,
            'md_factor': self.config.decay_factor,
            'ai_factor': self.config.recovery_rate,
            'good_latency_count': diag.get('good_latency_count', 0),
            'recovery_phase': diag.get('recovery_phase', False),
            'objective': self.config.target_latency,
            'state': self.state
        })
        
        # Update CAKE bandwidth with new rate
        success = self.radio_monitor.update_cake_bandwidth(self.bandwidth_rate)
        
        return success, diag
    
    def run(self):
        """Main loop to run the controller"""
        self.running = True
        
        print(f"Starting PURPLE-AIMD Controller with target latency {self.config.target_latency}ms")
        print(f"Sample interval: {self.sample_interval}s, Log file: {self.log_file}")
        print(f"AIMD Parameters: AI={self.config.recovery_rate}, MD={self.config.decay_factor}")
        print(f"Trajectory file will be saved to: {self.trajectory_file}")
        
        try:
            last_log_time = 0
            while self.running:
                # Update controller with latest measurements
                success, diag = self.update_controller()
                
                # Log status regularly
                current_time = time.time()
                if current_time - last_log_time >= 2:  # Every 2 seconds
                    last_log_time = current_time
                    
                    # Calculate oscillation magnitude based on recent data points
                    if len(self.trajectory_data) > 10:
                        recent_rates = [point['rate'] for point in self.trajectory_data[-10:]]
                        oscillation = max(recent_rates) - min(recent_rates)
                    else:
                        oscillation = 0
                    
                    self.logger.info(
                        f"Latency: {self.latency:.1f}ms, Rate: {self.bandwidth_rate:.1f}Mbps, "
                        f"TX: {self.tx_rate:.1f}Mbps, RX: {self.rx_rate:.1f}Mbps, Capacity: {self.radio_capacity:.1f}Mbps, "
                        f"State: {self.state}, Backlog: {self.backlog}, Drops: {self.drops}, "
                        f"Oscillation: {oscillation:.2f}Mbps"
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
                    print("Use the visualisation script to analyse the trajectory:")
                    print(f"python3 visualisation.py --data {self.trajectory_file}")
                except Exception as e:
                    print(f"Error saving trajectory data: {e}")
            
            # Close radio monitor connections
            if hasattr(self.radio_monitor, 'close'):
                self.radio_monitor.close()
                
            print("Controller stopped")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='PURPLE-AIMD Controller Adapter for Ubiquiti Radios')
    
    # Controller parameters
    parser.add_argument('--interval', type=float, default=DEFAULT_SAMPLE_INTERVAL,
                        help=f'Sample interval in seconds (default: {DEFAULT_SAMPLE_INTERVAL})')
    parser.add_argument('--logfile', default=DEFAULT_LOG_FILE,
                        help=f'Log file path (default: {DEFAULT_LOG_FILE})')
    parser.add_argument('--trajectory', default=DEFAULT_TRAJECTORY_FILE,
                        help=f'Trajectory data file (default: {DEFAULT_TRAJECTORY_FILE})')
    parser.add_argument('--target', type=float, default=10.0,
                        help='Target latency in ms (default: 10.0)')
    parser.add_argument('--min-rate', type=float, default=10.0,
                        help='Minimum bandwidth rate in Mbps (default: 10.0)')
    parser.add_argument('--max-rate', type=float, default=100.0,
                        help='Maximum bandwidth rate in Mbps (default: 100.0)')
    parser.add_argument('--initial-rate', type=float, default=50.0,
                        help='Initial bandwidth rate in Mbps (default: 50.0)')
    parser.add_argument('--decay', type=float, default=0.05,
                        help='Decay factor for high latency (default: 0.05)')
    parser.add_argument('--recovery', type=float, default=0.25,
                        help='Recovery rate in Mbps per second (default: 0.25)')
    parser.add_argument('--tolerance', type=float, default=2.0,
                        help='Latency tolerance in ms (default: 2.0)')
    
    # Radio connection parameters
    parser.add_argument('--radio-host', type=str, required=True,
                        help='Radio hostname or IP address')
    parser.add_argument('--radio-web-user', type=str, required=True,
                        help='Radio web interface username')
    parser.add_argument('--radio-web-password', type=str, required=True,
                        help='Radio web interface password')
    parser.add_argument('--radio-ssh-user', type=str, required=True,
                        help='Radio SSH username')
    parser.add_argument('--radio-ssh-password', type=str,
                        help='Radio SSH password (optional if using key)')
    parser.add_argument('--radio-ssh-key', type=str,
                        help='Path to Radio SSH private key (optional if using password)')
    parser.add_argument('--radio-interface', type=str, default='ath0',
                        help='Radio network interface (default: ath0)')
    parser.add_argument('--radio-ssh-port', type=int, default=22,
                        help='Radio SSH port (default: 22)')
    
    # Gateway connection parameters
    parser.add_argument('--gateway-host', type=str,
                        help='Gateway hostname or IP address (optional)')
    parser.add_argument('--gateway-user', type=str,
                        help='Gateway SSH username (required if gateway-host is provided)')
    parser.add_argument('--gateway-password', type=str,
                        help='Gateway SSH password (optional if using key)')
    parser.add_argument('--gateway-key', type=str,
                        help='Path to Gateway SSH private key (optional if using password)')
    parser.add_argument('--gateway-interface', type=str, default='eth0',
                        help='Gateway network interface (default: eth0)')
    parser.add_argument('--gateway-ssh-port', type=int, default=22,
                        help='Gateway SSH port (default: 22)')
    
    # Other parameters
    parser.add_argument('--ping-target', type=str, default='10.0.10.250',
                        help='Target IP address for latency measurement (default: 8.8.8.8)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Validate parameters
    if args.gateway_host and not args.gateway_user:
        print("Error: Gateway username is required when gateway hostname is provided")
        sys.exit(1)
        
    if args.gateway_host and not (args.gateway_password or args.gateway_key):
        print("Error: Either gateway password or key is required when gateway hostname is provided")
        sys.exit(1)
    
    # Create radio monitor
    try:
        radio_monitor = RadioMonitor(
            radio_hostname=args.radio_host,
            radio_web_username=args.radio_web_user,
            radio_web_password=args.radio_web_password,
            radio_ssh_username=args.radio_ssh_user,
            radio_ssh_password=args.radio_ssh_password,
            radio_ssh_key=args.radio_ssh_key,
            radio_interface=args.radio_interface,
            radio_ssh_port=args.radio_ssh_port,
            gateway_hostname=args.gateway_host,
            gateway_username=args.gateway_user,
            gateway_password=args.gateway_password,
            gateway_key_path=args.gateway_key,
            gateway_interface=args.gateway_interface,
            gateway_ssh_port=args.gateway_ssh_port,
            ping_target=args.ping_target,
            csv_path=args.logfile
        )
    except Exception as e:
        print(f"Error creating radio monitor: {str(e)}")
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
    try:
        adapter = PurpleAIMD(
            config=config,
            radio_monitor=radio_monitor,
            log_file=args.logfile,
            trajectory_file=args.trajectory,
            sample_interval=args.interval
        )
        
        adapter.run()
    except Exception as e:
        print(f"Error running adapter: {str(e)}")
        if hasattr(radio_monitor, 'close'):
            radio_monitor.close()
        sys.exit(1)


if __name__ == "__main__":
    main()