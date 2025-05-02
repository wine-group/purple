import requests
import paramiko
import json
import time
import logging
from datetime import datetime
import pytz
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import urllib3
from pathlib import Path
import csv
import re
import subprocess
import numpy as np

# Disable HTTPS verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class BlueQueueMonitor:
    """Implements BLUE-inspired queue monitoring."""
    def __init__(self,
                 freeze_time: float = 0.1,        # Min time between probability increases
                 decr_freeze_time: float = 1.0,   # Min time between probability decreases  
                 increment: float = 0.15,         # Increment step for probability
                 decrement: float = 0.025,        # Decrement step for probability
                 drop_threshold: int = 10):       
        self.freeze_time = freeze_time
        self.decr_freeze_time = decr_freeze_time
        self.increment = increment
        self.decrement = decrement
        self.drop_threshold = drop_threshold
        
        # State variables
        self.marking_probability = 0.0
        self.last_update_time = 0.0
        self.last_congestion_time = 0.0
        self.last_decrement_time = 0.0
        self.last_drops = 0.0
        
    # def should_increment(self, backlog: int, drops: int) -> bool:
    #     """Determine if probability should increase based on queue state."""
    #     return backlog > 0 #or drops > 0 # needs to be delta drops/drops since last, leaving out for now

    def should_increment(self, backlog: int, drops: int) -> bool:
        """Determine if probability should increase based on queue state."""
        # Calculate delta drops since last check
        delta_drops = drops - self.last_drops
        
        # Update last_drops for next time
        self.last_drops = drops
        
        # Return true if either condition is met
        return backlog > 0 or delta_drops >= self.drop_threshold
        
    def update(self, backlog: int, drops: int, current_time: float) -> None:
        """Update marking probability based on queue state."""
        if self.should_increment(backlog, drops):
            # Only increment if enough time has passed since last increment
            if current_time - self.last_congestion_time >= self.freeze_time:
                self.marking_probability = min(1.0, self.marking_probability + self.increment)
                self.last_congestion_time = current_time
        else:
            # Only decrement if enough time has passed since last decrement
            if current_time - self.last_decrement_time >= self.decr_freeze_time:
                self.marking_probability = max(0.0, self.marking_probability - self.decrement)
                self.last_decrement_time = current_time
                
        self.last_update_time = current_time


class CapacityEstimator:
    """Capacity estimator using BLUE concepts and P95 measurements."""
    def __init__(self,
                 window_size: int = 10,            # Window size for P50 calculation. Default 60. Might be too high...
                 update_interval: float = 1.0,     # Update interval in seconds
                 freeze_time: float = 1.0,         # BLUE freeze time parameter
                 decr_freeze_time: float = 1.0,    # BLUE decrement freeze time. Default of 0.1
                 minimum_decrease: float = 0.95,   # Minimum capacity decrease factor
                 recovery_factor: float = 0.1):    # Factor for recovery to tx_capacity
        
        self.window_size = window_size
        self.update_interval = update_interval
        self.minimum_decrease = minimum_decrease
        self.recovery_factor = recovery_factor
        
        # Initialize BLUE-inspired queue monitor
        self.queue_monitor = BlueQueueMonitor(
            freeze_time=freeze_time,
            decr_freeze_time=decr_freeze_time
        )
        
        # State variables
        self.useful_capacity = None
        self.tx_rate_history = []
        self.capacity_history = []  # New: Track radio capacity history
        self.last_update_time = 0.0
        self.in_congestion = False
        
    def _calculate_p50(self, values: List[float]) -> float:
        """Calculate P50 from a list of values."""
        if not values:
            return 0.0
        
        try:
            # return float(np.percentile(values, 90))
            return float(np.median(values))
        except Exception as e:
            logging.error(f"Error calculating P50: {e}")
            return 0.0
        
    def _calculate_capacity_target(self, current_tx: float, radio_capacity: float) -> float:
        """Calculate capacity target using P50 of tx rates during congestion and radio capacity during normal operation."""
        # Keep track of histories
        self.capacity_history.append(radio_capacity)
        self.tx_rate_history.append(current_tx)
        if len(self.capacity_history) > self.window_size:
            self.capacity_history = self.capacity_history[-self.window_size:]
        if len(self.tx_rate_history) > self.window_size:
            self.tx_rate_history = self.tx_rate_history[-self.window_size:]
        
        # Calculate P50 of both histories
        capacity_p50 = self._calculate_p50(self.capacity_history)
        tx_rate_p50 = self._calculate_p50(self.tx_rate_history)

        # print("\n--- Debug ---\n")
        # print("P50 TX Capacity: {}\nP50 TX Rate" {}).format(capacity_p50, tx_rate_p50)
        # print("\n------\n")
        
        # Get current marking probability
        marking_prob = self.queue_monitor.marking_probability
        
        # Determine congestion state with hysteresis
        # Enter congestion at 0.3, but require dropping to 0.1 to exit
        if not self.in_congestion:
            self.in_congestion = marking_prob > 0.3
        else:
            self.in_congestion = marking_prob > 0.1
        
        if self.in_congestion:
            # In congestion - reduce useful capacity based on actual achieved tx rates
            reduction = marking_prob * 0.5
            # Use P50 of tx rate history as the baseline during congestion
            target = tx_rate_p50 * (1.0 - reduction)
            # Apply minimum decrease threshold
            target = max(target, current_tx * self.minimum_decrease)
        else:
            # Not in congestion - gradually approach P50 of radio capacity
            if self.useful_capacity is None:
                target = capacity_p50 * 0.80  # Start at 85% of P50
            else:
                # Recovery rate depends on distance to P50 capacity
                headroom = max(0, (capacity_p50 * 0.80) - self.useful_capacity)
                if not headroom:
                    print("NO HEADROOM")
                    target = capacity_p50 * 0.80
                else:
                    target = self.useful_capacity + (headroom * self.recovery_factor)
        
        # Never exceed P50 of radio capacity
        # Actually, it never should be able to...
        # target = min(target, capacity_p50)

        # Only apply smoothing if we're in congestion or target is higher
        # Maybe only if target is higher?... Not sure it makes sense for congestion, probably want to decrease ASAP to avoid blaot.
        # But... will it decrease too quickly and cause problems? There is some balance here.
        # if self.in_congestion or target > self.useful_capacity:

        if target > self.useful_capacity:
        # if target:
            alpha = 0.3  # Smoothing factor
            target = (alpha * target + (1 - alpha) * self.useful_capacity)
        # Removed below as this should always return useful_capacity as a higher rate would be caught by the if statement
        

        print("--- Debug ---")
        print("IN CONGESTION? {}".format(self.in_congestion))
        print("P50 TX Capacity: {}\nP50 TX Rate {}".format(capacity_p50, tx_rate_p50))
        print("USEFUL CAPACITY: {}".format(self.useful_capacity))
        if not self.in_congestion:
            print("Headroom: {}\nTarget: {}".format(headroom, target))
        print("------")

        # Example:
        # IN CONGESTION? False
        # P50 TX Capacity: 137.28
        # P50 TX Rate 100.6371
        # USEFUL CAPACITY: 92.93162138442716
        # Headroom: 44.34837861557284
        # Target: 97.36645924598444

        
        return target
        

    def _get_state(self) -> str:
        """Determine current network state based on BLUE marking probability."""
        marking_prob = self.queue_monitor.marking_probability
        if marking_prob > 0.8:
            return "SEVERE_CONGESTION"
        elif marking_prob > 0.3:
            return "MILD_CONGESTION"
        return "STABLE"
    
    def update(self, timestamp: datetime, tx_rate: float, radio_capacity: float,
            backlog: int, drops: int) -> Tuple[float, str]:
        """Update estimator state with new measurements."""
        try:
            # Convert values and handle NaN/None
            tx_rate_val = float(tx_rate) if tx_rate is not None else 0.0
            tx_capacity_val = float(radio_capacity) if radio_capacity is not None else 0.0
            backlog_val = int(backlog) if backlog is not None else 0
            drops_val = int(drops) if drops is not None else 0
            current_time = timestamp.timestamp()
            
            # Only do full update at update interval
            if current_time - self.last_update_time >= self.update_interval:
                # Update BLUE queue monitor first
                self.queue_monitor.update(backlog_val, drops_val, current_time)
                
                # Calculate new capacity target
                if self.useful_capacity is None:
                    self.useful_capacity = tx_capacity_val * 0.75  # Apply safety factor immediately. Default of 0.85 (last 15% is always bad)
                else:
                    self.useful_capacity = self._calculate_capacity_target(tx_rate_val, tx_capacity_val)

                    # Setting useful capacity without smoothing:
                    # self.useful_capacity = target * 0.95
                    # self.useful_capacity = target


                    # Testing without...
                    # Only apply smoothing if we're in congestion or target is higher
                    # if self.in_congestion or target > self.useful_capacity:
                    #     alpha = 0.3  # Smoothing factor
                    #     self.useful_capacity = (alpha * target + (1 - alpha) * self.useful_capacity)
                    # Removed below as this should always return useful_capacity as a higher rate would be caught by the if statement
                    # else:
                    #     # In stable state, maintain current capacity unless target is higher
                    #     self.useful_capacity = max(self.useful_capacity, target)
                
                self.last_update_time = current_time
            
            return self.useful_capacity, self._get_state()
            
        except Exception as e:
            logging.error(f"Error in capacity estimator update: {str(e)}")
            return tx_capacity_val, "STABLE"

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

class RadioMonitor:
    """Monitor radio statistics using Web API and SSH with capacity estimation"""
    def __init__(self, 
                 hostname: str,
                 web_username: str,
                 web_password: str,
                 ssh_username: str,
                 ssh_password: str,
                 interface: str = "ath0",
                 local_interface: str = "eth0",
                 ssh_port: int = 22,
                 csv_path: Optional[str] = None):
        
        self.hostname = hostname
        self.interface = interface
        self.local_interface = local_interface
        
        # Web API setup
        self.base_url = f"https://{hostname}"
        self.web_username = web_username
        self.web_password = web_password
        self.session = requests.Session()
        
        # SSH setup
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_port = ssh_port
        self.ssh_client = None
        
        # CSV logging setup
        self.csv_path = csv_path
        if csv_path:
            self._setup_csv_logging()
            
        # Initialize capacity estimator
        self.capacity_estimator = CapacityEstimator()
        self.last_tc_update = 0.0
        self.tc_update_interval = 1.0
        
        # Setup logging
        self.logger = logging.getLogger('RadioMonitor')
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Connect on initialization
        self._web_login()
        self._connect_ssh()
    
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
                self.logger.info("Successfully logged into web interface")
            else:
                raise Exception(f"Web login failed with status code: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Web login failed: {str(e)}")
            raise

    def _connect_ssh(self) -> None:
        """Establish SSH connection with radio"""
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            
            self.ssh_client.connect(
                hostname=self.hostname,
                username=self.ssh_username,
                password=self.ssh_password,
                port=self.ssh_port,
                allow_agent=False,
                look_for_keys=False,
                disabled_algorithms={'keys': ['rsa-sha2-256', 'rsa-sha2-512']}
            )
            
            self.logger.info("Successfully connected via SSH")
            
        except Exception as e:
            self.logger.error(f"SSH connection failed: {str(e)}")
            raise
    
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
            cmd = f"tc -s qdisc show dev {self.interface}"
            stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
            
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
            return 0.0, 0.0, 0.0
    
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
                        'backlog', 'drops', 'useful_capacity', 'state'
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
                    stats.state
                ])

    def _update_tc_rate(self, rate_mbps: float) -> None:
        """Update traffic control rate via SSH if change threshold is met"""
        try:
            current_time = time.time()
            
            # Only update if enough time has passed since last update
            if current_time - self.last_tc_update >= self.tc_update_interval:
                # Get current rate by executing tc command
                tc_ssh = paramiko.SSHClient()
                tc_ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                
                try:
                    # Connect using key-based auth
                    tc_ssh.connect(
                        hostname="10.0.10.250",
                        username="wine",
                        key_filename="/home/wine/auth.txt",
                        look_for_keys=False
                    )
                    
                    # Get current rate
                    cmd = f"sudo tc qdisc show dev {self.local_interface} | grep cake"
                    stdin, stdout, stderr = tc_ssh.exec_command(cmd)
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
                    abs_diff = abs(rate_mbps - current_rate_mbps)
                    percent_diff = (abs_diff / current_rate_mbps) * 100 if current_rate_mbps > 0 else float('inf')
                    
                    # Only update if difference exceeds thresholds
                    if abs_diff >= 2.5 or percent_diff >= 1.0:
                        # Convert Mbps to bps for tc
                        rate_bps = int(rate_mbps * 1_000_000)
                        
                        # Build tc command
                        cmd = f"sudo tc qdisc change dev {self.local_interface} handle 8001: cake bandwidth {rate_bps}bit"
                        
                        # Execute command
                        stdin, stdout, stderr = tc_ssh.exec_command(cmd)
                        
                        # Check for errors
                        error_msg = stderr.read().decode().strip()
                        if error_msg:
                            self.logger.warning(f"TC command warning: {error_msg}")
                        
                        self.logger.info(f"Updated TC rate from {current_rate_mbps:.2f} to {rate_mbps:.2f} Mbps (diff: {abs_diff:.2f} Mbps, {percent_diff:.1f}%)")
                        self.last_tc_update = current_time
                    else:
                        self.logger.debug(f"Skipping TC update - difference too small (abs: {abs_diff:.2f} Mbps, {percent_diff:.1f}%)")
                    
                finally:
                    tc_ssh.close()
                
        except paramiko.AuthenticationException:
            self.logger.error("Authentication failed when connecting for TC update")
        except paramiko.SSHException as e:
            self.logger.error(f"SSH error during TC update: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error updating TC rate: {str(e)}")
    
    def get_stats(self) -> RadioStats:
        """Get all radio statistics and update capacity estimation"""
        try:
            # Get current timestamp
            timestamp = datetime.now(pytz.UTC)
            
            # Get web API stats
            status = self._get_web_status()
            tx_rate, rx_rate, radio_capacity = self._parse_web_stats(status)
            
            # Get queue stats via SSH
            backlog, drops = self._get_queue_stats()
            
            # Update capacity estimation
            useful_capacity, state = self.capacity_estimator.update(
                timestamp=timestamp,
                tx_rate=tx_rate,
                radio_capacity=radio_capacity,
                backlog=backlog,
                drops=drops
            )
            
            # Update traffic control rate if needed
            if useful_capacity is not None:
                self._update_tc_rate(useful_capacity)
            
            # Create stats object
            stats = RadioStats(
                timestamp=timestamp,
                tx_rate=tx_rate,
                rx_rate=rx_rate,
                radio_capacity=radio_capacity,
                backlog=backlog,
                drops=drops,
                useful_capacity=useful_capacity if useful_capacity is not None else radio_capacity,
                state=state
            )
            
            # Log stats to CSV
            self._log_stats(stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting radio stats: {str(e)}")
            raise
    
    def monitor(self, interval: float = 1.0):
        """Continuously monitor radio statistics"""
        try:
            while True:
                stats = self.get_stats()
                
                # Log the statistics
                self.logger.info(
                    f"Stats: TX Rate={stats.tx_rate:.2f} Mbps, "
                    f"RX Rate={stats.rx_rate:.2f} Mbps, "
                    f"TX Capacity={stats.radio_capacity:.2f} Mbps, "
                    f"Useful Capacity={stats.useful_capacity:.2f} Mbps, "
                    f"Backlog={stats.backlog}, Drops={stats.drops}, "
                    f"State={stats.state}"
                )
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            self.logger.info("Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Error during monitoring: {str(e)}")
        finally:
            if self.ssh_client:
                self.ssh_client.close()

def main():
    # Radio connection details
    RADIO_HOST = "10.0.10.247"
    WEB_USER = "ubnt"
    WEB_PASS = "u(a)=UP~0OBu"
    SSH_USER = "ubnt"
    SSH_PASS = "u(a)=UP~0OBu"
    RADIO_INTERFACE = "ath0"
    LOCAL_INTERFACE = "eth0"
    
    # Setup CSV logging
    csv_path = "radio_stats.csv"
    
    try:
        # Create monitor instance
        monitor = RadioMonitor(
            hostname=RADIO_HOST,
            web_username=WEB_USER,
            web_password=WEB_PASS,
            ssh_username=SSH_USER,
            ssh_password=SSH_PASS,
            interface=RADIO_INTERFACE,
            local_interface=LOCAL_INTERFACE,
            csv_path=csv_path
        )
        
        # Start monitoring
        monitor.monitor(interval=0.1) # was 1.0, limit here is ssh really
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")

if __name__ == "__main__":
    main()