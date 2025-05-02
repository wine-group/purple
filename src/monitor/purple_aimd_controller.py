#!/usr/bin/env python3
"""
PURPLE-AIMD Controller

A controller that adjusts network bandwidth based solely on latency measurements.
Uses an AIMD (Additive Increase/Multiplicative Decrease) approach for stability.
"""

import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple


@dataclass
class PurpleAIMDConfig:
    """Configuration for the PURPLE-AIMD Controller"""
    
    # Latency targets (milliseconds)
    target_latency: float = 5.0
    critical_latency: float = 30.0
    latency_tolerance: float = 2.0
    
    # Rate parameters (Mbps)
    min_rate: float = 10.0
    max_rate: float = 1000.0
    initial_rate: float = 95.0
    
    # AIMD parameters
    decay_factor: float = 0.1  # Multiplicative decrease factor when latency is high
    recovery_rate: float = 0.5  # Additive increase rate (Mbps per second) when latency is good
    
    # Controller parameters
    good_latency_threshold: int = 5  # How many consecutive good latency samples before recovery
    trend_window_size: int = 5  # Window size for latency trend calculation


class PurpleAIMDController:
    """
    A controller that determines bandwidth settings based on latency measurements.
    Implements AIMD (Additive Increase/Multiplicative Decrease) for stability.
    """
    
    def __init__(self, config: PurpleAIMDConfig):
        """Initialise the controller with the given configuration"""
        self.config = config
        
        # Internal state
        self.current_rate = config.initial_rate
        self.last_update_time = 0.0
        self.good_latency_count = 0
        self.recovery_phase = False
        self.latency_history = []
        self.latency_trend = 0.0
        
    def get_diagnostics(self) -> Dict[str, Any]:
        """Return internal state for diagnostics"""
        return {
            'good_latency_count': self.good_latency_count,
            'recovery_phase': self.recovery_phase,
            'latency_trend': self.latency_trend
        }
    
    def _calculate_latency_trend(self, new_latency: float) -> float:
        """Calculate the trend of latency values over time"""
        # Store the new latency value
        self.latency_history.append(new_latency)
        
        # Keep only the most recent values
        if len(self.latency_history) > self.config.trend_window_size:
            self.latency_history = self.latency_history[-self.config.trend_window_size:]
        
        # Need at least 2 points to calculate a trend
        if len(self.latency_history) < 2:
            return 0.0
        
        # Simple linear trend (positive = increasing latency, negative = decreasing)
        x = list(range(len(self.latency_history)))
        y = self.latency_history
        
        # Calculate linear regression slope
        n = len(x)
        if n <= 1:
            return 0.0
            
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum([x[i] * y[i] for i in range(n)])
        sum_xx = sum([x[i] * x[i] for i in range(n)])
        
        # Avoid division by zero
        if sum_xx - (sum_x * sum_x) / n == 0:
            return 0.0
            
        slope = (sum_xy - (sum_x * sum_y) / n) / (sum_xx - (sum_x * sum_x) / n)
        return slope
    
    def update(self, latency: float, current_time: float) -> Tuple[float, float]:
        """
        Update the controller with a new latency measurement
        
        Args:
            latency: The current latency in milliseconds
            current_time: The current time in seconds
            
        Returns:
            Tuple containing:
            - New bandwidth rate in Mbps
            - Objective value (target latency)
        """
        # Calculate time since last update
        if self.last_update_time == 0:
            time_delta = 0.1  # Default for first update
        else:
            time_delta = current_time - self.last_update_time
        
        self.last_update_time = current_time
        
        # Calculate latency trend
        self.latency_trend = self._calculate_latency_trend(latency)
        
        # Determine if latency is within acceptable range
        is_good_latency = latency <= self.config.target_latency + self.config.latency_tolerance
        is_critical_latency = latency >= self.config.critical_latency
        
        # Update good latency counter
        if is_good_latency:
            self.good_latency_count += 1
        else:
            self.good_latency_count = 0
            
        # Check if we can switch to recovery phase
        if self.good_latency_count >= self.config.good_latency_threshold:
            self.recovery_phase = True
            
        # Handle critical latency - immediate action needed
        if is_critical_latency:
            print("Critical latency detected!")
            self.recovery_phase = False
            self.good_latency_count = 0
            
            # Multiplicative decrease - aggressive
            self.current_rate *= (1.0 - self.config.decay_factor)
            
        # Regular operation
        elif not is_good_latency:
            self.recovery_phase = False
            
            # Multiplicative decrease - normal
            self.current_rate *= (1.0 - self.config.decay_factor/2)
            
        # Recovery phase - latency is good, gradually increase
        elif self.recovery_phase:
            # Additive increase - gentle recovery proportional to time elapsed
            self.current_rate += self.config.recovery_rate * time_delta
        
        # Enforce min/max rate
        self.current_rate = max(self.config.min_rate, min(self.config.max_rate, self.current_rate))
        
        return self.current_rate, self.config.target_latency


# Simple test routine
if __name__ == "__main__":
    config = PurpleAIMDConfig()
    controller = PurpleAIMDController(config)
    
    print("Simple PURPLE-AIMD Controller Test")
    print(f"Target latency: {config.target_latency}ms")
    print(f"Initial rate: {config.initial_rate}Mbps")
    
    # Simulate a few control cycles
    for i in range(10):
        # Simulate latency based on rate
        simulated_latency = 10.0 + 100.0 / controller.current_rate
        if i == 5:
            simulated_latency = 40.0  # Simulate a latency spike
            
        new_rate, target = controller.update(simulated_latency, time.time())
        
        print(f"Cycle {i+1}: Latency={simulated_latency:.1f}ms, Rate={new_rate:.1f}Mbps")
        time.sleep(0.5)
    
    print("Test complete")