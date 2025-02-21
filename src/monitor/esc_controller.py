# import numpy as np
# import logging
# from dataclasses import dataclass
# from datetime import datetime
# from typing import Tuple, Optional

# @dataclass
# class ESCState:
#     """State variables for ESC controller"""
#     estimate: float           # Current capacity estimate
#     filtered_capacity: float  # Filtered measurement
#     adaptive_gain: float     # Adaptive gain parameter
#     last_time: float         # Last update timestamp
#     last_perturbation: float # Last perturbation value
#     last_objective: float    # Last objective function value
#     last_drops: int         # Previous drop count
#     drop_rate: float        # Filtered drop rate

# class ESCCapacityEstimator:
#     """Implements capacity estimation using Extremum Seeking Control."""
#     def __init__(self,
#                  omega: float = 0.5,        # Perturbation frequency
#                  alpha: float = 0.2,        # Filter coefficient
#                  perturb_amp: float = 1.0,  # Perturbation amplitude
#                  safety_margin: float = 0.80, # Base safety margin (using 0.80 to match current code)
#                  update_interval: float = 1.0): # Min time between updates
        
#         self.omega = omega
#         self.alpha = alpha
#         self.perturb_amp = perturb_amp
#         self.safety_margin = safety_margin
#         self.update_interval = update_interval
        
#         # State initialization
#         self.state: Optional[ESCState] = None
#         self.logger = logging.getLogger('ESCCapacityEstimator')
    
#     def _calculate_perturbation(self, t: float) -> float:
#         """Calculate perturbation signal."""
#         return self.perturb_amp * np.sin(self.omega * t)
    
#     def _calculate_objective(self, 
#                            estimate: float,
#                            safe_capacity: float,
#                            tx_rate: float,
#                            backlog: int,
#                            delta_drops: int,
#                            drop_rate: float) -> float:
#         """Calculate multi-objective function value."""
#         # Capacity error term
#         capacity_error = (estimate - safe_capacity) / safe_capacity
#         capacity_term = -capacity_error * capacity_error
        
#         # Throughput utilization term
#         utilization = tx_rate / estimate if estimate > 0 else 0
#         throughput_term = utilization * 0.5
        
#         # Backlog penalty (more aggressive than before)
#         backlog_penalty = -0.2 if backlog > 0 else 0
        
#         # Drop penalties
#         drop_penalty = -0.2 if delta_drops > 0 else 0
#         drop_rate_penalty = -0.3 * (drop_rate / 10.0) if drop_rate > 0 else 0
        
#         return (capacity_term + 
#                 throughput_term + 
#                 backlog_penalty + 
#                 drop_penalty + 
#                 drop_rate_penalty)
    
#     def update(self, 
#                timestamp: datetime,
#                tx_rate: float,
#                radio_capacity: float,
#                backlog: int,
#                drops: int) -> Tuple[float, str]:
#         """Update capacity estimate with new measurements."""
#         try:
#             # Convert values and handle NaN/None
#             tx_rate_val = float(tx_rate) if tx_rate is not None else 0.0
#             capacity_val = float(radio_capacity) if radio_capacity is not None else 0.0
#             backlog_val = int(backlog) if backlog is not None else 0
#             drops_val = int(drops) if drops is not None else 0
#             current_time = timestamp.timestamp()
            
#             if capacity_val <= 0:
#                 return 0.0, "ERROR"
                
#             # Initialize state if needed
#             if self.state is None:
#                 self.state = ESCState(
#                     estimate=capacity_val * self.safety_margin,
#                     filtered_capacity=capacity_val,
#                     adaptive_gain=0.5,
#                     last_time=current_time,
#                     last_perturbation=0.0,
#                     last_objective=0.0,
#                     last_drops=drops_val,
#                     drop_rate=0.0
#                 )
#                 return self.state.estimate, "INIT"
            
#             # Check update interval
#             dt = current_time - self.state.last_time
#             if dt < self.update_interval:
#                 return self.state.estimate, self._get_state(backlog_val)
            
#             # Update filtered capacity
#             self.state.filtered_capacity = (
#                 self.state.filtered_capacity * (1 - self.alpha) +
#                 self.alpha * capacity_val
#             )
            
#             # Calculate drop metrics
#             delta_drops = max(0, drops_val - self.state.last_drops)
#             if dt > 0:
#                 instant_drop_rate = delta_drops / dt
#                 self.state.drop_rate = (
#                     self.state.drop_rate * (1 - self.alpha) +
#                     self.alpha * instant_drop_rate
#                 )
            
#             # Calculate safe capacity target
#             safe_capacity = self.state.filtered_capacity * self.safety_margin
            
#             # Generate perturbation
#             perturbation = self._calculate_perturbation(current_time)
            
#             # Calculate objective value
#             objective = self._calculate_objective(
#                 self.state.estimate,
#                 safe_capacity,
#                 tx_rate_val,
#                 backlog_val,
#                 delta_drops,
#                 self.state.drop_rate
#             )
            
#             # Estimate gradient using perturbation
#             gradient = objective * self.state.last_perturbation
            
#             # Update adaptive gain
#             gain_update = 0.01 * np.sign(gradient) * min(1.0, abs(gradient))
#             self.state.adaptive_gain = max(0.1, min(1.0, 
#                 self.state.adaptive_gain + gain_update))
            
#             # Update estimate
#             raw_estimate = (
#                 self.state.estimate * (1 - self.alpha) +
#                 self.alpha * (safe_capacity + self.state.adaptive_gain * gradient)
#             )
            
#             # Apply safety bounds
#             min_capacity = capacity_val * 0.7
#             max_capacity = capacity_val
#             self.state.estimate = max(min_capacity, min(max_capacity, raw_estimate))
            
#             # Store state for next update
#             self.state.last_perturbation = perturbation
#             self.state.last_objective = objective
#             self.state.last_time = current_time
#             self.state.last_drops = drops_val
            
#             # Add perturbation to final estimate
#             final_estimate = self.state.estimate + perturbation
            
#             return final_estimate, self._get_state(backlog_val)
            
#         except Exception as e:
#             self.logger.error(f"Error in ESC update: {str(e)}")
#             return radio_capacity * self.safety_margin, "ERROR"
    
#     def _get_state(self, backlog: int) -> str:
#         """Determine current state based on conditions."""
#         if backlog > 0:
#             return "CONGESTION"
#         elif self.state.adaptive_gain < 0.3:
#             return "CONSERVATIVE"
#         elif self.state.adaptive_gain > 0.7:
#             return "AGGRESSIVE"
#         return "STABLE"



# New version:

import numpy as np
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Optional, Dict

@dataclass
class ESCState:
    """Complete state variables for enhanced ESC controller"""
    estimate: float              # Ĉ(t): Current capacity estimate
    filtered_capacity: float     # Filtered C(t)
    adaptive_gain: float         # k(t): Current adaptive gain
    backlog_gain: float         # kb(t): Backlog-based gain component
    drop_gain: float            # kd(t): Drop-based gain component
    last_time: float            # Previous update timestamp
    last_perturbation: float    # Previous perturbation value
    last_objective: float       # Previous objective value
    last_drops: int             # d(t-Δt): Previous drop count
    last_backlog: int           # b(t-Δt): Previous backlog
    drop_rate: float           # đ̄(t): Filtered drop rate
    backlog_rate: float        # ḃ(t): Current backlog rate
    sustained_congestion: bool  # Flag for extended congestion periods

@dataclass
class ESCParameters:
    """Tunable parameters for the ESC controller"""
    # Fundamental ESC parameters
    omega: float = 0.5          # Perturbation frequency
    alpha: float = 0.2          # Main learning rate
    perturb_amp: float = 1.0    # Perturbation amplitude
    safety_margin: float = 0.75 # Base safety margin γ. Default 0.85
    update_interval: float = 1.0 # Minimum update interval
    
    # Objective function weights
    Q_c: float = 1.0           # Capacity tracking weight
    Q_r: float = 0.5           # Throughput reward weight
    Q_b: float = 0.2           # Backlog penalty weight
    Q_d: float = 0.3           # Drop penalty weight
    
    # Backlog penalty parameters
    lambda1: float = 0.1        # Immediate backlog penalty
    lambda2: float = 0.2        # Backlog growth penalty
    
    # Drop penalty parameters
    lambda3: float = 0.15       # Immediate drop penalty
    lambda4: float = 0.25       # Sustained drop penalty
    
    # Gain adaptation parameters
    beta: float = 0.01         # Gain adaptation rate
    eta1: float = 0.1          # Backlog sensitivity
    eta2: float = 0.2          # Drop sensitivity
    k_min: float = 0.1         # Minimum gain
    k_max: float = 1.0         # Maximum gain
    
    # Safety bounds
    c_min_factor: float = 0.7   # Minimum capacity factor
    congestion_reduction: float = 0.8  # Additional reduction during congestion

class EnhancedESCEstimator:
    """
    Enhanced implementation of capacity estimation using ESC,
    following the mathematical design more precisely.
    """
    def __init__(self, params: Optional[ESCParameters] = None):
        self.params = params or ESCParameters()
        self.state: Optional[ESCState] = None
        self.logger = logging.getLogger('EnhancedESCEstimator')
    
    def _calculate_perturbation(self, t: float) -> float:
        """Calculate sinusoidal perturbation signal: p(t) = a sin(ωt)"""
        return self.params.perturb_amp * np.sin(self.params.omega * t)
    
    def _calculate_backlog_penalty(self, 
                                 backlog: int,
                                 backlog_rate: float) -> float:
        """
        Calculate backlog penalty function φ(b):
        φ(b) = 0 if b = 0
        φ(b) = λ₁b + λ₂ḃ otherwise
        """
        if backlog == 0:
            return 0.0
        return (self.params.lambda1 * backlog + 
                self.params.lambda2 * backlog_rate)
    
    def _calculate_drop_penalty(self,
                              drop_rate: float,
                              filtered_drop_rate: float) -> float:
        """
        Calculate drop penalty function ψ(ḋ):
        ψ(ḋ) = λ₃ḋ + λ₄đ̄
        """
        return (self.params.lambda3 * drop_rate + 
                self.params.lambda4 * filtered_drop_rate)
    
    def _calculate_objective(self,
                           estimate: float,
                           safe_capacity: float,
                           tx_rate: float,
                           backlog: int,
                           backlog_rate: float,
                           drop_rate: float,
                           filtered_drop_rate: float) -> float:
        """
        Calculate enhanced multi-objective function J(t):
        J(t) = -Qc(Ĉ(t) - γC(t))² + Qr(r(t)/Ĉ(t)) - Qbφ(b(t)) - Qdψ(ḋ(t))
        """
        # Capacity tracking term
        capacity_error = (estimate - safe_capacity) ** 2
        capacity_term = -self.params.Q_c * capacity_error
        
        # Throughput reward term
        utilization = tx_rate / estimate if estimate > 0 else 0
        throughput_term = self.params.Q_r * utilization
        
        # Backlog penalty term
        backlog_term = -self.params.Q_b * self._calculate_backlog_penalty(
            backlog, backlog_rate)
        
        # Drop penalty term
        drop_term = -self.params.Q_d * self._calculate_drop_penalty(
            drop_rate, filtered_drop_rate)
        
        return capacity_term + throughput_term + backlog_term + drop_term
    
    def _update_gains(self,
                     backlog: int,
                     filtered_drop_rate: float,
                     gradient: float) -> Tuple[float, float, float]:
        """
        Update adaptive gains with backlog and drop sensitivity:
        kb(t) = e^(-η₁b(t))
        kd(t) = e^(-η₂đ̄(t))
        k(t+Δt) = clip(kb(t)kd(t)(k(t) + βg(t)), kmin, kmax)
        """
        # Calculate component gains
        kb = np.exp(-self.params.eta1 * backlog)
        kd = np.exp(-self.params.eta2 * filtered_drop_rate)
        
        # Update main gain
        gain_update = self.params.beta * gradient
        new_gain = np.clip(
            kb * kd * (self.state.adaptive_gain + gain_update),
            self.params.k_min,
            self.params.k_max
        )
        
        return new_gain, kb, kd
    
    def _calculate_safety_bounds(self,
                               capacity: float) -> Tuple[float, float]:
        """Calculate dynamic safety bounds with congestion awareness"""
        base_min = capacity * self.params.c_min_factor
        if self.state.sustained_congestion:
            base_min *= self.params.congestion_reduction
        return base_min, capacity
    
    def update(self,
              timestamp: datetime,
              tx_rate: float,
              radio_capacity: float,
              backlog: int,
              drops: int) -> Tuple[float, Dict[str, float]]:
        """
        Update capacity estimate with new measurements.
        Returns: (estimate, diagnostic_data)
        """
        try:
            # Convert and validate inputs
            tx_rate_val = float(tx_rate) if tx_rate is not None else 0.0
            capacity_val = float(radio_capacity) if radio_capacity is not None else 0.0
            backlog_val = int(backlog) if backlog is not None else 0
            drops_val = int(drops) if drops is not None else 0
            current_time = timestamp.timestamp()
            
            if capacity_val <= 0:
                return 0.0, {"state": "ERROR"}
            
            # Initialize state if needed
            if self.state is None:
                self.state = ESCState(
                    estimate=capacity_val * self.params.safety_margin,
                    filtered_capacity=capacity_val,
                    adaptive_gain=0.5,
                    backlog_gain=1.0,
                    drop_gain=1.0,
                    last_time=current_time,
                    last_perturbation=0.0,
                    last_objective=0.0,
                    last_drops=drops_val,
                    last_backlog=backlog_val,
                    drop_rate=0.0,
                    backlog_rate=0.0,
                    sustained_congestion=False
                )
                return self.state.estimate, {"state": "INIT"}
            
            # Check update interval
            dt = current_time - self.state.last_time
            if dt < self.params.update_interval:
                return self.state.estimate, self._get_diagnostics()
            
            # Update filtered capacity
            self.state.filtered_capacity = (
                self.state.filtered_capacity * (1 - self.params.alpha) +
                self.params.alpha * capacity_val
            )
            
            # Calculate rates
            if dt > 0:
                # Update drop rate metrics
                delta_drops = max(0, drops_val - self.state.last_drops)
                instant_drop_rate = delta_drops / dt
                self.state.drop_rate = (
                    self.state.drop_rate * (1 - self.params.alpha) +
                    self.params.alpha * instant_drop_rate
                )
                
                # Update backlog rate
                delta_backlog = backlog_val - self.state.last_backlog
                self.state.backlog_rate = delta_backlog / dt
            
            # Calculate safe capacity target
            safe_capacity = self.state.filtered_capacity * self.params.safety_margin
            
            # Generate perturbation
            perturbation = self._calculate_perturbation(current_time)
            
            # Calculate objective value
            objective = self._calculate_objective(
                self.state.estimate,
                safe_capacity,
                tx_rate_val,
                backlog_val,
                self.state.backlog_rate,
                instant_drop_rate,
                self.state.drop_rate
            )
            
            # Estimate gradient using perturbation
            gradient = objective * self.state.last_perturbation
            
            # Update gains
            new_gain, kb, kd = self._update_gains(
                backlog_val,
                self.state.drop_rate,
                gradient
            )
            self.state.adaptive_gain = new_gain
            self.state.backlog_gain = kb
            self.state.drop_gain = kd
            
            # Update sustained congestion state
            self.state.sustained_congestion = (
                backlog_val > 0 and self.state.drop_rate > 0
            )
            
            # Calculate safety bounds
            min_capacity, max_capacity = self._calculate_safety_bounds(capacity_val)
            
            # Update estimate
            raw_estimate = (
                self.state.estimate * (1 - self.params.alpha) +
                self.params.alpha * (safe_capacity + 
                                   self.state.adaptive_gain * gradient)
            )
            
            # Apply safety bounds
            self.state.estimate = np.clip(raw_estimate, min_capacity, max_capacity)
            
            # Store state for next update
            self.state.last_perturbation = perturbation
            self.state.last_objective = objective
            self.state.last_time = current_time
            self.state.last_drops = drops_val
            self.state.last_backlog = backlog_val
            
            # Add perturbation to final estimate
            final_estimate = self.state.estimate + perturbation
            
            return final_estimate, self._get_diagnostics()
            
        except Exception as e:
            self.logger.error(f"Error in ESC update: {str(e)}")
            return (radio_capacity * self.params.safety_margin, 
                   {"state": "ERROR", "error": str(e)})
    
    def _get_diagnostics(self) -> Dict[str, float]:
        """Get diagnostic data about controller state"""
        return {
            "state": self._get_state(),
            "estimate": self.state.estimate,
            "filtered_capacity": self.state.filtered_capacity,
            "adaptive_gain": self.state.adaptive_gain,
            "backlog_gain": self.state.backlog_gain,
            "drop_gain": self.state.drop_gain,
            "drop_rate": self.state.drop_rate,
            "backlog_rate": self.state.backlog_rate,
            "objective": self.state.last_objective
        }
    
    def _get_state(self) -> str:
        """Determine current controller state"""
        if self.state.sustained_congestion:
            return "CONGESTION"
        elif self.state.backlog_gain < 0.5 or self.state.drop_gain < 0.5:
            return "CONSERVATIVE"
        elif self.state.adaptive_gain > 0.7:
            return "AGGRESSIVE"
        return "STABLE"