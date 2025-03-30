import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum, auto
import pandas as pd
class FaultType(Enum):
    """Enumeration of possible fault types, expanded for battery faults."""
    LINE_SHORT_CIRCUIT = auto()
    LINE_PROLONGED_UNDERVOLTAGE = auto()
    INVERTER_IGBT_FAILURE = auto()
    GENERATOR_FIELD_FAILURE = auto()
    GRID_VOLTAGE_SAG = auto()
    GRID_OUTAGE = auto()
    BATTERY_OVERDISCHARGE = auto()
    BATTERY_OVERCHARGE = auto()  # New: Overcharging fault
    BATTERY_THERMAL = auto()     # New: Thermal runaway or overheating
    NO_FAULT = auto()

@dataclass
class FaultEvent:
    """Data class for fault events."""
    fault_type: FaultType
    start_time: int
    duration: int
    severity: float
    affected_parameters: Dict[str, float]

class FaultInjectionSystem:
    """Fault injection system with enhanced battery fault generation."""
    
    def __init__(self, seed: int = 42, base_fault_rate: float = 0.005):
        """
        Initialize with a base fault rate and ripple mechanics.

        Args:
            seed: Random seed for reproducibility.
            base_fault_rate: Base probability of a random fault per hour (default 0.005).
        """
        np.random.seed(seed)
        self.base_fault_rate = base_fault_rate
        
        # Base fault probabilities (per hour), higher for battery faults
        self.base_probabilities = {
            FaultType.LINE_SHORT_CIRCUIT: 0.002,
            FaultType.LINE_PROLONGED_UNDERVOLTAGE: 0.003,
            FaultType.INVERTER_IGBT_FAILURE: 0.002,
            FaultType.GENERATOR_FIELD_FAILURE: 0.002,
            FaultType.GRID_VOLTAGE_SAG: 0.004,
            FaultType.GRID_OUTAGE: 0.003,
            FaultType.BATTERY_OVERDISCHARGE: 0.005,  # Increased
            FaultType.BATTERY_OVERCHARGE: 0.005,    # New
            FaultType.BATTERY_THERMAL: 0.005        # New
        }
        
        # Fault durations (hours)
        self.fault_durations = {
            FaultType.LINE_SHORT_CIRCUIT: (1, 4),
            FaultType.LINE_PROLONGED_UNDERVOLTAGE: (2, 8),
            FaultType.INVERTER_IGBT_FAILURE: (4, 24),
            FaultType.GENERATOR_FIELD_FAILURE: (8, 48),
            FaultType.GRID_VOLTAGE_SAG: (1, 6),
            FaultType.GRID_OUTAGE: (2, 12),
            FaultType.BATTERY_OVERDISCHARGE: (1, 4),
            FaultType.BATTERY_OVERCHARGE: (1, 4),
            FaultType.BATTERY_THERMAL: (2, 12)      # Longer for thermal issues
        }
        
        # Ripple effect parameters
        self.ripple_window = 48
        self.ripple_multiplier = 3.0
        self.min_fault_gap = 12
        
        # Track last fault times and ripple states
        self.last_fault_times: Dict[FaultType, int] = {ftype: -self.min_fault_gap for ftype in FaultType}
        self.ripple_active: Dict[FaultType, int] = {ftype: -self.ripple_window for ftype in FaultType}

    def _get_ripple_factor(self, fault_type: FaultType, current_hour: int) -> float:
        """Calculate ripple factor based on recent faults of the same type."""
        time_since_last = current_hour - self.ripple_active[fault_type]
        if time_since_last <= self.ripple_window:
            decay_factor = 1 - (time_since_last / self.ripple_window)
            return 1 + (self.ripple_multiplier - 1) * decay_factor
        return 1.0

    def check_fault_conditions(self, system_state: Dict[str, np.ndarray], 
                              hour: int, df: pd.DataFrame) -> List[Tuple[FaultType, float]]:
        """
        Check conditions with enhanced battery fault triggers.

        Args:
            system_state: Dictionary of system parameter arrays.
            hour: Current simulation hour index.
            df: DataFrame with additional system data.
        
        Returns:
            List of (fault_type, probability) tuples for potential faults.
        """
        potential_faults = []
        hour_of_day = df.index[hour].hour

        # Base random fault chance with ripple effect
        for fault_type in self.base_probabilities:
            if hour - self.last_fault_times[fault_type] >= self.min_fault_gap:
                ripple_factor = self._get_ripple_factor(fault_type, hour)
                prob = self.base_fault_rate * self.base_probabilities[fault_type] * ripple_factor
                potential_faults.append((fault_type, prob))

        # Condition-based boosts
        if 'grid_voltage' in system_state and system_state['grid_voltage'] is not None:
            voltage_ratio = system_state['grid_voltage'][hour] / 25000
            if voltage_ratio < 0.9:
                ripple_factor = self._get_ripple_factor(FaultType.LINE_SHORT_CIRCUIT, hour)
                prob = self.base_probabilities[FaultType.LINE_SHORT_CIRCUIT] * 2 * ripple_factor
                potential_faults.append((FaultType.LINE_SHORT_CIRCUIT, prob))

        if 'inverter_temp' in system_state and system_state['inverter_temp'] is not None:
            temp = system_state['inverter_temp'][hour]
            if temp > 70:
                ripple_factor = self._get_ripple_factor(FaultType.INVERTER_IGBT_FAILURE, hour)
                prob = self.base_probabilities[FaultType.INVERTER_IGBT_FAILURE] * (1 + (temp - 70) / 20) * ripple_factor
                potential_faults.append((FaultType.INVERTER_IGBT_FAILURE, prob))

        if 'generator_runtime' in system_state and system_state['generator_runtime'] is not None:
            runtime = system_state['generator_runtime'][hour]
            if runtime > 75:
                ripple_factor = self._get_ripple_factor(FaultType.GENERATOR_FIELD_FAILURE, hour)
                prob = self.base_probabilities[FaultType.GENERATOR_FIELD_FAILURE] * (1 + runtime / 150) * ripple_factor
                potential_faults.append((FaultType.GENERATOR_FIELD_FAILURE, prob))

        # Enhanced battery fault triggers
        if 'battery_soc' in system_state and system_state['battery_soc'] is not None:
            soc = system_state['battery_soc'][hour]
            # Overdischarge: Relaxed threshold to 0.25
            if soc < 0.25:
                ripple_factor = self._get_ripple_factor(FaultType.BATTERY_OVERDISCHARGE, hour)
                prob = self.base_probabilities[FaultType.BATTERY_OVERDISCHARGE] * (1 + (0.25 - soc) * 10) * ripple_factor
                potential_faults.append((FaultType.BATTERY_OVERDISCHARGE, prob))
            # Overcharge: SOC > 0.9
            if soc > 0.9:
                ripple_factor = self._get_ripple_factor(FaultType.BATTERY_OVERCHARGE, hour)
                prob = self.base_probabilities[FaultType.BATTERY_OVERCHARGE] * (1 + (soc - 0.9) * 20) * ripple_factor
                potential_faults.append((FaultType.BATTERY_OVERCHARGE, prob))
            # Rapid SOC change
            if hour > 0 and abs(soc - system_state['battery_soc'][hour-1]) > 0.05:
                ripple_factor = self._get_ripple_factor(FaultType.BATTERY_THERMAL, hour)
                prob = self.base_probabilities[FaultType.BATTERY_THERMAL] * 2 * ripple_factor
                potential_faults.append((FaultType.BATTERY_THERMAL, prob))

        if 'battery_temperature' in system_state and system_state['battery_temperature'] is not None:
            temp = system_state['battery_temperature'][hour]
            if temp > 40:  # Realistic threshold for lithium-ion batteries
                ripple_factor = self._get_ripple_factor(FaultType.BATTERY_THERMAL, hour)
                prob = self.base_probabilities[FaultType.BATTERY_THERMAL] * (1 + (temp - 40) / 10) * ripple_factor
                potential_faults.append((FaultType.BATTERY_THERMAL, prob))

        # Grid outage during peak hours
        if hour_of_day in range(17, 23):
            ripple_factor = self._get_ripple_factor(FaultType.GRID_OUTAGE, hour)
            prob = self.base_probabilities[FaultType.GRID_OUTAGE] * 1.5 * ripple_factor
            potential_faults.append((FaultType.GRID_OUTAGE, prob))

        return potential_faults
    
    def generate_fault_events(self, df, system_state: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Generate fault events with enhanced battery fault logic."""
        hours = len(df)
        fault_occurred = np.zeros(hours, dtype=bool)
        fault_types = np.full(hours, 'NO_FAULT', dtype=object)
        fault_severity = np.zeros(hours)
        fault_starts = np.zeros(hours, dtype=bool)
        fault_durations = np.zeros(hours)
        active_faults: List[FaultEvent] = []

        for hour in range(hours):
            active_faults = [f for f in active_faults if hour < f.start_time + f.duration]
            potential_faults = self.check_fault_conditions(system_state, hour, df)
            
            for fault_type, probability in potential_faults:
                if np.random.random() < min(probability, 0.3):
                    duration = np.random.randint(*self.fault_durations[fault_type])
                    severity = np.random.uniform(0.1, 0.8)
                    fault_event = FaultEvent(
                        fault_type=fault_type,
                        start_time=hour,
                        duration=min(duration, hours - hour),
                        severity=severity,
                        affected_parameters=self._generate_fault_effects(fault_type, severity)
                    )
                    active_faults.append(fault_event)
                    self.last_fault_times[fault_type] = hour
                    self.ripple_active[fault_type] = hour

            if active_faults:
                fault_occurred[hour] = True
                primary_fault = max(active_faults, key=lambda x: x.severity)
                fault_types[hour] = primary_fault.fault_type.name
                fault_severity[hour] = primary_fault.severity
                for fault in active_faults:
                    if fault.start_time == hour:
                        fault_starts[hour] = True
                        fault_durations[hour] = fault.duration

        return {
            'occurred': fault_occurred,
            'type': fault_types,
            'severity': fault_severity,
            'start': fault_starts,
            'duration': fault_durations
        }
    
    def _generate_fault_effects(self, fault_type: FaultType, severity: float) -> Dict[str, float]:
        """Generate effects with battery-specific impacts."""
        effects = {}
        if fault_type == FaultType.LINE_SHORT_CIRCUIT:
            effects.update({'voltage_drop': 0.5 + 0.5 * severity, 'current_spike': 1.0 + severity})
        elif fault_type == FaultType.LINE_PROLONGED_UNDERVOLTAGE:
            effects.update({'voltage_reduction': 0.1 + 0.4 * severity})
        elif fault_type == FaultType.INVERTER_IGBT_FAILURE:
            effects.update({'efficiency_drop': 0.3 * severity, 'temperature_rise': 10 * severity})
        elif fault_type == FaultType.GENERATOR_FIELD_FAILURE:
            effects.update({'voltage_deviation': 0.1 * severity, 'frequency_deviation': 0.05 * severity})
        elif fault_type == FaultType.GRID_VOLTAGE_SAG:
            effects.update({'voltage_drop': 0.05 + 0.25 * severity})
        elif fault_type == FaultType.GRID_OUTAGE:
            effects.update({'power_loss': 0.8 + 0.2 * severity})
        elif fault_type == FaultType.BATTERY_OVERDISCHARGE:
            effects.update({'capacity_loss': 0.1 * severity, 'internal_resistance': 0.8 + 0.4 * severity})
        elif fault_type == FaultType.BATTERY_OVERCHARGE:
            effects.update({'capacity_loss': 0.05 * severity, 'voltage_spike': 0.5 + 0.5 * severity})
        elif fault_type == FaultType.BATTERY_THERMAL:
            effects.update({'temperature_rise': 10 * severity, 'efficiency_drop': 0.2 * severity})
        return effects