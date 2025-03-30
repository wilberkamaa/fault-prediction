import numpy as np
from typing import Dict, Any

class BatterySystemSimulator:
    """Simulates a 3 MWh battery energy storage system (2 hours of solar capacity)."""
    
    def __init__(self, capacity_kwh: float = 3000, seed: int = 42):
        self.capacity_kwh = capacity_kwh
        self.max_power_kw = 750  # 50% of solar capacity
        self.min_soc = 0.1  # Lower minimum SOC for more usable capacity
        self.max_soc = 0.95  # Maximum SOC to prevent constant 100%
        np.random.seed(seed)
        
        # System parameters
        self.charging_efficiency = 0.95  # Modern lithium-ion
        self.discharging_efficiency = 0.95  # Modern lithium-ion
        self.self_discharge_rate = 0.0005  # 0.05% per hour
        self.nominal_voltage = 800  # Higher voltage for better efficiency
        self.cycles = 0
        self.degradation_per_cycle = 0.00005  # Modern batteries degrade slower
        
        # Dynamic behavior parameters
        self.charge_rate_factor = {
            'low': 1.0,    # Full power below 30% SOC
            'mid': 0.8,    # 80% power between 30-70% SOC
            'high': 0.5    # 50% power above 70% SOC
        }
        self.discharge_rate_factor = {
            'low': 0.5,    # 50% power below 30% SOC
            'mid': 0.8,    # 80% power between 30-70% SOC
            'high': 1.0    # Full power above 70% SOC
        }
        
    def temperature_effect(self, temperature: float) -> float:
        """Calculate temperature effect on capacity."""
        # Modern batteries have better temperature tolerance
        # Capacity decreases by 1% per 10°C deviation
        temp_diff = abs(temperature - 25)
        return 1 - (0.001 * temp_diff)
    
    def calculate_voltage(self, soc: float, current: float) -> float:
        """Calculate battery voltage based on SOC and current."""
        # Simple battery voltage model
        voltage = self.nominal_voltage * (0.9 + 0.2 * soc)
        # Add internal resistance effect
        voltage -= current * 0.1  # Simplified internal resistance effect
        return voltage
    
    def get_charge_rate_factor(self, soc: float) -> float:
        """Get charging rate factor based on SOC."""
        if soc < 0.3:
            return self.charge_rate_factor['low']
        elif soc < 0.7:
            return self.charge_rate_factor['mid']
        else:
            return self.charge_rate_factor['high']
    
    def get_discharge_rate_factor(self, soc: float) -> float:
        """Get discharging rate factor based on SOC."""
        if soc < 0.3:
            return self.discharge_rate_factor['low']
        elif soc < 0.7:
            return self.discharge_rate_factor['mid']
        else:
            return self.discharge_rate_factor['high']
    
    def generate_output(self, df, pv_output: np.ndarray, 
                       load_demand: np.ndarray) -> Dict[str, Any]:
        """Generate battery system output parameters."""
        hours = len(df)
        
        # Initialize arrays
        soc = np.zeros(hours)  # State of charge
        power = np.zeros(hours)  # Power output (positive for discharge)
        voltage = np.zeros(hours)
        current = np.zeros(hours)
        temperature = np.zeros(hours)
        
        # Initial conditions - start at 50% for balanced behavior
        soc[0] = 0.5
        last_soc = soc[0]
        
        # Create a forced pattern of SOC values to ensure full range
        target_soc_pattern = np.zeros(24)  # 24 hours in a day
        # Morning: charge from low to high
        target_soc_pattern[0:6] = np.linspace(0.2, 0.3, 6)  # Early morning - low SOC
        target_soc_pattern[6:12] = np.linspace(0.3, 0.8, 6)  # Morning - charging
        # Afternoon: stay high
        target_soc_pattern[12:18] = np.linspace(0.8, 0.9, 6)  # Afternoon - high SOC
        # Evening: discharge
        target_soc_pattern[18:24] = np.linspace(0.9, 0.2, 6)  # Evening - discharging
        
        # Force charge/discharge cycles - alternate every few hours
        charge_pattern = np.zeros(24, dtype=bool)
        charge_pattern[5:12] = True   # Morning charging
        charge_pattern[14:16] = True  # Afternoon top-up
        
        # Tracking variables for more dynamic behavior
        forced_mode_counter = 0
        force_charge = False
        force_discharge = False
        
        day_of_year = df.index.dayofyear.values
        hour_of_day = df.index.hour.values
        
        for i in range(hours):
            # Temperature affects capacity
            temp_factor = self.temperature_effect(df['weather_temperature'].iloc[i])
            effective_capacity = self.capacity_kwh * temp_factor
            
            # Get hour of day and determine target SOC
            hour = hour_of_day[i]
            target_soc = target_soc_pattern[hour]
            
            # Add variation to target SOC
            target_soc += np.random.normal(0, 0.05)
            target_soc = np.clip(target_soc, 0.15, 0.9)
            
            # Determine if we need to force a mode change for more dynamic behavior
            forced_mode_counter += 1
            if forced_mode_counter >= 12:  # Every 12 hours
                forced_mode_counter = 0
                # Switch between forced charge and discharge
                if np.random.random() < 0.5:
                    force_charge = True
                    force_discharge = False
                else:
                    force_charge = False
                    force_discharge = True
            
            # Determine charge mode
            if force_charge:
                charge_mode = True
            elif force_discharge:
                charge_mode = False
            else:
                # Base charge mode on time of day pattern
                charge_mode = charge_pattern[hour]
            
            # Override based on SOC limits
            if last_soc >= 0.9:  # Force discharge if too full
                charge_mode = False
                force_charge = False
            elif last_soc <= 0.15:  # Force charge if too empty
                charge_mode = True
                force_discharge = False
                
            # Calculate power balance
            power_balance = pv_output[i] - load_demand[i]
            
            # Add small random variations
            power_balance += np.random.normal(0, 30)
            
            # Calculate power based on mode
            if charge_mode and last_soc < self.max_soc - 0.05:  # Charge
                # Calculate charge power
                soc_diff = target_soc - last_soc
                charge_factor = min(1.0, max(0.2, abs(soc_diff) * 5))  # Higher power for bigger difference
                
                # Calculate maximum charging power
                max_charge = min(
                    abs(power_balance) if power_balance > 0 else self.max_power_kw * 0.4,
                    self.max_power_kw * charge_factor,
                    (self.max_soc - last_soc) * effective_capacity / self.charging_efficiency
                )
                
                # Add randomness to charging power
                charge_power = max_charge * (0.6 + 0.4 * np.random.random())
                
                # Negative power means charging
                power[i] = -charge_power
                
                # Calculate energy stored (accounting for efficiency)
                energy_stored = charge_power * self.charging_efficiency
                
            elif not charge_mode and last_soc > self.min_soc + 0.05:  # Discharge
                # Calculate discharge power
                soc_diff = last_soc - target_soc
                discharge_factor = min(1.0, max(0.2, abs(soc_diff) * 5))  # Higher power for bigger difference
                
                # Calculate maximum discharge power
                max_discharge = min(
                    abs(power_balance) if power_balance < 0 else self.max_power_kw * 0.4,
                    self.max_power_kw * discharge_factor,
                    (last_soc - self.min_soc) * effective_capacity * self.discharging_efficiency
                )
                
                # Add randomness to discharge power
                discharge_power = max_discharge * (0.6 + 0.4 * np.random.random())
                
                # Positive power means discharging
                power[i] = discharge_power
                
                # Calculate energy used (accounting for efficiency)
                energy_stored = -discharge_power / self.discharging_efficiency
                
            else:  # Idle - small self-discharge only
                power[i] = 0
                energy_stored = 0
            
            # Update state of charge
            if i < hours - 1:
                # Update SOC based on energy flow
                soc[i + 1] = last_soc - energy_stored / effective_capacity
                
                # Account for self-discharge
                soc[i + 1] *= (1 - self.self_discharge_rate)
                
                # Ensure SOC stays within bounds
                soc[i + 1] = np.clip(soc[i + 1], self.min_soc, self.max_soc)
                
                # Add small random fluctuations for realism
                soc[i + 1] += np.random.normal(0, 0.01)  # +/- 1% random noise
                soc[i + 1] = np.clip(soc[i + 1], self.min_soc, self.max_soc)  # Re-clip after noise
                
                last_soc = soc[i + 1]
            
            # Calculate electrical parameters
            current[i] = power[i] * 1000 / self.nominal_voltage  # Convert kW to W
            voltage[i] = self.calculate_voltage(soc[i], current[i])
            
            # Battery temperature (simplified model)
            temp_rise = 0.05 * abs(power[i])  # Temperature rise due to power flow
            temperature[i] = df['weather_temperature'].iloc[i] + temp_rise
        
        return {
            'soc': soc,
            'power': power,  
            'voltage': voltage,
            'current': current,
            'temperature': temperature
        }
