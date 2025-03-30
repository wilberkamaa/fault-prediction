import numpy as np
from typing import Dict, Any

class DieselGeneratorSimulator:
    """Simulates a 1 MVA diesel generator system for backup power."""
    
    def __init__(self, capacity_kva: float = 1000, seed: int = 42):
        self.capacity_kva = capacity_kva
        self.fuel_tank_capacity = 2000  # Smaller tank for backup use
        np.random.seed(seed)
        
        # Operating parameters
        self.min_load_percent = 0.40  # Higher minimum for better efficiency
        self.min_runtime_hours = 2    # Minimum runtime once started
        self.fuel_consumption_rate = {
            'idle': 8,       # More efficient modern engine
            'full_load': 80  # Better fuel economy
        }
        self.maintenance_interval = 750  # Modern engine needs less maintenance
        
    def calculate_fuel_consumption(self, load_percent: float) -> float:
        """Calculate fuel consumption based on load percentage."""
        if load_percent < self.min_load_percent:
            return 0  # Don't run below minimum load
        
        # Quadratic consumption curve for better part-load efficiency
        norm_load = (load_percent - self.min_load_percent) / (1 - self.min_load_percent)
        consumption = (self.fuel_consumption_rate['idle'] + 
                      (self.fuel_consumption_rate['full_load'] - 
                       self.fuel_consumption_rate['idle']) * (0.8 * norm_load + 0.2 * norm_load**2))
        return consumption
    
    def calculate_efficiency(self, load_percent: float) -> float:
        """Calculate generator efficiency based on load percentage."""
        # Modern diesel generator efficiency curve
        if load_percent < self.min_load_percent:
            return 0
        elif load_percent > 0.9:
            return 0.42  # Higher peak efficiency
        else:
            # Peak efficiency around 80% load
            return 0.40 * np.sin(np.pi * (load_percent - 0.3) / 1.2) + 0.35
    
    def generate_output(self, df, load_demand: np.ndarray, 
                       pv_output: np.ndarray,
                       battery_output: np.ndarray) -> Dict[str, Any]:
        """Generate generator output parameters."""
        # Initialize arrays
        hours = len(df)
        fuel_level = np.zeros(hours)
        power = np.zeros(hours)  
        frequency = np.zeros(hours)
        temperature = np.zeros(hours)
        runtime = np.zeros(hours)  
        
        # Initial conditions
        fuel_level[0] = self.fuel_tank_capacity
        last_maintenance = 0
        cumulative_runtime = 0
        consecutive_runtime = 0  # Track consecutive runtime hours
        generator_on = False     # Track if generator is currently running
        
        for i in range(hours):
            # Calculate required power
            required_power = load_demand[i]  # This is now the remaining load after solar, battery, and grid
            
            # Calculate load percentage
            load_percent = min(required_power / self.capacity_kva, 1.0)
            
            # Determine if generator should run
            should_run = False
            
            # If generator is already running, maintain minimum runtime
            if generator_on and consecutive_runtime < self.min_runtime_hours:
                should_run = True
            # Otherwise, only run if load is significant and above minimum threshold
            elif load_percent >= self.min_load_percent:
                should_run = True
            # Special case: grid is unavailable and we have significant load
            elif 'grid_available' in df.columns and not df['grid_available'].iloc[i] and load_percent > 0.1:
                should_run = True
            
            if should_run:
                # Generator is running
                generator_on = True
                cumulative_runtime += 1
                consecutive_runtime += 1
                runtime[i] = cumulative_runtime
                
                # Calculate fuel consumption
                fuel_consumption = self.calculate_fuel_consumption(max(load_percent, self.min_load_percent))
                
                # Update fuel level
                if i > 0:
                    fuel_level[i] = fuel_level[i-1] - fuel_consumption
                
                # Automatic refill when below 20%
                if fuel_level[i] < 0.2 * self.fuel_tank_capacity:
                    fuel_level[i] = self.fuel_tank_capacity
                
                # Calculate output
                efficiency = self.calculate_efficiency(max(load_percent, self.min_load_percent))
                power[i] = required_power
                
                # Calculate frequency
                base_freq = 60 + 0.1 * (load_percent - 0.5)
                frequency[i] = base_freq + np.random.normal(0, 0.01)
                
                # Calculate temperature
                base_temp = 80 + 40 * load_percent
                temperature[i] = base_temp + np.random.normal(0, 2)
                
                # Maintenance effect
                hours_since_maintenance = cumulative_runtime - last_maintenance
                if hours_since_maintenance >= self.maintenance_interval:
                    last_maintenance = cumulative_runtime
                    efficiency *= 1.05  # Efficiency boost after maintenance
            else:
                # Generator is off
                generator_on = False
                consecutive_runtime = 0
                
                if i > 0:
                    fuel_level[i] = fuel_level[i-1]
                    
        return {
            'power': power,
            'fuel_level': fuel_level,
            'frequency': frequency,
            'temperature': temperature,
            'runtime': runtime  
        }
