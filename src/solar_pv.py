import numpy as np
from typing import Dict, Any
import pandas as pd

class SolarPVSimulator:
    """Simulates a 1500 kW Solar PV system with high-efficiency panels."""
    
    def __init__(self, capacity_kw: float = 1500, seed: int = 42):
        self.capacity_kw = capacity_kw
        np.random.seed(seed)
        
        # System parameters
        self.nominal_efficiency = 0.21  # High-efficiency panels
        self.temp_coefficient = -0.003  # Better temperature performance
        self.dust_loss_rate = 0.0005   # Improved cleaning schedule
        self.noct = 42  # Better thermal design
        self.base_efficiency = 0.23  # Premium panels
        self.rated_power = capacity_kw
        self.system_efficiency = 0.85  # Modern inverters
        
    def calculate_irradiance(self, df):
        """Calculate solar irradiance based on time and weather."""
        # Base irradiance pattern (daily)
        hour = df.index.hour
        base_irradiance = np.zeros(len(df))
        daytime_mask = (hour >= 6) & (hour <= 18)
        base_irradiance[daytime_mask] = 1000 * np.sin(np.pi * (hour[daytime_mask] - 6) / 12)
        
        # Seasonal variation
        seasonal_factor = 1 - 0.3 * np.sin(2 * np.pi * (df['weather_day_of_year'] + 81) / 365)
        
        # Apply cloud effects
        cloud_effect = 1 - df['weather_cloud_cover']
        
        # Combine factors
        irradiance = base_irradiance * seasonal_factor * cloud_effect
        
        return irradiance
    
    def calculate_cell_temperature(self, ambient_temp, irradiance):
        """Calculate PV cell temperature."""
        return ambient_temp + (self.noct - 20) * irradiance / 800
    
    def calculate_power(self, irradiance, cell_temp):
        """Calculate power output based on irradiance and cell temperature."""
        # Temperature effect on efficiency
        temp_factor = 1 + self.temp_coefficient * (cell_temp - 25)
        
        # Calculate power
        power = (
            self.rated_power * 
            (irradiance / 1000) * 
            temp_factor * 
            self.system_efficiency
        )
        
        return np.clip(power, 0, self.rated_power)
    
    def generate_output(self, df) -> Dict[str, Any]:
        """Generate PV system output parameters."""
        # Calculate irradiance
        irradiance = self.calculate_irradiance(df)
        
        # Calculate cell temperature
        cell_temp = self.calculate_cell_temperature(df['weather_temperature'], irradiance)
        
        # Calculate power output
        power = self.calculate_power(irradiance, cell_temp)
        
        return {
            'irradiance': irradiance,
            'cell_temp': cell_temp,
            'power': power
        }
