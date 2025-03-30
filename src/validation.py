import numpy as np
import pandas as pd
from typing import Dict, Any

class DataValidator:
    """Validates and cleans the generated data."""
    
    def __init__(self):
        # Define valid ranges for different parameters
        self.valid_ranges = {
            'weather_temperature': (-10, 45),  # °C
            'weather_humidity': (0, 100),  # %
            'weather_cloud_cover': (0, 100),  # %
            'weather_wind_speed': (0, 30),  # m/s
            'solar_power': (0, 1500),  # kW
            'solar_cell_temp': (0, 85),  # °C
            'battery_soc': (0, 1),  # 0-1
            'battery_power': (-200, 200),  # kW
            'battery_voltage': (350, 450),  # V
            'battery_current': (-500, 500),  # A
            'battery_temperature': (0, 60),  # °C
            'generator_power': (0, 2000),  # kW
            'generator_fuel_level': (0, 5000),  # L
            'generator_frequency': (55, 65),  # Hz
            'generator_temperature': (0, 120),  # °C
            'grid_voltage': (0.8 * 25000, 1.2 * 25000),  # V
            'grid_frequency': (48, 52),  # Hz
            'grid_power': (-2000, 2000),  # kW
            'load_demand': (0, 2000),  # kW
            'load_power_factor': (0.8, 1.0),  # unitless
            'fault_severity': (0, 1)  # unitless
        }
    
    def validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean the data, ensuring all values are within valid ranges."""
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Clip numerical values to valid ranges
        for column, (min_val, max_val) in self.valid_ranges.items():
            if column in df.columns:
                df[column] = df[column].clip(min_val, max_val)
        
        # Validate power balance
        total_generation = (
            df['solar_power'] +
            df['generator_power'] +
            df['grid_power'] +
            df['battery_power']
        )
        total_load = df['load_demand']
        
        # Allow for small imbalances (1% of max load)
        tolerance = 0.01 * df['load_demand'].max()
        df['power_balanced'] = (total_generation - total_load).abs() <= tolerance
        
        # Check for NaN values
        if df.isna().any().any():
            print("Warning: NaN values found in dataset")
            df = df.fillna(0)  # Replace NaN with 0
        
        return df
    
    def check_power_balance(self, data: Dict[str, Any], tolerance: float = 0.01) -> bool:
        """
        Verify that power generation matches load demand within tolerance.
        Returns True if balance is maintained, False otherwise.
        """
        total_generation = 0
        
        # Sum up all generation sources
        if 'solar' in data and 'power_output' in data['solar']:
            total_generation += data['solar']['power_output']
        
        if 'generator' in data and 'output_power' in data['generator']:
            total_generation += data['generator']['output_power']
        
        if 'grid' in data and 'available' in data['grid']:
            grid_power = np.where(
                data['grid']['available'],
                data['load']['active_power'] - total_generation,
                0
            )
            total_generation += grid_power
        
        if 'battery' in data and 'power_output' in data['battery']:
            total_generation += data['battery']['power_output']
        
        # Get load demand
        load_demand = data['load']['active_power']
        
        # Check balance
        imbalance = np.abs(total_generation - load_demand)
        max_allowed_imbalance = load_demand * tolerance
        
        return np.all(imbalance <= max_allowed_imbalance)
