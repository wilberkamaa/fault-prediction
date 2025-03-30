import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional

from .weather import WeatherSimulator
from .solar_pv import SolarPVSimulator
from .diesel_generator import DieselGeneratorSimulator
from .battery_system import BatterySystemSimulator
from .grid_connection import GridConnectionSimulator
from .load_profile import LoadProfileGenerator
from .fault_injection import FaultInjectionSystem
from .validation import DataValidator

class HybridSystemDataGenerator:
    """Main class for generating synthetic data for the hybrid energy system."""
    
    def __init__(self, seed: int = 42):
        """Initialize all system components."""
        self.seed = seed
        np.random.seed(seed)
        
        # Initialize components
        self.weather_sim = WeatherSimulator(seed=seed)
        self.solar_sim = SolarPVSimulator(seed=seed)
        self.generator_sim = DieselGeneratorSimulator(seed=seed)
        self.battery_sim = BatterySystemSimulator(seed=seed)
        self.grid_sim = GridConnectionSimulator(seed=seed)
        self.load_gen = LoadProfileGenerator(seed=seed)
        self.fault_sim = FaultInjectionSystem(seed=seed)
        self.validator = DataValidator()
        
    def generate_dataset(self, 
                        start_date: str = '2023-01-01',
                        periods_years: int = 2,
                        output_file: Optional[str] = None) -> pd.DataFrame:
        """
        Generate a complete dataset for the specified time period.
        
        Args:
            start_date: Start date for the dataset
            periods_years: Number of years to simulate
            output_file: Optional path to save the dataset
            
        Returns:
            DataFrame containing all system parameters and fault labels
        """
        print("Generating time series base...")
        # Generate time series
        hours = periods_years * 365 * 24
        dates = pd.date_range(start=start_date, periods=hours, freq='H')
        df = pd.DataFrame(index=dates)
        
        # Add temporal features for weather
        df['weather_hour'] = df.index.hour
        df['weather_day_of_year'] = df.index.dayofyear
        df['weather_is_weekend'] = df.index.weekday >= 5
        
        # Add temporal features for load
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        df['is_weekend'] = df.index.weekday >= 5
        
        # Add cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        # Add Kenyan seasons
        def get_season(date):
            month = date.month
            if 3 <= month <= 5:
                return 'long_rains'
            elif 10 <= month <= 12:
                return 'short_rains'
            else:
                return 'dry'
        
        df['weather_season'] = df.index.map(get_season)
        df['season'] = df['weather_season']  # Keep a copy for other components
        
        print("Generating weather conditions...")
        # Generate weather conditions FIRST
        weather_data = self.weather_sim.generate_weather(df)
        for key, value in weather_data.items():
            df[f'weather_{key}'] = value
        
        print("Generating load profile...")
        # Generate load profile
        load_data = self.load_gen.generate_load(df)
        for key, value in load_data.items():
            df[f'load_{key}'] = value
        
        print("Simulating solar PV system...")
        # Generate solar PV output (now has access to weather data)
        solar_data = self.solar_sim.generate_output(df)
        for key, value in solar_data.items():
            df[f'solar_{key}'] = value
            
        # Initialize remaining power needs after solar
        df['remaining_load'] = load_data['demand'] - solar_data['power']
        df['remaining_load'] = df['remaining_load'].clip(lower=0)  # Only consider deficit
        
        print("Simulating battery system...")
        # Generate battery parameters - now takes remaining load after solar
        battery_data = self.battery_sim.generate_output(
            df,
            solar_data['power'],  
            load_data['demand']   
        )
        for key, value in battery_data.items():
            df[f'battery_{key}'] = value
            
        # Update remaining power needs after battery
        df['remaining_load'] = df['remaining_load'] - battery_data['power']
        df['remaining_load'] = df['remaining_load'].clip(lower=0)  # Only consider deficit
        
        print("Simulating grid connection...")
        # Generate grid parameters - now takes remaining load after solar and battery
        # Grid will only provide power when it's available
        grid_data = self.grid_sim.generate_output(df)
        for key, value in grid_data.items():
            df[f'grid_{key}'] = value
            
        # Update remaining power needs after grid
        df['remaining_load'] = df['remaining_load'] - grid_data['power']
        df['remaining_load'] = df['remaining_load'].clip(lower=0)  # Only consider deficit
        
        print("Simulating diesel generator...")
        # Generate generator parameters - only for remaining load after solar, battery, and grid
        generator_data = self.generator_sim.generate_output(
            df,
            df['remaining_load'],  # Only the remaining load after other sources
            solar_data['power'],  
            battery_data['power'] 
        )
        for key, value in generator_data.items():
            df[f'generator_{key}'] = value
        
        # Clean up intermediate calculation columns
        if 'remaining_load' in df.columns:
            df = df.drop(columns=['remaining_load'])
        
        print("Injecting faults...")
        # Generate fault events
        system_state = {
            'grid_voltage': df['grid_voltage'],
            'inverter_temp': df['solar_cell_temp'],
            'generator_runtime': df['generator_runtime'],
            'battery_soc': df['battery_soc']
        }
        fault_data = self.fault_sim.generate_fault_events(df, system_state)
        for key, value in fault_data.items():
            df[f'fault_{key}'] = value
        
        print("Validating data...")
        # Validate and clean data
        df = self.validator.validate_and_clean(df)
        
        # Save if output file provided
        if output_file:
            print(f"Saving dataset to {output_file}...")
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_file)
        
        return df
