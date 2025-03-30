import numpy as np
from typing import Dict, Any
import pandas as pd

class LoadProfileGenerator:
    """Generates realistic load profiles for a hybrid energy system."""
    
    def __init__(self, base_load_kw: float = 500, peak_load_kw: float = 2000, seed: int = 42):
        self.base_load_kw = base_load_kw
        self.peak_load_kw = peak_load_kw
        np.random.seed(seed)
        
        # Load parameters
        self.weekday_factors = {
            'morning_peak': {'hours': (6, 9), 'factor': 1.3},
            'evening_peak': {'hours': (18, 22), 'factor': 1.5},
            'night_valley': {'hours': (23, 5), 'factor': 0.7}
        }
        self.weekend_reduction = 0.8
        
    def is_holiday(self, date: pd.Timestamp) -> bool:
        """Check if date is a Kenyan holiday."""
        holidays = [
            # Major Kenyan holidays
            (1, 1),    # New Year's Day
            (5, 1),    # Labour Day
            (6, 1),    # Madaraka Day
            (10, 20),  # Mashujaa Day
            (12, 12),  # Jamhuri Day
            (12, 25),  # Christmas Day
            (12, 26),  # Boxing Day
        ]
        return (date.month, date.day) in holidays
    
    def get_time_factor(self, hour: int, is_weekend: bool) -> float:
        """Calculate load factor based on time of day and week."""
        if is_weekend:
            # Weekend pattern
            if 8 <= hour <= 20:  # Active hours
                return 0.9
            else:  # Night hours
                return 0.6
        else:
            # Weekday pattern
            for period, info in self.weekday_factors.items():
                start, end = info['hours']
                if start <= hour < end:
                    return info['factor']
            return 1.0  # Default factor
    
    def get_seasonal_factor(self, season: str) -> float:
        """Calculate load factor based on season."""
        seasonal_factors = {
            'long_rains': 0.9,   # Lower demand during long rains
            'short_rains': 0.95,  # Slightly lower demand during short rains
            'dry': 1.1           # Higher demand during dry season
        }
        return seasonal_factors.get(season, 1.0)
    
    def generate_load(self, df) -> Dict[str, Any]:
        """Generate load profile with various factors."""
        hours = len(df)
        
        # Initialize arrays
        load_demand = np.zeros(hours)
        power_factor = np.zeros(hours)
        
        # Generate base load with random walk
        random_walk = np.cumsum(np.random.normal(0, 0.02, hours))
        random_walk = (random_walk - random_walk.min()) / (random_walk.max() - random_walk.min())
        
        for i in range(hours):
            current_time = df.index[i]
            hour = current_time.hour
            is_weekend = current_time.weekday() >= 5
            is_holiday = self.is_holiday(current_time)
            season = df['weather_season'][i]  # Use weather_season for consistency
            
            # Calculate various factors
            time_factor = self.get_time_factor(hour, is_weekend or is_holiday)
            seasonal_factor = self.get_seasonal_factor(season)
            
            # Calculate base load
            base_pattern = self.base_load_kw + \
                         (self.peak_load_kw - self.base_load_kw) * \
                         (0.5 + 0.5 * np.sin(np.pi * (hour - 6) / 12))
            
            # Combine all factors
            load = base_pattern * time_factor * seasonal_factor
            
            # Add random variations
            load *= (1 + 0.1 * random_walk[i])
            
            # Apply weekend reduction if applicable
            if is_weekend or is_holiday:
                load *= self.weekend_reduction
            
            load_demand[i] = load
            
            # Generate power factor (typically 0.8 to 0.95)
            base_pf = 0.85 + 0.1 * np.sin(2 * np.pi * hour / 24)
            power_factor[i] = base_pf + np.random.normal(0, 0.02)
        
        # Clip power factor to realistic range
        power_factor = np.clip(power_factor, 0.8, 0.95)
        
        return {
            'demand': load_demand,  # Changed from active_power to demand
            'power_factor': power_factor
        }
