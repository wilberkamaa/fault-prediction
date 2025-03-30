import numpy as np
from typing import Dict, Any

class WeatherSimulator:
    """Simulates weather conditions for Kenya's climate."""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
        # Kenya's seasonal parameters
        self.season_params = {
            'long_rains': {'cloud_cover': (0.4, 0.6), 'temp_range': (20, 28)},
            'short_rains': {'cloud_cover': (0.3, 0.5), 'temp_range': (22, 30)},
            'dry': {'cloud_cover': (0.1, 0.3), 'temp_range': (25, 33)}
        }

    def generate_weather(self, df) -> Dict[str, Any]:
        """
        Generate weather conditions based on time and season.
        
        Args:
            df: DataFrame with datetime index and 'weather_season' column
            
        Returns:
            Dictionary containing weather parameters
        """
        # Base temperature pattern (daily cycle)
        temp_base = 25 + 5 * np.sin(2 * np.pi * (df['weather_hour'] - 14) / 24)  # Peak at 2 PM
        
        # Add seasonal variation
        season_temp_offset = {
            'long_rains': -2,
            'short_rains': 0,
            'dry': 2
        }
        temp_seasonal = df['weather_season'].map(season_temp_offset)
        
        # Add random variations
        temp_noise = np.random.normal(0, 0.5, len(df))
        temperature = temp_base + temp_seasonal + temp_noise
        
        # Generate cloud cover based on season and time of day
        cloud_cover = np.zeros(len(df))
        for i in range(len(df)):
            season = df['weather_season'].iloc[i]
            base_prob = np.random.uniform(*self.season_params[season]['cloud_cover'])
            # More clouds in early morning and late afternoon
            hour = df['weather_hour'].iloc[i]
            hour_factor = 0.2 * np.sin(2 * np.pi * (hour - 6) / 12)
            cloud_cover[i] = np.clip(base_prob + hour_factor, 0, 1)
        
        # Generate humidity
        humidity_base = 60 + 20 * np.sin(2 * np.pi * df['weather_hour'] / 24)
        humidity = humidity_base + 10 * cloud_cover + np.random.normal(0, 2, len(df))
        humidity = np.clip(humidity, 30, 100)
        
        # Generate wind speed (m/s)
        wind_speed = np.random.weibull(2, len(df)) * 3  # Weibull distribution for wind
        
        return {
            'temperature': temperature,
            'cloud_cover': cloud_cover,
            'humidity': humidity,
            'wind_speed': wind_speed
        }
