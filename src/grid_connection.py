import numpy as np
from typing import Dict, Any

class GridConnectionSimulator:
    """Simulates a 25kV grid connection with realistic behavior."""
    
    def __init__(self, nominal_voltage: float = 25000, seed: int = 42):
        self.nominal_voltage = nominal_voltage
        np.random.seed(seed)
        
        # Grid parameters
        self.base_reliability = 0.98
        self.voltage_variation = 0.02  # ±2% normal variation
        self.peak_hours = list(range(18, 22))  # 6 PM to 10 PM
        self.maintenance_schedule = []  # Will be populated during simulation
        
    def is_peak_hour(self, hour: int) -> bool:
        """Check if current hour is during peak demand."""
        return hour in self.peak_hours
    
    def calculate_reliability(self, season: str, hour: int) -> float:
        """Calculate grid reliability based on season and time."""
        # Base reliability
        reliability = self.base_reliability
        
        # Seasonal effects
        season_factors = {
            'long_rains': 0.95,   # More outages during long rains
            'short_rains': 0.97,  # Some outages during short rains
            'dry': 0.99           # Better reliability in dry season
        }
        reliability *= season_factors.get(season, 1.0)
        
        # Time of day effects
        if self.is_peak_hour(hour):
            reliability *= 0.97  # Lower reliability during peak hours
        
        return reliability
    
    def generate_output(self, df) -> Dict[str, Any]:
        """Generate grid connection parameters."""
        hours = len(df)
        
        # Initialize arrays
        voltage = np.zeros(hours)
        frequency = np.zeros(hours)
        available = np.zeros(hours, dtype=bool)
        power_quality = np.zeros(hours)
        power = np.zeros(hours)  # Power output (positive = import, negative = export)
        
        # Generate planned maintenance schedule (every 90 days)
        maintenance_days = np.arange(0, df.index[-1].dayofyear, 90)
        
        for i in range(hours):
            current_time = df.index[i]
            hour = current_time.hour
            season = df['weather_season'].iloc[i]
            
            # Check if time is during maintenance
            is_maintenance = current_time.dayofyear in maintenance_days and 8 <= hour <= 16
            
            # Determine grid availability
            reliability = self.calculate_reliability(season, hour)
            is_available = np.random.random() < reliability and not is_maintenance
            available[i] = is_available
            
            if is_available:
                # Calculate base voltage with daily pattern
                base_voltage = self.nominal_voltage + \
                             500 * np.sin(2 * np.pi * hour / 24)
                
                # Add random variations
                voltage_noise = np.random.normal(0, self.voltage_variation * self.nominal_voltage)
                voltage[i] = base_voltage + voltage_noise
                
                # Calculate grid frequency (nominal 50 Hz in Kenya)
                frequency[i] = 50 + np.random.normal(0, 0.1)
                
                # Calculate power quality (1 = perfect, 0 = poor)
                quality_factors = [
                    0.7 + 0.3 * np.random.random(),  # Base quality
                    0.9 if not self.is_peak_hour(hour) else 0.7,  # Time of day
                    0.9 if season == 'dry' else 0.8  # Season
                ]
                power_quality[i] = np.prod(quality_factors)
                
                # Calculate grid power based on remaining load
                # Positive = importing from grid, negative = exporting to grid
                if 'remaining_load' in df.columns:
                    # If there's remaining load, grid supplies it
                    remaining_load = df['remaining_load'].iloc[i]
                    
                    # During peak hours, limit grid import to encourage battery/generator use
                    if self.is_peak_hour(hour):
                        # Reduce grid usage during peak hours by 30%
                        power[i] = remaining_load * 0.7
                    else:
                        power[i] = remaining_load
                else:
                    # Fallback calculation if remaining_load not available
                    load = df['load_demand'].iloc[i]
                    solar = df['solar_power'].iloc[i] if 'solar_power' in df.columns else 0
                    battery = df['battery_power'].iloc[i] if 'battery_power' in df.columns else 0
                    
                    # Calculate power balance
                    power_balance = load - (solar + battery)
                    
                    # If power_balance > 0, we need to import from grid
                    # If power_balance < 0, we can export to grid
                    if power_balance > 0:
                        # During peak hours, limit grid import
                        if self.is_peak_hour(hour):
                            power[i] = power_balance * 0.7
                        else:
                            power[i] = power_balance
                    else:
                        # Allow export to grid (negative value)
                        # Limit export based on grid capacity and regulations
                        max_export = 500  # Maximum export capacity in kW
                        power[i] = max(-max_export, power_balance)
            else:
                voltage[i] = 0
                frequency[i] = 0
                power_quality[i] = 0
                power[i] = 0
            
        # Store maintenance days in the class for future reference
        self.maintenance_schedule = maintenance_days
            
        return {
            'voltage': voltage,
            'frequency': frequency,
            'available': available,
            'power_quality': power_quality,
            'power': power
        }
