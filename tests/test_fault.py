import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class FaultAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """Initialize the FaultAnalyzer with a DataFrame containing system and fault data."""
        required_columns = ['fault_type', 'fault_occurred']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"DataFrame must contain '{col}' column")
        self.df = df

    def _get_consecutive_periods(self, bool_series):
        """Helper method to identify consecutive periods of True values in a boolean series."""
        changes = bool_series.astype(int).diff()
        starts = np.where(changes == 1)[0]
        ends = np.where(changes == -1)[0]
        if bool_series.iloc[0]:
            starts = np.insert(starts, 0, 0)
        if bool_series.iloc[-1]:
            ends = np.append(ends, len(bool_series))
        return list(zip(starts, ends))

    def plot_time_series(self, parameters: list[str], fault_type: str = None):
        """Plot time series of specified parameters with fault periods highlighted."""
        plt.figure(figsize=(15, 5))
        for param in parameters:
            if param not in self.df.columns:
                raise ValueError(f"Parameter {param} not found in DataFrame")
            plt.plot(self.df.index, self.df[param], label=param)

        if fault_type:
            fault_periods = self.df['fault_type'] == fault_type
        else:
            fault_periods = self.df['fault_occurred']

        for start, end in self._get_consecutive_periods(fault_periods):
            plt.axvspan(self.df.index[start], self.df.index[end], color='red', alpha=0.3)

        plt.legend()
        plt.title(f"Time Series of Parameters with {'specific' if fault_type else 'any'} fault periods highlighted")
        plt.xlabel("Time")
        plt.ylabel("Parameter Value")
        plt.savefig('output_/time_series.png', dpi=300, bbox_inches='tight')

    def compare_distributions(self, parameter: str, fault_type: str):
        """Compare the distribution of a parameter during fault vs. no-fault periods."""
        if parameter not in self.df.columns:
            raise ValueError(f"Parameter {parameter} not found in DataFrame")
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df[parameter][self.df['fault_type'] == 'NO_FAULT'], label='No Fault', kde=True)
        sns.histplot(self.df[parameter][self.df['fault_type'] == fault_type], label=fault_type, kde=True)
        plt.legend()
        plt.title(f"Distribution of {parameter} during {fault_type} vs. No Fault")
        plt.xlabel(parameter)
        plt.ylabel("Density")
        #plt.show()
        plt.savefig('output_/distribution.png', dpi=300, bbox_inches='tight')

    def plot_fault_timeline(self):
        """Plot a timeline showing fault types over time."""
        plt.figure(figsize=(15, 3))
        fault_codes = {ftype: i for i, ftype in enumerate(self.df['fault_type'].unique())}
        fault_nums = self.df['fault_type'].map(fault_codes)
        plt.step(self.df.index, fault_nums, where='post')
        plt.yticks(list(fault_codes.values()), list(fault_codes.keys()))
        plt.title("Fault Types Over Time")
        plt.xlabel("Time")
        plt.ylabel("Fault Type")
        #plt.show()
        plt.savefig('output_/fault_timeline.png', dpi=300, bbox_inches='tight')

    def get_fault_events(self):
        """Extract individual fault events with start times, end times, and durations."""
        events = []
        current_fault = None
        start_time = None
        for i, row in self.df.iterrows():
            if row['fault_occurred']:
                if row['fault_type'] != current_fault:
                    if current_fault is not None:
                        events.append({
                            'fault_type': current_fault,
                            'start_time': start_time,
                            'end_time': i,
                            'duration': (i - start_time).total_seconds() / 3600
                        })
                    current_fault = row['fault_type']
                    start_time = i
            else:
                if current_fault is not None:
                    events.append({
                        'fault_type': current_fault,
                        'start_time': start_time,
                        'end_time': i,
                        'duration': (i - start_time).total_seconds() / 3600
                    })
                    current_fault = None
                    start_time = None
        if current_fault is not None:
            events.append({
                'fault_type': current_fault,
                'start_time': start_time,
                'end_time': self.df.index[-1],
                'duration': (self.df.index[-1] - start_time).total_seconds() / 3600
            })
        return pd.DataFrame(events)

    def fault_statistics(self):
        """Calculate statistics for fault events, including count and duration metrics."""
        events = self.get_fault_events()
        if events.empty:
            return pd.DataFrame(columns=['count', 'avg_duration', 'total_duration'])
        stats = events.groupby('fault_type').agg(
            count=('fault_type', 'size'),
            avg_duration=('duration', 'mean'),
            total_duration=('duration', 'sum')
        )
        return stats