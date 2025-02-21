import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import seaborn as sns
import numpy as np

def load_and_process_data(filepath):
    """Load and process the radio statistics data."""
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df

def plot_radio_stats(df, save_path='radio_stats_simple.png'):
    """Generate simple visualization for radio capacity and rates."""
    # Set up the style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.2)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot capacity and rates
    ax.plot(df['timestamp'], df['radio_capacity'], 
            label='TX Capacity', color='gray', alpha=0.5)
    ax.plot(df['timestamp'], df['tx_rate'],
            label='TX Rate', color='blue', alpha=0.7)
    ax.plot(df['timestamp'], df['useful_capacity'],
            label='Useful TX Capacity', color='red', linewidth=2)
    
    # Customize plot
    ax.set_title('TX Capacity and TX Rate Measurements')
    ax.set_xlabel('Time')
    ax.set_ylabel('Rate (Mbps)')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    
    # Layout adjustments
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def print_statistics(df):
    """Print summary statistics of the capacity and rate data."""
    print("\nCapacity and Rate Statistics (Mbps):")
    print("===================================")
    
    for column in ['radio_capacity', 'tx_rate', 'useful_capacity']:
        if column in df.columns:
            stats = df[column].describe()
            print(f"\n{column.replace('_', ' ').title()}:")
            print(f"  Mean: {stats['mean']:.2f}")
            print(f"  Min:  {stats['min']:.2f}")
            print(f"  Max:  {stats['max']:.2f}")
            print(f"  95th percentile: {np.percentile(df[column], 95):.2f}")

def main():
    # File path
    filepath = 'radio_stats.csv'
    
    try:
        # Load and process data
        print("Loading and processing data...")
        df = load_and_process_data(filepath)
        
        # Generate visualization
        print("Generating visualization...")
        plot_radio_stats(df)
        
        # Print statistics
        print_statistics(df)
        
        print("\nAnalysis complete! Check radio_stats_simple.png for the visualization.")
        
    except Exception as e:
        print(f"Error during analysis: {e}")

if __name__ == "__main__":
    main()