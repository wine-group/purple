import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
from scipy import stats
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)

# Global dictionaries for colors and labels
colors = {
    'fq_codel': 'blue',
    'cake_100': 'green',
    'cake_150': 'lightgreen',
    'purple': 'red',
    'purple_esc': 'lightcoral'
}
labels = {
    'fq_codel': 'FQ-CoDel',
    'cake_100': 'CAKE 100Mbit',
    'cake_150': 'CAKE 150Mbit',
    'purple': 'PURPLE',
    'purple_esc': 'PURPLE ESC'
}

# Helper function to load and process data
def load_data(base_name, approaches):
    data = {}
    for approach in approaches:
        loaded_file = f'ping_results_loaded_{approach}.csv'
        unloaded_file = f'ping_results_unloaded_{approach}.csv'
        
        try:
            data[approach] = {
                'loaded': pd.read_csv(loaded_file, parse_dates=['Timestamp']).dropna(),
                'unloaded': pd.read_csv(unloaded_file, parse_dates=['Timestamp']).dropna()
            }
        except Exception as e:
            print(f"Error loading {approach}: {e}")
    return data

def calculate_stats(data):
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'min': np.min(data),
        'max': np.max(data),
        'std': np.std(data)
    }

def plot_timeseries(data, approach, save_path):
    """Create time series plot for a single approach."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot unloaded data
    ax1.plot(data[approach]['unloaded']['Timestamp'], 
             data[approach]['unloaded']['Ping Time (ms)'],
             label='Unloaded', color='green', alpha=0.7)
    ax1.set_ylabel('RTT (ms)')
    ax1.set_title(f'ICMP RTT Unloaded ({labels[approach]})')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    
    # Plot loaded data
    ax2.plot(data[approach]['loaded']['Timestamp'],
             data[approach]['loaded']['Ping Time (ms)'],
             label='Loaded', color='blue', alpha=0.7)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('RTT (ms)')
    ax2.set_title(f'ICMP RTT Loaded ({labels[approach]})')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/timeseries_{approach}.eps')
    plt.close()

def plot_cdfs(data, approaches, save_path):
    """Create CDF comparison plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Using global colors and labels dictionaries
    
    # Plot unloaded CDFs
    for approach in approaches:
        unloaded_data = data[approach]['unloaded']['Ping Time (ms)']
        sorted_data = np.sort(unloaded_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax1.plot(sorted_data, cdf, label=labels[approach], color=colors[approach])
    
    ax1.set_xlabel('RTT (ms)')
    ax1.set_ylabel('CDF')
    ax1.set_title('Unloaded RTT Distribution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot loaded CDFs
    for approach in approaches:
        loaded_data = data[approach]['loaded']['Ping Time (ms)']
        sorted_data = np.sort(loaded_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax2.plot(sorted_data, cdf, label=labels[approach], color=colors[approach])
    
    ax2.set_xlabel('RTT (ms)')
    ax2.set_ylabel('CDF')
    ax2.set_title('Loaded RTT Distribution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/cdf_comparison.eps')
    plt.close()

def plot_pdfs(data, approaches, save_path):
    """Create PDF comparison plot using kernel density estimation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    # Using global colors and labels dictionaries
    
    # Plot unloaded PDFs
    for approach in approaches:
        unloaded_data = data[approach]['unloaded']['Ping Time (ms)']
        kde = stats.gaussian_kde(unloaded_data)
        x_range = np.linspace(min(unloaded_data), max(unloaded_data), 200)
        ax1.plot(x_range, kde(x_range), label=labels[approach], color=colors[approach])
    
    ax1.set_xlabel('RTT (ms)')
    ax1.set_ylabel('Density')
    ax1.set_title('Unloaded RTT Density')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot loaded PDFs
    for approach in approaches:
        loaded_data = data[approach]['loaded']['Ping Time (ms)']
        kde = stats.gaussian_kde(loaded_data)
        x_range = np.linspace(min(loaded_data), max(loaded_data), 200)
        ax2.plot(x_range, kde(x_range), label=labels[approach], color=colors[approach])
    
    ax2.set_xlabel('RTT (ms)')
    ax2.set_ylabel('Density')
    ax2.set_title('Loaded RTT Density')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/pdf_comparison.eps')
    plt.close()

def plot_boxplots(data, approaches, save_path):
    """Create boxplot comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for boxplots
    unloaded_data = []
    loaded_data = []
    plot_labels = []
    
    for approach in approaches:
        unloaded_data.append(data[approach]['unloaded']['Ping Time (ms)'])
        loaded_data.append(data[approach]['loaded']['Ping Time (ms)'])
        plot_labels.append(labels[approach])
    
    # Plot unloaded boxplots
    ax1.boxplot(unloaded_data, labels=plot_labels)
    ax1.set_xlabel('Approach')
    ax1.set_ylabel('RTT (ms)')
    ax1.set_title('Unloaded RTT Distribution')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loaded boxplots
    ax2.boxplot(loaded_data, labels=plot_labels)
    ax2.set_xlabel('Approach')
    ax2.set_ylabel('RTT (ms)')
    ax2.set_title('Loaded RTT Distribution')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/boxplot_comparison.eps')
    plt.close()

def main():
    # Define approaches
    approaches = ['fq_codel', 'cake_100', 'cake_150', 'purple', 'purple_esc']
    save_path = 'plots'
    
    # Load data
    data = load_data('ping_results', approaches)
    
    # Create individual time series plots
    for approach in approaches:
        plot_timeseries(data, approach, save_path)
    
    # Create comparison plots
    plot_cdfs(data, approaches, save_path)
    plot_pdfs(data, approaches, save_path)
    plot_boxplots(data, approaches, save_path)
    
    # Print statistics
    for approach in approaches:
        print(f"\nStatistics for {approach}:")
        print("\nUnloaded Network Statistics:")
        unloaded_stats = calculate_stats(data[approach]['unloaded']['Ping Time (ms)'])
        for key, value in unloaded_stats.items():
            print(f"{key.capitalize()}: {value:.2f} ms")
        
        print("\nLoaded Network Statistics:")
        loaded_stats = calculate_stats(data[approach]['loaded']['Ping Time (ms)'])
        for key, value in loaded_stats.items():
            print(f"{key.capitalize()}: {value:.2f} ms")

if __name__ == "__main__":
    main()