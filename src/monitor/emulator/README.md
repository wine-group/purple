# PURPLE-AIMD Emulator

This guide provides instructions for setting up and running the PURPLE-AIMD emulator. While not subject to the usual wireless dynamics, the emulator can still show how PURPLE or PURPLE-AIMD can dynamically adjust the CAKE bandwidth parameter based on the control inputs (i.e., backlog and drops for PURPLE, and latency for PURPLE-AIMD).

## System Requirements

For best results, set up a dedicated virtual machine:
- **Recommended specs**: Ubuntu 24.04.2 LTS, 8 GB RAM, 2+ vCPUs
- **Hypervisor**: VirtualBox, UTM (Mac), or equivalent
- Ubuntu Server can be downloaded from: https://ubuntu.com/download/server

## Overview of Components

The system consists of several components:

1. **Network Emulator** (`simple_bottleneck_emulator.sh`): Creates a three-node topology (Client → Radio → Server) with a configurable bottleneck
2. **PURPLE-AIMD Controller** (`purple_aimd_controller.py`): Core PURPLE-AIMD algorithm that implements Additive Increase/Multiplicative Decrease for CAKE bandwidth control
3. **Adapter** (`purple_aimd_adapter.py`): Connects the controller to the network environment
4. **Test Scenarios** (`bottleneck_test_scenarios.sh`): Pre-configured test cases to evaluate controller performance. These can be expanded based on your requirements
5. **Visualisation** (`latency_bandwidth_visualiser.py`): Creates plots to analyse the relationship between bandwidth and latency

## Installation Instructions

1. **Set up the virtual machine**
   ```bash
   # After installing Ubuntu Server, update the system
   sudo apt update
   sudo apt upgrade -y
   
   # Install required packages
   sudo apt install -y python3 python3-pip iproute2 iptables tcpdump iperf
   
   # Install Python dependencies
   pip3 install numpy matplotlib scipy pandas
   ```

2. **Clone or download all scripts to your home directory**
   ```bash
   # Create a directory for the project
   mkdir -p ~/purple-aimd
   cd ~/purple-aimd
   
   # Save each script to this directory
   # Make the bash scripts executable
   chmod +x *.sh
   ```

## Using the System

### Step 1: Set up the Network Emulation Environment

```bash
# Create the network topology
sudo ./simple_bottleneck_emulator.sh setup

# Verify the setup
sudo ./simple_bottleneck_emulator.sh status

# Test connectivity
sudo ip netns exec client_ns ping -c 3 192.168.200.2
```

### Step 2: Run the PURPLE-AIMD Controller

```bash
# Run the controller with default settings
sudo python3 purple_aimd_adapter.py

# Or customise settings (example)
sudo python3 purple_aimd_adapter.py --target 5.5 --min-rate 10.0 --max-rate 120.0 --initial-rate 95.0 --decay 0.05 --recovery 0.25
```

The controller will continuously measure latency and adjust the bandwidth parameter of CAKE in the client namespace.

### Step 3: Run Test Scenarios

```bash
# In a separate terminal, run one of the test scenarios
sudo ./bottleneck_test_scenarios.sh scenario1   # Stable bottleneck with increasing traffic
sudo ./bottleneck_test_scenarios.sh scenario2   # Fluctuating bottleneck capacity
sudo ./bottleneck_test_scenarios.sh scenario3   # Gradual capacity degradation
sudo ./bottleneck_test_scenarios.sh scenario4   # Bottleneck capacity steps
sudo ./bottleneck_test_scenarios.sh scenario5   # Rapid capacity changes

# Run all scenarios sequentially
sudo ./bottleneck_test_scenarios.sh all
```

Each scenario generates a CSV file with statistics about latency, bandwidth, backlogs, and drops.

### Step 4: Visualise and Analyse Results

```bash
# Create visualisation from collected data
python3 latency_bandwidth_visualiser.py --csv scenario1_stats.csv

# Analyse AIMD controller trajectory data
python3 latency_bandwidth_visualiser.py --data aimd_trajectory.json
```

This will generate PNG files with plots showing latency and bandwidth over time, and their relationship. These scripts are a work in progress, and might not generate the nicest looking graphics.

## Understanding How PURPLE-AIMD Works

PURPLE-AIMD uses the following algorithm to adjust bandwidth:

1. **Measure latency** between client and server
2. **Update bandwidth based on latency**:
   - If latency ≥ critical threshold: Aggressively decrease bandwidth (multiply by 1-β)
   - If latency > target + tolerance: Moderately decrease bandwidth (multiply by 1-β/2)
   - If latency ≤ target + tolerance and in recovery phase: Increase bandwidth (add α each interval)
   - Otherwise: Maintain current bandwidth

3. **Recovery phase** is entered after multiple consecutive good latency measurements

### Key Parameters

- `target_latency`: Desired latency (default: 5.0 ms)
- `critical_latency`: Threshold for aggressive reduction (default: 30.0 ms)
- `latency_tolerance`: Acceptable deviation from target (default: 2.0 ms)
- `decay_factor`: How much to reduce bandwidth when latency is high (default: 0.1)
- `recovery_rate`: How much to increase bandwidth per second (default: 0.5 Mbps)

## Cleaning Up

When finished, clean up the network environment:

```bash
sudo ./simple_bottleneck_emulator.sh clean
```

## Troubleshooting

- **Permission denied errors**: Ensure you're running scripts with sudo
- **Network namespace errors**: Make sure the emulator is set up before running controllers or tests
- **Python errors**: Verify all Python dependencies are installed
- **No visualisation**: Check that matplotlib and pandas are properly installed

## Example Workflow

1. Set up the network emulation environment
2. Run the PURPLE-AIMD controller in one terminal
3. Run a test scenario in another terminal
4. Observe how the controller adjusts bandwidth in response to latency changes
5. Visualise the results to analyse performance