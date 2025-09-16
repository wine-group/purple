# PURPLE Quick Start Experimentation Guide

This guide provides step-by-step instructions for setting up and running PURPLE experiments in a testbed environment with Ubiquiti wireless radios that run airOS. PURPLE-AIMD can be modified to run with other radios by removing the airOS-specific web API calls which are currently used for monitoring.

## Prerequisites

### Hardware Requirements
- Unix-based control machine (Ubuntu Desktop, Ubuntu Server, Raspberry Pi, etc.)
- Ubiquiti wireless radio pair configured as point-to-point link
- Two endpoint devices (e.g., Raspberry Pi computers or similar)
- Network connectivity between all devices

### Software Requirements
- Python 3.x with required libraries (see installation section)
- SSH access to all devices
- iPerf installed on endpoint devices
- CAKE qdisc support on the gateway device

## Installation

### Install Required Python Libraries

On your control machine:

```bash
# Install Python dependencies
pip3 install paramiko requests pytz numpy urllib3

# Clone or download the PURPLE repository
git clone https://github.com/wine-group/purple.git
cd purple/src/monitor
```

## Network Configuration

### Typical Network Topology

TBC...

### Device Access Configuration

You'll need the following information:

**Wireless Radio (Bottleneck):**
- IP Address: `<RADIO_IP>`
- Web Interface Username: `<WEB_USERNAME>`
- Web Interface Password: `<WEB_PASSWORD>`
- SSH Username: `<SSH_USERNAME>`
- SSH Password or Key Path: `<SSH_PASSWORD>` or `<KEY_PATH>`

**Gateway Device (where CAKE is configured):**
- IP Address: `<GATEWAY_IP>`
- SSH Username: `<GATEWAY_USER>`
- SSH Password or Key Path: `<GATEWAY_PASSWORD>` or `<KEY_PATH>`

**Endpoint Devices:**
- Server IP: `<SERVER_IP>`
- Client IP: `<CLIENT_IP>`
- SSH Credentials for each

## Step-by-Step Instructions

### 1. Initial CAKE Configuration

On the gateway device (typically the endpoint behind the bottleneck link), configure CAKE with an initial bandwidth:

```bash
# SSH into the gateway device
ssh <USERNAME>@<GATEWAY_IP>

# Set initial CAKE bandwidth (adjust interface name as needed)
sudo tc qdisc replace dev eth0 root cake ethernet bandwidth 150mbit

# Verify configuration
tc qdisc show dev eth0
```

### 2. Start PURPLE-AIMD Controller

On your control machine, navigate to the PURPLE directory and start the controller (set sane minimum/maximum bandwidth values, as well as a target latency):

```bash
cd /path/to/purple/src/monitor

# Basic command structure
python3 purple_aimd_adapter.py \
  --radio-host <RADIO_IP> \
  --radio-web-user <WEB_USERNAME> \
  --radio-web-password '<WEB_PASSWORD>' \
  --radio-ssh-user <SSH_USERNAME> \
  --radio-ssh-password '<SSH_PASSWORD>' \
  --gateway-host <GATEWAY_IP> \
  --gateway-user <GATEWAY_USER> \
  --gateway-key <KEY_PATH> \
  --min-rate 50 \
  --initial-rate 100 \
  --target 5
```

**Example with typical values:**
```bash
python3 purple_aimd_adapter.py \
  --radio-host 192.168.1.20 \
  --radio-web-user admin \
  --radio-web-password 'YourPassword123' \
  --radio-ssh-user admin \
  --radio-ssh-password 'YourPassword123' \
  --gateway-host 192.168.1.100 \
  --gateway-user pi \
  --gateway-key ~/.ssh/id_rsa \
  --min-rate 50 \
  --initial-rate 100 \
  --max-rate 200 \
  --target 5
```

**Note:** Use `Ctrl+C` to stop PURPLE-AIMD when the experiment is complete.

#### Key PURPLE-AIMD Parameters

- `--target`: Target latency in milliseconds (default: 10)
- `--min-rate`: Minimum bandwidth in Mbps (default: 10)
- `--max-rate`: Maximum bandwidth in Mbps (default: 100)
- `--initial-rate`: Starting bandwidth in Mbps (default: 50)
- `--decay`: Multiplicative decrease factor (default: 0.05)
- `--recovery`: Additive increase rate in Mbps/sec (default: 0.25)
- `--tolerance`: Latency tolerance in ms (default: 2.0)
- `--interval`: Sample interval in seconds (default: 0.1)

### 3. Monitor Latency (Optional)

To monitor latency during experiments, you can use the included ping utility:

```bash
cd /path/to/purple/src/ping

# Run ping utility (default: 300 seconds)
python3 ping_util.py --target <TARGET_IP> --duration 300
```

### 4. Traffic Generation with iPerf

#### Start iPerf Server

On the endpoint device that will receive traffic:

```bash
# SSH into the server endpoint
ssh <USERNAME>@<SERVER_IP>

# Start iPerf server in TCP mode
iperf -s

# OR start in UDP mode with specified bandwidth
iperf -s -u -b 100M
```

#### Start iPerf Client

On the endpoint device that will generate traffic:

```bash
# SSH into the client endpoint
ssh <USERNAME>@<CLIENT_IP>

# TCP test for 320 seconds
iperf -c <SERVER_IP> -t 320

# UDP test with specified bandwidth
iperf -c <SERVER_IP> -u -b 100M -t 320
```

### 5. Data Collection and Visualisation

PURPLE-AIMD automatically logs data to CSV files and trajectory data:

```bash
# Default output files in the monitor directory
radio_stats.csv          # Radio statistics and latency data

# Generate plots from collected data
cd /path/to/purple/src/monitor
python3 radio_rate_plotter.py

# View generated plot
# Output: radio_stats_simple.png
```

**Important:** Rename output files between runs to prevent overwriting:

```bash
# Save results with timestamp
timestamp=$(date +%Y%m%d_%H%M%S)
mv radio_stats.csv radio_stats_${timestamp}.csv
mv radio_stats_simple.png radio_stats_simple_${timestamp}.png
mv aimd_trajectory.json aimd_trajectory_${timestamp}.json
```

## Typical Experiment Workflow

1. **Prepare Environment**
   - Verify all devices are accessible via SSH
   - Check network connectivity between components
   - Ensure CAKE is available on the gateway device

2. **Configure Initial State**
   - Set initial CAKE bandwidth on gateway
   - Note current network conditions

3. **Start Monitoring**
   - Launch PURPLE-AIMD controller on control machine
   - (Optional) Start latency monitoring utility

4. **Generate Traffic**
   - Start iPerf server on receiving endpoint
   - Start iPerf client on sending endpoint
   - Monitor console output for status updates

5. **Complete Experiment**
   - Wait for iPerf test to complete
   - Stop PURPLE-AIMD with `Ctrl+C`
   - Save and rename output files

6. **Analyse Results**
   - Generate plots from CSV data
   - Compare with baseline measurements


## Output File Formats

### radio_stats.csv
Contains timestamped measurements:
- `timestamp`: ISO format timestamp
- `tx_rate`: Transmission rate (Mbps)
- `rx_rate`: Receive rate (Mbps)
- `radio_capacity`: Radio reported capacity (Mbps)
- `backlog`: Queue backlog (bytes)
- `drops`: Packet drops count
- `useful_capacity`: PURPLE-calculated capacity (Mbps)
- `state`: Controller state (NORMAL/HIGH/CRITICAL/RECOVERY)
- `latency`: Measured latency (ms)
