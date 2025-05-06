#!/bin/bash
# Bottleneck Test Scenarios
#
# This script runs various test scenarios for the simplified bottleneck environment
# to evaluate the ESC controller's ability to manage bandwidth.
#
# Usage: ./bottleneck_test_scenarios.sh <scenario_name>
#

set -e

# Namespaces
CLIENT_NS="client_ns"
SERVER_NS="server_ns"
RADIO_NS="radio_ns"

# Test duration parameters
DURATION=300  # Test duration in seconds
UPDATE_INTERVAL=10  # How often to change network conditions

# Function to check if emulator is running
check_environment() {
    if ! ip netns list | grep -q $RADIO_NS; then
        echo "Error: Network emulation environment not found."
        echo "Please run 'sudo ./simple_bottleneck_emulator.sh setup' first."
        exit 1
    fi
}

# Function to run the traffic generator
start_traffic() {
    local duration=$1
    local rate=$2
    
    echo "Starting iperf traffic generator (${rate}Mbps for ${duration}s)..."
    
    # Start iperf server in server namespace
    ip netns exec $SERVER_NS iperf3 -s -D
    
    # Start iperf client in client namespace
    ip netns exec $CLIENT_NS iperf3 -c 192.168.200.2 -t $duration -b ${rate}M &
    
    echo "Traffic generator started"
}

# Function to stop traffic generator
stop_traffic() {
    echo "Stopping traffic generator..."
    
    # Kill iperf processes
    ip netns exec $SERVER_NS pkill -f iperf3 || true
    ip netns exec $CLIENT_NS pkill -f iperf3 || true
    
    echo "Traffic generator stopped"
}

# Function to monitor link statistics
monitor_stats() {
    local duration=$1
    local interval=${2:-1}
    local output_file=${3:-"bottleneck_stats.csv"}
    
    echo "Starting statistics monitoring (every ${interval}s for ${duration}s)..."
    
    # Create CSV header
    echo "timestamp,bottleneck_bw,cake_bw,latency_ms,client_backlog,radio_backlog,client_drops,radio_drops,throughput_mbps" > $output_file
    
    # Monitor for the specified duration
    local end_time=$(($(date +%s) + $duration))
    
    while [ $(date +%s) -lt $end_time ]; do
        # Get current bottleneck bandwidth
        local bottleneck_info=$(ip netns exec $RADIO_NS tc qdisc show dev veth2)
        local bottleneck_bw=$(echo "$bottleneck_info" | grep -oP 'rate \K[0-9]+[kmg]?bit')
        
        # Get current CAKE bandwidth
        local cake_info=$(ip netns exec $CLIENT_NS tc qdisc show dev veth0)
        local cake_bw=$(echo "$cake_info" | grep -oP 'bandwidth \K[0-9]+[kmg]?bit')
        
        # Measure RTT
        local rtt=$(ip netns exec $CLIENT_NS ping -c 1 -q 192.168.200.2 | grep "time=" | awk -F'time=' '{print $2}' | awk -F' ' '{print $1}')
        
        # Get queue statistics for client
        local client_queue_stats=$(ip netns exec $CLIENT_NS tc -s qdisc show dev veth0)
        local client_backlog=$(echo "$client_queue_stats" | grep -oP 'backlog \K[0-9]+[kmg]?b' | sed 's/[^0-9]//g')
        local client_drops=$(echo "$client_queue_stats" | grep -oP 'dropped \K[0-9]+')
        
        # Get queue statistics for radio
        local radio_queue_stats=$(ip netns exec $RADIO_NS tc -s qdisc show dev veth2)
        local radio_backlog=$(echo "$radio_queue_stats" | grep -oP 'backlog \K[0-9]+[kmg]?b' | sed 's/[^0-9]//g')
        local radio_drops=$(echo "$radio_queue_stats" | grep -oP 'dropped \K[0-9]+')
        
        # Measure throughput (using quick iperf3 measurement)
        local throughput=$(ip netns exec $CLIENT_NS iperf3 -c 192.168.200.2 -t 1 -J | grep "bits_per_second" | head -1 | awk -F': ' '{print $2}' | awk -F',' '{print $1}' | awk '{print $1/1000000}')
        
        # Handle missing values
        [ -z "$client_backlog" ] && client_backlog=0
        [ -z "$client_drops" ] && client_drops=0
        [ -z "$radio_backlog" ] && radio_backlog=0
        [ -z "$radio_drops" ] && radio_drops=0
        [ -z "$throughput" ] && throughput=0
        [ -z "$rtt" ] && rtt=0
        
        # Record statistics
        echo "$(date +%s),$bottleneck_bw,$cake_bw,$rtt,$client_backlog,$radio_backlog,$client_drops,$radio_drops,$throughput" >> $output_file
        
        # Wait for the next interval
        sleep $interval
    done
    
    echo "Statistics monitoring completed"
}

# Scenario 1: Stable bottleneck with increasing traffic
scenario_stable_bottleneck() {
    echo "Running scenario: Stable bottleneck with increasing traffic"
    
    # Configure bottleneck
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 100mbit
    
    # Start monitoring
    monitor_stats $DURATION 1 "scenario1_stats.csv" &
    
    # # Start with moderate traffic
    # start_traffic $((DURATION / 4)) 50
    # sleep $((DURATION / 4))
    
    # # Increase to heavy traffic
    # stop_traffic
    # start_traffic $((DURATION / 4)) 90
    # sleep $((DURATION / 4))
    
    # # Increase to overload
    # stop_traffic
    # start_traffic $((DURATION / 2)) 120


    # Testing...

    # Start with moderate traffic
    start_traffic $((DURATION / 4)) 120
    sleep $((DURATION / 4))
    
    # Wait for completion
    wait
    
    echo "Scenario 1 completed"
}

# Scenario 2: Fluctuating bottleneck capacity
scenario_fluctuating_bottleneck() {
    echo "Running scenario: Fluctuating bottleneck capacity"
    
    # Configure initial bottleneck
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 100mbit
    
    # Start monitoring
    monitor_stats $DURATION 1 "scenario2_stats.csv" &
    
    # Start steady traffic
    start_traffic $DURATION 80
    
    # Fluctuate bottleneck capacity
    local end_time=$(($(date +%s) + $DURATION))
    
    while [ $(date +%s) -lt $end_time ]; do
        # Random bandwidth between 50-120 Mbps
        local bandwidth=$((50 + RANDOM % 70))
        sudo ./simple_bottleneck_emulator.sh update_bottleneck ${bandwidth}mbit
        sleep $UPDATE_INTERVAL
    done
    
    # Wait for completion
    wait
    
    echo "Scenario 2 completed"
}

# Scenario 3: Gradual capacity degradation
scenario_gradual_degradation() {
    echo "Running scenario: Gradual capacity degradation"
    
    # Configure initial bottleneck
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 100mbit
    
    # Start monitoring
    monitor_stats $DURATION 1 "scenario3_stats.csv" &
    
    # Start moderate traffic
    start_traffic $DURATION 80
    
    # Gradually reduce bandwidth
    local end_time=$(($(date +%s) + $DURATION))
    local bandwidth=100
    
    while [ $(date +%s) -lt $end_time ] && [ $bandwidth -gt 30 ]; do
        sudo ./simple_bottleneck_emulator.sh update_bottleneck ${bandwidth}mbit
        bandwidth=$((bandwidth - 5))
        sleep $UPDATE_INTERVAL
    done
    
    # Wait for completion
    wait
    
    echo "Scenario 3 completed"
}

# Scenario 4: Bottleneck capacity steps
scenario_capacity_steps() {
    echo "Running scenario: Bottleneck capacity steps"
    
    # Configure initial bottleneck
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 100mbit
    
    # Start monitoring
    monitor_stats $DURATION 1 "scenario4_stats.csv" &
    
    # Start moderate traffic
    start_traffic $DURATION 90
    
    # Step through different capacities
    sleep $((DURATION / 6))
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 75mbit
    
    sleep $((DURATION / 6))
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 50mbit
    
    sleep $((DURATION / 6))
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 75mbit
    
    sleep $((DURATION / 6))
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 100mbit
    
    sleep $((DURATION / 6))
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 125mbit
    
    # Wait for completion
    wait
    
    echo "Scenario 4 completed"
}

# Scenario 5: Rapid capacity changes
scenario_rapid_changes() {
    echo "Running scenario: Rapid capacity changes"
    
    # Configure initial bottleneck
    sudo ./simple_bottleneck_emulator.sh update_bottleneck 100mbit
    
    # Start monitoring
    monitor_stats $DURATION 1 "scenario5_stats.csv" &
    
    # Start moderate traffic
    start_traffic $DURATION 90
    
    # Create rapid capacity changes
    local end_time=$(($(date +%s) + $DURATION))
    local short_interval=5  # 5 seconds between changes
    
    while [ $(date +%s) -lt $end_time ]; do
        # Random bandwidth between 50-120 Mbps
        local bandwidth=$((50 + RANDOM % 70))
        sudo ./simple_bottleneck_emulator.sh update_bottleneck ${bandwidth}mbit
        sleep $short_interval
    done
    
    # Wait for completion
    wait
    
    echo "Scenario 5 completed"
}

# Main function
main() {
    check_environment
    
    case "$1" in
        scenario1)
            scenario_stable_bottleneck
            ;;
        scenario2)
            scenario_fluctuating_bottleneck
            ;;
        scenario3)
            scenario_gradual_degradation
            ;;
        scenario4)
            scenario_capacity_steps
            ;;
        scenario5)
            scenario_rapid_changes
            ;;
        all)
            scenario_stable_bottleneck
            sleep 5
            scenario_fluctuating_bottleneck
            sleep 5
            scenario_gradual_degradation
            sleep 5
            scenario_capacity_steps
            sleep 5
            scenario_rapid_changes
            ;;
        *)
            echo "Usage: $0 {scenario1|scenario2|scenario3|scenario4|scenario5|all}"
            echo ""
            echo "Available scenarios:"
            echo "  scenario1: Stable bottleneck with increasing traffic"
            echo "  scenario2: Fluctuating bottleneck capacity"
            echo "  scenario3: Gradual capacity degradation"
            echo "  scenario4: Bottleneck capacity steps"
            echo "  scenario5: Rapid capacity changes"
            echo "  all: Run all scenarios sequentially"
            exit 1
            ;;
    esac
    
    # Clean up at the end
    stop_traffic
    
    echo "Test completed successfully"
}

main "$@"


