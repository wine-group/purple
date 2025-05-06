#!/bin/bash
# Simple Bottleneck Network Emulator
# 
# Creates a three-node topology: Client -> Radio -> Server
# Where the Radio->Server link acts as the bottleneck
#
# Usage: ./simple_bottleneck_emulator.sh setup|clean|status|update_bottleneck <rate>
#

set -e

CLIENT_NS="client_ns"
RADIO_NS="radio_ns"
SERVER_NS="server_ns"
CLIENT_VETH="veth0"
RADIO_VETH_CLIENT="veth1"
RADIO_VETH_SERVER="veth2"
SERVER_VETH="veth3"
CLIENT_IP="192.168.100.2/24"
RADIO_IP_CLIENT="192.168.100.1/24"
RADIO_IP_SERVER="192.168.200.1/24"
SERVER_IP="192.168.200.2/24"

# Default link parameters
DEFAULT_BOTTLENECK_BW="100mbit"
DEFAULT_BOTTLENECK_LATENCY="5ms"
DEFAULT_BOTTLENECK_QUEUE="1000"  # Queue length in packets

# Function to check if we're running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "This script must be run as root"
        exit 1
    fi
}

# Function to create network namespaces and connections
setup_namespaces() {
    echo "Setting up network namespaces..."
    
    # Create namespaces if they don't exist
    ip netns list | grep -q $CLIENT_NS || ip netns add $CLIENT_NS
    ip netns list | grep -q $RADIO_NS || ip netns add $RADIO_NS
    ip netns list | grep -q $SERVER_NS || ip netns add $SERVER_NS
    
    # Create virtual ethernet pairs
    ip link add $CLIENT_VETH type veth peer name $RADIO_VETH_CLIENT
    ip link add $RADIO_VETH_SERVER type veth peer name $SERVER_VETH
    
    # Move interfaces to namespaces
    ip link set $CLIENT_VETH netns $CLIENT_NS
    ip link set $RADIO_VETH_CLIENT netns $RADIO_NS
    ip link set $RADIO_VETH_SERVER netns $RADIO_NS
    ip link set $SERVER_VETH netns $SERVER_NS
    
    # Configure IP addresses
    ip netns exec $CLIENT_NS ip addr add $CLIENT_IP dev $CLIENT_VETH
    ip netns exec $RADIO_NS ip addr add $RADIO_IP_CLIENT dev $RADIO_VETH_CLIENT
    ip netns exec $RADIO_NS ip addr add $RADIO_IP_SERVER dev $RADIO_VETH_SERVER
    ip netns exec $SERVER_NS ip addr add $SERVER_IP dev $SERVER_VETH
    
    # Bring up interfaces
    ip netns exec $CLIENT_NS ip link set $CLIENT_VETH up
    ip netns exec $CLIENT_NS ip link set lo up
    ip netns exec $RADIO_NS ip link set $RADIO_VETH_CLIENT up
    ip netns exec $RADIO_NS ip link set $RADIO_VETH_SERVER up
    ip netns exec $RADIO_NS ip link set lo up
    ip netns exec $SERVER_NS ip link set $SERVER_VETH up
    ip netns exec $SERVER_NS ip link set lo up
    
    # Set up routing
    ip netns exec $CLIENT_NS ip route add default via ${RADIO_IP_CLIENT%/*}
    ip netns exec $SERVER_NS ip route add default via ${RADIO_IP_SERVER%/*}
    
    # Enable IP forwarding in the radio namespace
    ip netns exec $RADIO_NS sysctl -w net.ipv4.ip_forward=1
    
    echo "Network namespaces and links created successfully"
}

# Function to set up bottleneck on Radio->Server link
setup_bottleneck() {
    echo "Setting up bottleneck link (Radio->Server)..."
    
    # Remove any existing qdisc
    ip netns exec $RADIO_NS tc qdisc del dev $RADIO_VETH_SERVER root 2>/dev/null || true
    
    # Set up netem qdisc with bandwidth limit, latency, and queue
    ip netns exec $RADIO_NS tc qdisc add dev $RADIO_VETH_SERVER root handle 1: netem \
        rate $DEFAULT_BOTTLENECK_BW \
        delay $DEFAULT_BOTTLENECK_LATENCY \
        limit $DEFAULT_BOTTLENECK_QUEUE
    
    echo "Bottleneck configured with: Rate=$DEFAULT_BOTTLENECK_BW, Latency=$DEFAULT_BOTTLENECK_LATENCY, Queue=$DEFAULT_BOTTLENECK_QUEUE"
}

# Function to set up CAKE on Client->Radio link
setup_cake() {
    echo "Setting up CAKE on Client->Radio link..."
    
    # Remove any existing qdisc
    ip netns exec $CLIENT_NS tc qdisc del dev $CLIENT_VETH root 2>/dev/null || true
    
    # Set up CAKE with initial bandwidth of 90% of bottleneck
    # Extract numeric part of DEFAULT_BOTTLENECK_BW for calculation
    BOTTLENECK_BW_VALUE=$(echo $DEFAULT_BOTTLENECK_BW | sed -e 's/[^0-9]//g')
    INITIAL_CAKE_BW=$((BOTTLENECK_BW_VALUE * 9 / 10))
    
    if [[ $DEFAULT_BOTTLENECK_BW == *"mbit"* ]]; then
        CAKE_BW="${INITIAL_CAKE_BW}mbit"
    elif [[ $DEFAULT_BOTTLENECK_BW == *"kbit"* ]]; then
        CAKE_BW="${INITIAL_CAKE_BW}kbit"
    else
        CAKE_BW="${INITIAL_CAKE_BW}bit"
    fi
    
    ip netns exec $CLIENT_NS tc qdisc add dev $CLIENT_VETH root handle 1: cake \
        bandwidth $CAKE_BW \
        rtt 50ms \
        overhead 20 \
        besteffort
    
    echo "CAKE configured with initial bandwidth: $CAKE_BW"
}

# Function to clean up the environment
cleanup() {
    echo "Cleaning up network emulation environment..."
    
    # Delete namespaces (this also removes all interfaces in them)
    ip netns list | grep -q $CLIENT_NS && ip netns del $CLIENT_NS
    ip netns list | grep -q $RADIO_NS && ip netns del $RADIO_NS
    ip netns list | grep -q $SERVER_NS && ip netns del $SERVER_NS
    
    echo "Cleanup complete"
}

# Function to show the status of the environment
show_status() {
    echo "====== Network Emulation Environment Status ======"
    
    # Check if namespaces exist
    echo "Network Namespaces:"
    ip netns list
    echo ""
    
    # Show interfaces in each namespace
    for ns in $CLIENT_NS $RADIO_NS $SERVER_NS; do
        if ip netns list | grep -q $ns; then
            echo "Interfaces in $ns namespace:"
            ip netns exec $ns ip addr show
            echo ""
            
            # Show routing table
            echo "Routes in $ns namespace:"
            ip netns exec $ns ip route
            echo ""
            
            # Show TC configuration
            echo "Traffic Control configuration in $ns namespace:"
            ip netns exec $ns tc qdisc show
            ip netns exec $ns tc class show 2>/dev/null || true
            echo ""
        fi
    done
    
    # Show current bottleneck settings
    if ip netns list | grep -q $RADIO_NS; then
        echo "Bottleneck settings (Radio->Server link):"
        ip netns exec $RADIO_NS tc qdisc show dev $RADIO_VETH_SERVER
        echo ""
        
        echo "CAKE settings (Client->Radio link):"
        ip netns exec $CLIENT_NS tc qdisc show dev $CLIENT_VETH
        echo ""
    fi
}

# Function to update bottleneck bandwidth
update_bottleneck() {
    local new_rate=$1
    
    if [ -z "$new_rate" ]; then
        echo "Usage: $0 update_bottleneck <rate>"
        echo "Example: $0 update_bottleneck 80mbit"
        return 1
    fi
    
    echo "Updating bottleneck bandwidth to $new_rate..."
    
    # Update netem on Radio->Server link
    ip netns exec $RADIO_NS tc qdisc change dev $RADIO_VETH_SERVER root handle 1: netem \
        rate $new_rate \
        delay $DEFAULT_BOTTLENECK_LATENCY \
        limit $DEFAULT_BOTTLENECK_QUEUE
    
    echo "Bottleneck bandwidth updated successfully"
}

# Function to update CAKE bandwidth on Client->Radio link
update_cake() {
    local new_rate=$1
    
    if [ -z "$new_rate" ]; then
        echo "Usage: $0 update_cake <rate>"
        echo "Example: $0 update_cake 75mbit"
        return 1
    fi
    
    echo "Updating CAKE bandwidth to $new_rate..."
    
    # Update CAKE on Client->Radio link
    ip netns exec $CLIENT_NS tc qdisc change dev $CLIENT_VETH root handle 1: cake \
        bandwidth $new_rate \
        rtt 50ms \
        overhead 20 \
        besteffort
    
    echo "CAKE bandwidth updated successfully"
}

# Main function
main() {
    check_root
    
    case "$1" in
        setup)
            setup_namespaces
            setup_bottleneck
            setup_cake
            echo "Setup complete. To test connectivity, run:"
            echo "  sudo ip netns exec $CLIENT_NS ping -c 3 ${SERVER_IP%/*}"
            ;;
        clean)
            cleanup
            ;;
        status)
            show_status
            ;;
        update_bottleneck)
            update_bottleneck "$2"
            ;;
        update_cake)
            update_cake "$2"
            ;;
        *)
            echo "Usage: $0 {setup|clean|status|update_bottleneck <rate>|update_cake <rate>}"
            echo "Examples:"
            echo "  $0 setup                        # Set up environment"
            echo "  $0 update_bottleneck 75mbit     # Change bottleneck rate"
            echo "  $0 update_cake 70mbit           # Change CAKE rate"
            echo "  $0 status                       # Show status"
            echo "  $0 clean                        # Clean up environment"
            exit 1
            ;;
    esac
}

main "$@"