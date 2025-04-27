#!/usr/bin/env python3
import socket
import time
import struct
import argparse
from datetime import datetime
import statistics
import json
from typing import Dict, List

class SplitPing:
    def __init__(self, mode='server', host='0.0.0.0', port=6924, peer=None):
        self.mode = mode
        self.host = host
        self.port = port
        self.peer = peer
        
        # Store measurements locally
        self.measurements = []
        self.sequence = 0
        
        # UDP socket for pings
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Set timeout for receives
        self.sock.settimeout(0.1)
        
        # Print initialization info
        print(f"Initialized {mode} mode")
        print(f"Local address: {host}")
        print(f"Port: {port}")
        if peer:
            print(f"Peer address: {peer}")
        
    def start_server(self):
        """Run in server mode"""
        try:
            print(f"Starting server on {self.host}:{self.port}")
            self.sock.bind((self.host, self.port))
            print("Server bound successfully")
            
            while True:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    self._handle_packet(data, addr)
                except socket.timeout:
                    continue
                except Exception as e:
                    print(f"Error receiving packet: {e}")
                    
        except KeyboardInterrupt:
            print("\nShutting down server...")
            self.save_measurements()
        except Exception as e:
            print(f"Server error: {e}")
            self.save_measurements()
    
    def start_client(self):
        """Run in client mode"""
        if not self.peer:
            raise ValueError("Peer address required for client mode")
            
        try:
            # Bind to receive responses
            self.sock.bind((self.host, 0))
            bound_port = self.sock.getsockname()[1]
            print(f"Client bound to port {bound_port}")
            print(f"Starting client, sending to {self.peer}:{self.port}")
            
            # Send packets
            for i in range(1000):  # Send 1000 packets
                if i % 100 == 0:
                    print(f"Sending packet {i}/1000")
                
                self._send_packet()
                
                # Try to receive any pending responses
                try:
                    while True:
                        data, addr = self.sock.recvfrom(1024)
                        self._handle_packet(data, addr)
                except socket.timeout:
                    pass
                    
                time.sleep(0.1)  # 100ms between packets
                
            # Allow time for final responses
            print("Waiting for final responses...")
            end_time = time.time() + 1
            while time.time() < end_time:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    self._handle_packet(data, addr)
                except socket.timeout:
                    continue
                    
            self.save_measurements()
            
        except KeyboardInterrupt:
            print("\nShutting down client...")
            self.save_measurements()
        except Exception as e:
            print(f"Client error: {e}")
            self.save_measurements()
    
    def _send_packet(self):
        """Send a UDP packet with sequence number and timestamp"""
        try:
            send_time = time.time()
            
            # Pack sequence number and send timestamp
            packet = struct.pack('!Qd', self.sequence, send_time)
            bytes_sent = self.sock.sendto(packet, (self.peer, self.port))
            
            if self.sequence % 100 == 0:
                print(f"Sent packet {self.sequence} ({bytes_sent} bytes)")
            
            self.sequence += 1
            
        except Exception as e:
            print(f"Error sending packet: {e}")
    
    def _handle_packet(self, data: bytes, addr: tuple):
        """Handle received UDP packet"""
        try:
            recv_time = time.time()
            
            if len(data) == struct.calcsize('!Qd'):  # Initial packet
                # Unpack sequence and original send time
                sequence, send_time = struct.unpack('!Qd', data)
                
                # For server: send response with original timestamp
                if self.mode == 'server':
                    if sequence % 100 == 0:
                        print(f"Server received packet {sequence} from {addr}")
                    response = struct.pack('!Qdd', sequence, send_time, recv_time)
                    self.sock.sendto(response, addr)
            
            elif len(data) == struct.calcsize('!Qdd'):  # Response packet
                # Unpack sequence, original send time, and server recv time
                sequence, send_time, server_time = struct.unpack('!Qdd', data)
                
                # Calculate forward and reverse delays
                forward_delay = (server_time - send_time) * 1000  # ms
                reverse_delay = (recv_time - server_time) * 1000  # ms
                
                measurement = {
                    'sequence': sequence,
                    'send_time': send_time,
                    'server_time': server_time,
                    'recv_time': recv_time,
                    'forward_delay': forward_delay,
                    'reverse_delay': reverse_delay,
                    'total_delay': forward_delay + reverse_delay
                }
                
                self.measurements.append(measurement)
                
                # Print progress every 100 packets
                if sequence % 100 == 0:
                    print(f"\nReceived response for packet {sequence}")
                    self._print_statistics()
            
        except Exception as e:
            print(f"Error handling packet: {e}")
    
    def _print_statistics(self):
        """Print current statistics"""
        if not self.measurements:
            return
            
        forward_delays = [m['forward_delay'] for m in self.measurements]
        reverse_delays = [m['reverse_delay'] for m in self.measurements]
        total_delays = [m['total_delay'] for m in self.measurements]
        
        print(f"\nPacket count: {len(self.measurements)}")
        print(f"Forward delay (ms): min={min(forward_delays):.3f}, "
              f"avg={statistics.mean(forward_delays):.3f}, "
              f"max={max(forward_delays):.3f}")
        print(f"Reverse delay (ms): min={min(reverse_delays):.3f}, "
              f"avg={statistics.mean(reverse_delays):.3f}, "
              f"max={max(reverse_delays):.3f}")
        print(f"Total RTT (ms): min={min(total_delays):.3f}, "
              f"avg={statistics.mean(total_delays):.3f}, "
              f"max={max(total_delays):.3f}")
    
    def save_measurements(self):
        """Save measurements to file"""
        if not self.measurements:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"split_ping_{self.mode}_{timestamp}.json"
        
        # Calculate statistics
        forward_delays = [m['forward_delay'] for m in self.measurements]
        reverse_delays = [m['reverse_delay'] for m in self.measurements]
        total_delays = [m['total_delay'] for m in self.measurements]
        
        output = {
            'measurements': self.measurements,
            'statistics': {
                'packet_count': len(self.measurements),
                'forward_delay': {
                    'min': min(forward_delays),
                    'max': max(forward_delays),
                    'mean': statistics.mean(forward_delays),
                    'median': statistics.median(forward_delays),
                    'stdev': statistics.stdev(forward_delays)
                },
                'reverse_delay': {
                    'min': min(reverse_delays),
                    'max': max(reverse_delays),
                    'mean': statistics.mean(reverse_delays),
                    'median': statistics.median(reverse_delays),
                    'stdev': statistics.stdev(reverse_delays)
                },
                'total_delay': {
                    'min': min(total_delays),
                    'max': max(total_delays),
                    'mean': statistics.mean(total_delays),
                    'median': statistics.median(total_delays),
                    'stdev': statistics.stdev(total_delays)
                }
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"\nSaved measurements to {filename}")

def main():
    parser = argparse.ArgumentParser(description='Split Ping with Self-Contained UDP')
    parser.add_argument('mode', choices=['server', 'client'],
                       help='Operating mode')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (server) or local address (client)')
    parser.add_argument('--port', type=int, default=6924,
                       help='Port number')
    parser.add_argument('--peer', help='Peer address (required for client mode)')
    
    args = parser.parse_args()
    
    # For client mode, require peer address
    if args.mode == 'client' and not args.peer:
        parser.error("--peer argument is required for client mode")
    
    split_ping = SplitPing(
        mode=args.mode,
        host=args.host,
        port=args.port,
        peer=args.peer
    )
    
    if args.mode == 'server':
        split_ping.start_server()
    else:
        split_ping.start_client()

if __name__ == '__main__':
    main()
