#!/usr/bin/env python3
import socket
import time
import struct
import argparse
from datetime import datetime
import statistics
import json

class LatencyMeasurement:
    def __init__(self, mode='sender', host='0.0.0.0', port=5000, target=None):
        self.mode = mode
        self.host = host
        self.port = port
        self.target = target
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # For storing measurements
        self.measurements = []
        
    def start_sender(self, num_packets=1000, interval=0.1):
        """Send UDP packets with timestamps"""
        print(f"Sending {num_packets} packets to {self.target}:{self.port}")
        
        sequence = 0
        while sequence < num_packets:
            # Get precise timestamp
            send_time = time.time()
            
            # Create packet with sequence number and timestamp
            # Store timestamp as nanoseconds for higher precision
            packet = struct.pack('!Qd', sequence, send_time)  # Use double precision float for timestamp
            
            try:
                self.sock.sendto(packet, (self.target, self.port))
                sequence += 1
                time.sleep(interval)  # Wait before sending next packet
            except Exception as e:
                print(f"Error sending packet: {e}")
                
        print("Finished sending packets")

    def start_receiver(self):
        """Receive packets and calculate latency"""
        self.sock.bind((self.host, self.port))
        print(f"Listening on {self.host}:{self.port}")
        
        while True:
            try:
                data, addr = self.sock.recvfrom(1024)
                recv_time = time.time()
                
                # Unpack sequence number and send timestamp
                sequence, send_time = struct.unpack('!Qd', data)  # Unpack double precision float
                
                # Calculate latency
                # Both timestamps are in seconds, convert difference to milliseconds
                latency = (recv_time - send_time) * 1000.0  # Convert to milliseconds
                
                # Sanity check - if the latency is unreasonably large, it might indicate
                # a time synchronization issue or overflow
                if latency > 1000.0:  # More than 1 second is suspicious for LAN
                    print(f"Warning: Large latency detected: {latency:.3f} ms")
                    print(f"Send time: {send_time:.6f}, Receive time: {recv_time:.6f}")
                
                measurement = {
                    'sequence': sequence,
                    'send_time': send_time,
                    'recv_time': recv_time,
                    'latency': latency
                }
                
                self.measurements.append(measurement)
                print(f"Packet {sequence}: Latency = {latency:.3f} ms")
                
                # Save measurements periodically
                if len(self.measurements) % 100 == 0:
                    self.save_measurements()
                    
            except KeyboardInterrupt:
                print("\nStopping receiver...")
                self.save_measurements()
                break
            except Exception as e:
                print(f"Error receiving packet: {e}")
                
    def save_measurements(self):
        """Save measurements to file"""
        if not self.measurements:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"latency_measurements_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump({
                'measurements': self.measurements,
                'statistics': self.calculate_statistics()
            }, f, indent=2)
            
        print(f"Saved measurements to {filename}")
        
    def calculate_statistics(self):
        """Calculate basic statistics from measurements"""
        if not self.measurements:
            return {}
            
        latencies = [m['latency'] for m in self.measurements]
        
        return {
            'min': min(latencies),
            'max': max(latencies),
            'mean': statistics.mean(latencies),
            'median': statistics.median(latencies),
            'stdev': statistics.stdev(latencies) if len(latencies) > 1 else 0,
            'packet_count': len(latencies)
        }

def main():
    parser = argparse.ArgumentParser(description='Unidirectional Latency Measurement')
    parser.add_argument('mode', choices=['sender', 'receiver'],
                       help='Operating mode')
    parser.add_argument('--host', default='0.0.0.0',
                       help='Host to bind to (receiver) or target host (sender)')
    parser.add_argument('--port', type=int, default=5000,
                       help='UDP port number')
    parser.add_argument('--packets', type=int, default=1000,
                       help='Number of packets to send (sender only)')
    parser.add_argument('--interval', type=float, default=0.1,
                       help='Interval between packets in seconds (sender only)')
    
    args = parser.parse_args()
    
    measurement = LatencyMeasurement(
        mode=args.mode,
        host=args.host,
        port=args.port,
        target=args.host if args.mode == 'sender' else None
    )
    
    if args.mode == 'sender':
        measurement.start_sender(args.packets, args.interval)
    else:
        measurement.start_receiver()

if __name__ == '__main__':
    main()
