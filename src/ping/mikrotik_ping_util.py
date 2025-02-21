import paramiko
import subprocess
import time
import csv
import datetime

def run_ping_test(duration_minutes=5):
    end_time = time.time() + duration_minutes * 60
    ping_results = []
    while time.time() < end_time:
        # Run the ping command
        result = subprocess.run(['ping', '-c', '1', '10.0.10.1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if result.returncode == 0:
            output = result.stdout.decode()
            ping_time = output.split('time=')[-1].split(' ms')[0]
            ping_results.append([timestamp, ping_time])
        else:
            ping_results.append([timestamp, None])
        time.sleep(0.1)

    # Save results to CSV
    with open('ping_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Ping Time (ms)'])
        writer.writerows(ping_results)

def log_handover(router_ip, route_comment, handover_type):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('handover_log.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, router_ip, route_comment, handover_type])

def adjust_routing_costs(router_ip, username, password, route_comment, low_cost_distance, high_cost_distance, handover_type):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(router_ip, username=username, password=password)

    # Find the routes by comment and adjust the distances
    commands = [
        f"/ip route set [find comment={route_comment}_primary] distance={low_cost_distance}",
        f"/ip route set [find comment={route_comment}_secondary] distance={high_cost_distance}"
    ]

    for command in commands:
        stdin, stdout, stderr = ssh.exec_command(command)
        stdout.channel.recv_exit_status()

    ssh.close()

    # Log the handover
    log_handover(router_ip, route_comment, handover_type)

def main():
    router1_ip = '10.0.10.1'
    router2_ip = '10.0.20.1'
    username = 'admin'  # Replace with actual username
    password = '2Turtles#'  # Replace with actual password
    route1_comment = 'to_10.0.20.0'
    route2_comment = 'to_10.0.10.0'
    
    print("\nStarting...\n")
    run_ping_test()

    start_time = time.time()
    #while time.time() - start_time < 10 * 60:  # Run for 10 minutes
        #run_ping_test()
        #adjust_routing_costs(router1_ip, username, password, route1_comment, 100, 10, "Primary to Secondary")
        #adjust_routing_costs(router2_ip, username, password, route2_comment, 100, 10, "Primary to Secondary")
        #time.sleep(60)
        #adjust_routing_costs(router1_ip, username, password, route1_comment, 10, 100, "Secondary to Primary")
        #adjust_routing_costs(router2_ip, username, password, route2_comment, 10, 100, "Secondary to Primary")
        #print("Sleeping...\n")
        #time.sleep(60)
        #print("Running time is: {0}".format(time.time() - start_time * 60))

if __name__ == "__main__":
    main()
