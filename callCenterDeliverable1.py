import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# System Parameters
RANDOM_SEED = 42
SIM_TIME = 480  # simulation time in minutes (8 hours)
CALL_ARRIVAL_INTERVAL = (1, 6)  # customers arrive randomly between 1–6 minutes
CALL_SERVICE_TIME = (3, 8)      # service duration in minutes

random.seed(RANDOM_SEED)

# Customer Process
def customer(env, name, call_center, wait_times, service_times):
    arrival_time = env.now
    # Request an available agent
    with call_center.request() as request:
        yield request
        wait = env.now - arrival_time
        wait_times.append(wait)
        service_duration = random.uniform(*CALL_SERVICE_TIME)
        service_times.append(service_duration)
        yield env.timeout(service_duration)

# Call Arrival Generator
def setup(env, num_agents, wait_times, service_times):
    call_center = simpy.Resource(env, num_agents)
    i = 0
    while True:
        yield env.timeout(random.uniform(*CALL_ARRIVAL_INTERVAL))
        i += 1
        env.process(customer(env, f'Customer {i}', call_center, wait_times, service_times))

# Simulation Function
def run_simulation(num_agents, sim_time=SIM_TIME, reps=30):
    all_avg_waits, all_util, all_throughput, all_queue = [], [], [], []
    detailed_results = []  # Store individual replication results
    
    for rep in range(reps):
        env = simpy.Environment()
        wait_times, service_times = [], []
        env.process(setup(env, num_agents, wait_times, service_times))
        env.run(until=sim_time)
        
        avg_wait = np.mean(wait_times)
        utilization = np.sum(service_times) / (num_agents * sim_time)
        throughput = len(service_times) / (sim_time / 60)  # calls per hour
        max_queue = np.percentile(wait_times, 95)
        
        all_avg_waits.append(avg_wait)
        all_util.append(utilization)
        all_throughput.append(throughput)
        all_queue.append(max_queue)
        
        # Store detailed results for each replication
        detailed_results.append({
            'num_agents': num_agents,
            'replication': rep + 1,
            'total_customers': len(service_times),
            'avg_wait_min': avg_wait,
            'utilization': utilization,
            'throughput_per_hr': throughput,
            'queue_95th_percentile': max_queue
        })
    
    return detailed_results

# Experiments Conducted
all_detailed_results = []

for agents in [2, 3, 5]:  # test with 2, 3, and 5 agents
    print(f"\nRunning simulation with {agents} agents...")
    detailed = run_simulation(agents)
    all_detailed_results.extend(detailed)

# Create DataFrame
df_detailed = pd.DataFrame(all_detailed_results)

print("\n" + "="*80)
print("DETAILED RESULTS (All Replications)")
print("="*80)
print(df_detailed)

# Save to CSV
df_detailed.to_csv('call_center_detailed.csv', index=False)

print("\n" + "="*80)
print("FILE SAVED:")
print("="*80)
print(f"Detailed results: {os.path.abspath('call_center_detailed.csv')}")
print(f"Total rows: {len(df_detailed)} (30 replications × 3 agent configurations)")

# Visualizations
# Calculate means for plotting
df_summary = df_detailed.groupby('num_agents').agg({
    'avg_wait_min': 'mean',
    'utilization': 'mean',
    'throughput_per_hr': 'mean',
    'queue_95th_percentile': 'mean'
}).reset_index()

plt.figure(figsize=(10, 6))
plt.plot(df_summary["num_agents"], df_summary["avg_wait_min"], marker='o', label="Avg Wait Time (min)")
plt.xlabel("Number of Agents")
plt.ylabel("Average Wait Time (minutes)")
plt.title("Average Customer Wait Time vs. Number of Agents")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_summary["num_agents"], df_summary["utilization"], marker='o', color='orange', label="Agent Utilization")
plt.xlabel("Number of Agents")
plt.ylabel("Utilization (fraction of busy time)")
plt.title("Agent Utilization vs. Number of Agents")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_summary["num_agents"], df_summary["throughput_per_hr"], marker='o', color='green', label="Throughput (calls/hr)")
plt.xlabel("Number of Agents")
plt.ylabel("Throughput (calls per hour)")
plt.title("Throughput vs. Number of Agents")
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(df_summary["num_agents"], df_summary["queue_95th_percentile"], marker='o', color='purple', label="95th Percentile Queue Wait")
plt.xlabel("Number of Agents")
plt.ylabel("95th Percentile Queue Wait (minutes)")
plt.title("Queue Length vs. Number of Agents")
plt.grid(True)
plt.legend()
plt.show()