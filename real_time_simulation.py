import time
import random
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from collections import deque
import csv
from datetime import datetime


# Generate initial training data (only normal traffic)
training_data = np.array([[random.uniform(50, 200), random.uniform(200, 500)] for _ in range(500)])

# Train Isolation Forest
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(training_data)
print("Anomaly detection model trained!")

# Initialize log file
log_file = "anomalies_log.csv"
with open(log_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Latency (ms)", "Size (bytes)", "Status"])  # Write header
print(f"Logging anomalies to {log_file}...")

# Real-time data storage
traffic_data = deque(maxlen=100)  # Store the last 100 traffic points
anomaly_data = deque(maxlen=100)  # Store the anomaly flags (1 for anomaly, 0 for normal)

# Function to update the plot
def update_plot():
    plt.cla()  # Clear the plot
    plt.title("Real-Time Traffic Analysis")
    plt.xlabel("Traffic Instance")
    plt.ylabel("Feature Value")
    
    # Plot traffic data
    traffic_indices = range(len(traffic_data))
    plt.plot(traffic_indices, [d[0] for d in traffic_data], label="Latency (ms)", color="blue")
    plt.plot(traffic_indices, [d[1] for d in traffic_data], label="Size (bytes)", color="green")
    
    # Highlight anomalies
    anomaly_indices = [i for i, flag in enumerate(anomaly_data) if flag == 1]
    anomaly_latencies = [traffic_data[i][0] for i in anomaly_indices]
    anomaly_sizes = [traffic_data[i][1] for i in anomaly_indices]
    plt.scatter(anomaly_indices, anomaly_latencies, label="Anomalous Latency", color="red", marker="x")
    plt.scatter(anomaly_indices, anomaly_sizes, label="Anomalous Size", color="orange", marker="x")
    
    plt.legend()
    plt.pause(0.1)  # Pause for a short moment to update the plot

# Define the traffic generation function
def generate_traffic():
    while True:
        # Simulate normal traffic with random latency and size
        latency = random.uniform(50, 200)  # Random latency between 50 and 200 ms
        size = random.uniform(200, 500)    # Random size between 200 and 500 bytes
        
        # Yield the generated traffic data as a tuple (latency, size)
        yield (latency, size)

        # Simulate a short delay between traffic generation
        time.sleep(1)  # Adjust sleep time as needed for real-time simulation

# Run for 1 minute (60 seconds)
if __name__ == "__main__":
    print("Generating and analyzing real-time traffic...")
    duration_in_seconds = 60  # Run for 1 minute
    start_time = time.time()
    
    for traffic in generate_traffic():
        # Check if the specified time has passed
        if time.time() - start_time > duration_in_seconds:
            print("Simulation complete.")
            break
        
        # Extract features for prediction
        features = np.array(traffic[:2]).reshape(1, -2)
        prediction = model.predict(features)  # -1 = anomaly, 1 = normal
        
        # Determine status
        status = "Anomalous" if prediction == -1 else "Normal"
        
        # Store the traffic and anomaly data
        traffic_data.append(traffic[:2])
        anomaly_data.append(1 if prediction == -1 else 0)
        
        # Log anomaly if detected
        if prediction == -1:
            with open(log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    traffic[0],  # Latency
                    traffic[1],  # Size
                    "Anomalous"
                ])
        
        # Display real-time results
        print(f"Traffic: Latency={traffic[0]:.2f} ms, Size={traffic[1]:.2f} bytes, Status={status}")
        
        # Update the plot
        update_plot()
