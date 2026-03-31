import zmq
import struct
import numpy as np
import time

def dummy_quantum_kmeans(samples, centers, k):
    """
    Dummy placeholder for your Qiskit / Quantum K-Means logic!
    For now, it just calculates standard euclidean distances to return valid centers.
    """
    # 1. Assignment
    distances = np.linalg.norm(samples[:, np.newaxis] - centers, axis=2)
    labels = np.argmin(distances, axis=1)
    
    # 2. Update
    new_centers = np.zeros_like(centers)
    for i in range(k):
        cluster_points = samples[labels == i]
        if len(cluster_points) > 0:
            new_centers[i] = cluster_points.mean(axis=0)
        else:
            # Fallback to random point if cluster dies
            new_centers[i] = samples[np.random.choice(len(samples))]
            
    return new_centers

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://127.0.0.1:5555")
    
    print("Quantum Engine IPC Server is running on tcp://127.0.0.1:5555...")
    print("Waiting for C++ Coreset Data...")

    while True:
        try:
            # Receive request from C++
            message = socket.recv()
            
            # Unpack dimensions
            n, k = struct.unpack('ii', message[:8])
            
            # Unpack payload buffers using native numpy (insanely fast)
            samples_bytes = n * 5 * 4
            samples = np.frombuffer(message[8 : 8 + samples_bytes], dtype=np.float32).reshape((n, 5))
            centers = np.frombuffer(message[8 + samples_bytes :], dtype=np.float32).reshape((k, 5))
            
            # RUN YOUR QISKIT ALGORITHM HERE!
            # Make sure it returns a (k, 5) float32 numpy array.
            start_time = time.time()
            new_centers = dummy_quantum_kmeans(samples, centers, k)
            
            # Simulate a 50ms minimum "quantum simulator" delay just so
            # you can visibly see that your UI continues running exactly at 30 FPS!
            elapsed = time.time() - start_time
            if elapsed < 0.05:
                time.sleep(0.05 - elapsed)
                
            # Send new centers back to C++
            reply_bytes = new_centers.astype(np.float32).tobytes()
            socket.send(reply_bytes)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            # On crash, send an empty response so C++ socket times out safely
            socket.send(b"ERR")

if __name__ == "__main__":
    main()
