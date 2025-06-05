"""Command line interface for gllm."""

import argparse
import sys
import subprocess
import signal
import os
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from gllm import data_def
from gllm.consts import Endpoints


# Global variables to store subprocesses
balancer_process = None
worker_processes = []


def signal_handler(signum, frame):
    """Handle termination signals to clean up subprocess."""
    global balancer_process, worker_processes
    print(f"\nReceived signal {signum}, shutting down...")
    
    # Terminate worker processes first
    for i, worker_process in enumerate(worker_processes):
        if worker_process and worker_process.poll() is None:
            print(f"Terminating worker {i+1}...")
            worker_process.terminate()
    
    # Handle worker shutdown in parallel
    def shutdown_worker(worker_info):
        i, worker_process = worker_info
        if worker_process and worker_process.poll() is None:
            try:
                worker_process.wait(timeout=5)
                print(f"Worker {i+1} shut down gracefully")
            except subprocess.TimeoutExpired:
                print(f"Force killing worker {i+1}...")
                worker_process.kill()
                worker_process.wait()
    
    # Wait for workers to shut down gracefully in parallel
    if worker_processes:
        with ThreadPoolExecutor(max_workers=len(worker_processes)) as executor:
            worker_info = [(i, worker_process) for i, worker_process in enumerate(worker_processes)]
            executor.map(shutdown_worker, worker_info)
    
    # Terminate balancer process
    if balancer_process and balancer_process.poll() is None:
        print("Terminating balancer process...")
        balancer_process.terminate()
        try:
            balancer_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            print("Force killing balancer process...")
            balancer_process.kill()
            balancer_process.wait()
    
    sys.exit(0)


def parse_gpu_specification(gpu_spec):
    """Parse a GPU specification into a list of GPU IDs.
    
    Args:
        gpu_spec (str): GPU specification (e.g., "3", "3:5", "3,5,2")
        
    Returns:
        list: List of GPU IDs
    """
    if ':' in gpu_spec:
        # Range specification (e.g., "3:5")
        start, end = gpu_spec.split(':')
        return list(range(int(start), int(end) + 1))
    elif ',' in gpu_spec:
        # Comma-separated list (e.g., "3,5,2")
        return [int(gpu.strip()) for gpu in gpu_spec.split(',')]
    else:
        # Single GPU (e.g., "3")
        return [int(gpu_spec)]


def parse_worker_definitions(worker_defs):
    """Parse worker definitions into GPU configurations.
    
    Args:
        worker_defs (list): List of worker definition strings
        
    Returns:
        list: List of GPU ID lists for each worker
    """
    workers = []
    for worker_def in worker_defs:
        gpu_ids = parse_gpu_specification(worker_def)
        workers.append(gpu_ids)
    return workers


def start_worker(gpu_ids, worker_port, vllm_port, worker_index):
    """Start a single worker with specified GPU configuration and ports.
    
    Args:
        gpu_ids (list): List of GPU IDs for this worker
        worker_port (int): Port for the worker to listen on
        vllm_port (int): Port for the vLLM backend
        worker_index (int): Index of this worker (for logging)
        
    Returns:
        subprocess.Popen: The worker process
    """
    cuda_devices = ','.join(map(str, gpu_ids))
    
    print(f"Starting worker {worker_index + 1} on port {worker_port} (vLLM port {vllm_port}) with GPUs: {cuda_devices}")
    
    # Build the command to start the worker
    cmd = [sys.executable, "-m", "gllm.worker"]
    
    # Set environment variables for the worker
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["PORT"] = str(worker_port)
    env["VLLM_PORT"] = str(vllm_port)
    
    try:
        # Start the worker process
        worker_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        return worker_process
        
    except Exception as e:
        print(f"Error starting worker {worker_index + 1}: {e}")
        return None


def start_cluster(args):
    """Start a cluster with workers and balancer."""
    global balancer_process, worker_processes
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse worker definitions
    try:
        worker_configs = parse_worker_definitions(args.workers)
    except ValueError as e:
        print(f"Error parsing worker definitions: {e}")
        return 1
    
    if not worker_configs:
        print("No workers specified. Use --workers to specify worker configurations.")
        return 1
    
    print(f"Starting cluster with {len(worker_configs)} workers...")
    
    # Start workers
    worker_processes = []
    worker_urls = []
    
    for i, gpu_ids in enumerate(worker_configs):
        worker_port = args.worker_port_range_start + (i * 100)
        vllm_port = worker_port + 10
        
        worker_process = start_worker(gpu_ids, worker_port, vllm_port, i)
        if worker_process:
            worker_processes.append(worker_process)
            worker_urls.append(f"http://localhost:{worker_port}")
        else:
            print(f"Failed to start worker {i + 1}")
            # Clean up any started workers
            for wp in worker_processes:
                if wp and wp.poll() is None:
                    wp.terminate()
            return 1
    
    # Wait a moment for workers to start
    print("Waiting for workers to start...")
    time.sleep(5)
    
    # Check if workers are still running
    failed_workers = []
    for i, worker_process in enumerate(worker_processes):
        if worker_process.poll() is not None:
            failed_workers.append(i + 1)
    
    if failed_workers:
        print(f"Workers {failed_workers} failed to start")
        signal_handler(signal.SIGINT, None)
        return 1
    
    print(f"✓ All {len(worker_processes)} workers started successfully")
    
    # Create args for balancer with worker URLs
    balancer_args = argparse.Namespace(
        port=args.port,
        ip=args.ip,
        existing_workers=worker_urls,
        load_model=args.load_model
    )
    
    # Start the balancer
    print("Starting balancer...")
    result = start_balancer(balancer_args)
    
    return result


def load_model_on_single_worker(worker_url, load_request):
    """Load model on a single worker and return result."""
    try:
        print(f"Loading model on {worker_url}...")
        response = requests.post(
            f"{worker_url}{Endpoints.LOAD_MODEL.value}",
            json=load_request.model_dump(),
            timeout=5  # 5 seconds timeout since load_model kicks off and returns immediately
        )
        
        if response.status_code == 200:
            return ("success", worker_url, None, None)
        else:
            return ("error", worker_url, response.status_code, response.text)
            
    except requests.RequestException as e:
        return ("network_error", worker_url, "Network Error", str(e))


def load_model_on_workers(workers, model_path):
    """Load model on all specified workers in parallel and return results."""
    if not workers:
        raise ValueError("No workers specified. Use --workers to specify worker URLs.")
    
    print(f"Loading model '{model_path}' on {len(workers)} workers in parallel...")
    
    successful_workers = []
    failed_workers = []
    
    load_request = data_def.LoadModelRequest(model_path=model_path, force_reload=False)
    
    # Use ThreadPoolExecutor to make requests in parallel
    with ThreadPoolExecutor(max_workers=min(len(workers), 10)) as executor:
        # Submit all tasks
        future_to_worker = {
            executor.submit(load_model_on_single_worker, worker_url, load_request): worker_url 
            for worker_url in workers
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_worker):
            result_type, worker_url, error_code, error_msg = future.result()
            
            if result_type == "success":
                successful_workers.append(worker_url)
                print(f"✓ Successfully loaded model on {worker_url}")
            else:
                failed_workers.append((worker_url, error_code, error_msg))
                if result_type == "error":
                    print(f"✗ Failed to load model on {worker_url}: {error_code} - {error_msg}")
                else:  # network_error
                    print(f"✗ Failed to connect to {worker_url}: {error_msg}")
    
    # Print summary report
    print(f"\n{'='*60}")
    print("MODEL LOADING REPORT")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Total workers: {len(workers)}")
    print(f"Successful: {len(successful_workers)}")
    print(f"Failed: {len(failed_workers)}")
    
    if successful_workers:
        print(f"\n✓ SUCCESSFUL WORKERS ({len(successful_workers)}):")
        for worker in successful_workers:
            print(f"  - {worker}")
    
    if failed_workers:
        print(f"\n✗ FAILED WORKERS ({len(failed_workers)}):")
        for worker, error_code, error_msg in failed_workers:
            print(f"  - {worker}: {error_code} - {error_msg}")
    
    print(f"{'='*60}")
    
    return len(successful_workers), len(failed_workers)


def start_balancer(args):
    """Start the balancer with the given arguments."""
    global balancer_process
    
    # Check if load-model is specified without workers
    if args.load_model and not args.existing_workers:
        raise ValueError("--load-model requires --existing-workers to be specified. Please provide worker URLs.")
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build the command to start the balancer
    cmd = [
        sys.executable, "-m", "gllm.balancer"
    ]
    
    # Set environment variables for the balancer
    env = os.environ.copy()
    env["PORT"] = str(args.port)
    env["HOST"] = args.ip
    
    print(f"Starting balancer on {args.ip}:{args.port}...")
    
    try:
        # Start the balancer process
        balancer_process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait a moment for the balancer to start
        time.sleep(2)
        
        # Check if the balancer is running
        if balancer_process.poll() is not None:
            stdout, stderr = balancer_process.communicate()
            print(f"Balancer failed to start:")
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
            return 1
        
        print(f"Balancer started successfully (PID: {balancer_process.pid})")
        
        # Register workers if provided
        if args.existing_workers:
            print(f"Registering {len(args.existing_workers)} workers...")
            balancer_url = f"http://{args.ip}:{args.port}"
            
            for worker_url in args.existing_workers:
                try:
                    # Wait a bit more for balancer to be fully ready
                    time.sleep(1)
                    
                    worker_request = data_def.AddWorkerRequest(address=worker_url)
                    response = requests.post(
                        f"{balancer_url}/add_worker",
                        json=worker_request.model_dump(),
                        timeout=10
                    )
                    
                    if response.status_code == 200:
                        print(f"✓ Registered worker: {worker_url}")
                    else:
                        print(f"✗ Failed to register worker {worker_url}: {response.status_code}")
                        
                except requests.RequestException as e:
                    print(f"✗ Failed to register worker {worker_url}: {e}")
        
        # Load model if specified
        if args.load_model:
            try:
                print(f"\nInitiating model loading...")
                successful, failed = load_model_on_workers(args.existing_workers, args.load_model)
                
                if failed > 0:
                    print(f"\nWarning: Model loading failed on {failed} workers.")
                else:
                    print(f"\nSuccess: Model loaded on all {successful} workers.")
                    
            except Exception as e:
                print(f"Error during model loading: {e}")
                return 1
        
        print("Balancer is running. Press Ctrl+C to stop.")
        
        # Keep the main process alive and monitor the subprocess
        try:
            while True:
                if balancer_process.poll() is not None:
                    stdout, stderr = balancer_process.communicate()
                    print("Balancer process has terminated unexpectedly.")
                    print(f"stdout: {stdout}")
                    print(f"stderr: {stderr}")
                    return 1
                time.sleep(1)
                
        except KeyboardInterrupt:
            signal_handler(signal.SIGINT, None)
            
    except Exception as e:
        print(f"Error starting balancer: {e}")
        return 1
    
    return 0


def main():
    """Main entry point for the gllm command line interface."""
    parser = argparse.ArgumentParser(
        description="GLLM - Distributed Language Model Serving System",
        prog="gllm"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="gllm 0.1.0"
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # start-balancer subcommand
    balancer_parser = subparsers.add_parser(
        "start-balancer",
        help="Start the load balancer"
    )
    
    balancer_parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port for the balancer to listen on"
    )
    
    balancer_parser.add_argument(
        "--ip",
        type=str,
        default="0.0.0.0",
        help="IP address for the balancer to bind to (default: 0.0.0.0)"
    )
    
    balancer_parser.add_argument(
        "--existing-workers", "-ew",
        nargs="*",
        help="List of existing worker URLs to register with the balancer"
    )
    
    balancer_parser.add_argument(
        "--load-model",
        type=str,
        help="Model path/name to load on all workers (requires --existing-workers)"
    )
    
    # start-cluster subcommand
    cluster_parser = subparsers.add_parser(
        "start-cluster",
        help="Start a cluster with workers and load balancer"
    )
    
    cluster_parser.add_argument(
        "--port",
        type=int,
        required=True,
        help="Port for the balancer to listen on"
    )
    
    cluster_parser.add_argument(
        "--ip",
        type=str,
        default="0.0.0.0",
        help="IP address for the balancer to bind to (default: 0.0.0.0)"
    )
    
    cluster_parser.add_argument(
        "--workers",
        nargs="*",
        required=True,
        help="Worker GPU configurations (e.g., '3', '3:5', '3,5,2')"
    )
    
    cluster_parser.add_argument(
        "--worker-port-range-start", "-wp",
        type=int,
        required=True,
        help="Starting port for workers (each worker uses port+i*100, vLLM uses port+10)"
    )
    
    cluster_parser.add_argument(
        "--load-model",
        type=str,
        help="Model path/name to load on all workers after startup"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle subcommands
    if args.command == "start-balancer":
        return start_balancer(args)
    elif args.command == "start-cluster":
        return start_cluster(args)
    elif args.command is None:
        # If no subcommand provided, show help message
        print("Hello from GLLM! 🚀")
        print("This is a distributed language model serving system.")
        print("\nAvailable commands:")
        print("  start-balancer    Start the load balancer")
        print("  start-cluster     Start a cluster with workers and load balancer")
        print("\nUse 'gllm <command> --help' for more information about a command.")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 