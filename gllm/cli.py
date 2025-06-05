"""Command line interface for gllm."""

import argparse
import sys
import subprocess
import signal
import os
import time
import requests
from . import data_def
from .consts import Endpoints


# Global variable to store the subprocess
balancer_process = None


def signal_handler(signum, frame):
    """Handle termination signals to clean up subprocess."""
    global balancer_process
    print(f"\nReceived signal {signum}, shutting down...")
    
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


def load_model_on_workers(workers, model_path):
    """Load model on all specified workers and return results."""
    if not workers:
        raise ValueError("No workers specified. Use --workers to specify worker URLs.")
    
    print(f"Loading model '{model_path}' on {len(workers)} workers...")
    
    successful_workers = []
    failed_workers = []
    
    load_request = data_def.LoadModelRequest(model_path=model_path, force_reload=False)
    
    for worker_url in workers:
        try:
            print(f"Loading model on {worker_url}...")
            response = requests.post(
                f"{worker_url}{Endpoints.LOAD_MODEL.value}",
                json=load_request.model_dump(),
                timeout=30  # Longer timeout for model loading
            )
            
            if response.status_code == 200:
                successful_workers.append(worker_url)
                print(f"✓ Successfully loaded model on {worker_url}")
            else:
                failed_workers.append((worker_url, response.status_code, response.text))
                print(f"✗ Failed to load model on {worker_url}: {response.status_code} - {response.text}")
                
        except requests.RequestException as e:
            failed_workers.append((worker_url, "Network Error", str(e)))
            print(f"✗ Failed to connect to {worker_url}: {e}")
    
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
    if args.load_model and not args.workers:
        raise ValueError("--load-model requires --workers to be specified. Please provide worker URLs.")
    
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
        if args.workers:
            print(f"Registering {len(args.workers)} workers...")
            balancer_url = f"http://{args.ip}:{args.port}"
            
            for worker_url in args.workers:
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
                successful, failed = load_model_on_workers(args.workers, args.load_model)
                
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
        "--workers",
        nargs="*",
        help="List of worker URLs to register with the balancer"
    )
    
    balancer_parser.add_argument(
        "--load-model",
        type=str,
        help="Model path/name to load on all workers (requires --workers)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Handle subcommands
    if args.command == "start-balancer":
        return start_balancer(args)
    elif args.command is None:
        # If no subcommand provided, show help message
        print("Hello from GLLM! 🚀")
        print("This is a distributed language model serving system.")
        print("\nAvailable commands:")
        print("  start-balancer    Start the load balancer")
        print("\nUse 'gllm <command> --help' for more information about a command.")
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 