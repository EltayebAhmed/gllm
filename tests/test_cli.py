"""Unit tests for the CLI module."""

import pytest
import argparse
import sys
import subprocess
import signal
import time
from unittest.mock import Mock, patch, MagicMock, call
from concurrent.futures import ThreadPoolExecutor

import requests

from gllm.bin.cli import (
    parse_gpu_specification,
    parse_worker_definitions,
    start_worker,
    start_cluster,
    load_model_on_workers,
    register_workers_with_balancer,
    start_balancer_process,
    start_balancer,
    main,
    signal_handler
)
from gllm import data_def
from gllm.consts import Endpoints


class TestGpuParsing:
    """Test GPU specification parsing functions."""
    
    def test_parse_gpu_specification_single(self):
        """Test parsing a single GPU specification."""
        result = parse_gpu_specification("3")
        assert result == [3]
    
    def test_parse_gpu_specification_range(self):
        """Test parsing a GPU range specification."""
        result = parse_gpu_specification("3:5")
        assert result == [3, 4, 5]
    
    def test_parse_gpu_specification_comma_separated(self):
        """Test parsing comma-separated GPU specification."""
        result = parse_gpu_specification("3,5,2")
        assert result == [3, 5, 2]
    
    def test_parse_worker_definitions(self):
        """Test parsing worker definitions into GPU configurations."""
        worker_defs = ["3", "0:1", "2,4,6"]
        result = parse_worker_definitions(worker_defs)
        expected = [[3], [0, 1], [2, 4, 6]]
        assert result == expected


class TestWorkerManagement:
    """Test worker management functions."""
    
    @patch('gllm.bin.cli.subprocess.Popen')
    @patch('gllm.bin.cli.sys.executable', '/usr/bin/python')
    def test_start_worker_success(self, mock_popen):
        """Test successful worker startup."""
        mock_process = Mock()
        mock_popen.return_value = mock_process
        
        gpu_ids = [0, 1]
        worker_port = 8000
        vllm_port = 8010
        worker_index = 0
        router_address = "http://localhost:7000"
        
        result = start_worker(gpu_ids, worker_port, vllm_port, worker_index, router_address)
        
        assert result == mock_process
        mock_popen.assert_called_once()
        
        # Check the call arguments
        call_args = mock_popen.call_args
        cmd = call_args[0][0]
        assert cmd == ['/usr/bin/python', '-m', 'gllm.worker']
        
        env = call_args[1]['env']
        assert env['CUDA_VISIBLE_DEVICES'] == '0,1'
        assert env['PORT'] == '8000'
        assert env['VLLM_PORT'] == '8010'
        assert env['ROUTER_ADDRESS'] == 'http://localhost:7000'
    
    @patch('gllm.bin.cli.subprocess.Popen')
    def test_start_worker_exception(self, mock_popen):
        """Test worker startup with exception."""
        mock_popen.side_effect = Exception("Failed to start process")
        
        result = start_worker([0], 8000, 8010, 0, "http://localhost:7000")
        
        assert result is None
    
    @patch('gllm.bin.cli.subprocess.Popen')
    @patch('gllm.bin.cli.time.sleep')
    def test_start_balancer_process_success(self, mock_sleep, mock_popen):
        """Test successful balancer startup."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.pid = 12345
        mock_popen.return_value = mock_process
        
        result = start_balancer_process("0.0.0.0", 7000)
        
        assert result == mock_process
        mock_popen.assert_called_once()
        
        # Check environment variables
        call_args = mock_popen.call_args
        env = call_args[1]['env']
        assert env['PORT'] == '7000'
        assert env['HOST'] == '0.0.0.0'
    
    @patch('gllm.bin.cli.subprocess.Popen')
    @patch('gllm.bin.cli.time.sleep')
    def test_start_balancer_process_failure(self, mock_sleep, mock_popen):
        """Test balancer startup failure."""
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process exited
        mock_popen.return_value = mock_process
        
        result = start_balancer_process("0.0.0.0", 7000)
        
        assert result is None


class TestHttpCommunication:
    """Test HTTP communication with workers and balancer."""
    
    @patch('gllm.bin.cli.requests.post')
    def test_register_workers_with_balancer_success(self, mock_post):
        """Test successful worker registration."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        workers = ["http://localhost:8000", "http://localhost:8100"]
        balancer_url = "http://localhost:7000"
        
        successful, failed = register_workers_with_balancer(workers, balancer_url)
        
        assert successful == 2
        assert failed == 0
        assert mock_post.call_count == 2
    
    @patch('gllm.bin.cli.requests.post')
    def test_register_workers_with_balancer_partial_failure(self, mock_post):
        """Test worker registration with partial failures."""
        responses = [Mock(), Mock()]
        responses[0].status_code = 200
        responses[1].status_code = 500
        mock_post.side_effect = responses
        
        workers = ["http://localhost:8000", "http://localhost:8100"]
        balancer_url = "http://localhost:7000"
        
        successful, failed = register_workers_with_balancer(workers, balancer_url)
        
        assert successful == 1
        assert failed == 1
    
    @patch('gllm.bin.cli.requests.post')
    def test_register_workers_with_balancer_connection_error(self, mock_post):
        """Test worker registration with connection errors."""
        mock_post.side_effect = requests.RequestException("Connection failed")
        
        workers = ["http://localhost:8000"]
        balancer_url = "http://localhost:7000"
        
        successful, failed = register_workers_with_balancer(workers, balancer_url)
        
        assert successful == 0
        assert failed == 1
    
    @patch('gllm.bin.cli.requests.post')
    def test_load_model_on_workers_success(self, mock_post):
        """Test successful model loading."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "Model loaded successfully"
        mock_post.return_value = mock_response
        
        balancer_url = "http://localhost:7000"
        model_path = "meta-llama/Llama-2-7b-hf"
        
        successful, failed = load_model_on_workers(balancer_url, model_path)
        
        assert successful == 1
        assert failed == 0
        
        # Check the request
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == f"{balancer_url}{Endpoints.LOAD_MODEL.value}"
        
        # Check the request body
        request_data = call_args[1]['json']
        expected_request = data_def.LoadModelRequest(model_path=model_path, force_reload=False)
        assert request_data == expected_request.model_dump()
    
    @patch('gllm.bin.cli.requests.post')
    def test_load_model_on_workers_failure(self, mock_post):
        """Test model loading failure."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_post.return_value = mock_response
        
        balancer_url = "http://localhost:7000"
        model_path = "meta-llama/Llama-2-7b-hf"
        
        successful, failed = load_model_on_workers(balancer_url, model_path)
        
        assert successful == 0
        assert failed == 1


class TestSignalHandling:
    """Test signal handling and cleanup."""
    
    @patch('gllm.bin.cli.sys.exit')
    @patch('gllm.bin.cli.ThreadPoolExecutor')
    def test_signal_handler_cleanup(self, mock_executor, mock_exit):
        """Test signal handler cleanup of processes."""
        # Set up global variables
        import gllm.bin.cli as cli_module
        
        # Mock worker processes
        mock_worker1 = Mock()
        mock_worker1.poll.return_value = None  # Still running
        mock_worker2 = Mock()
        mock_worker2.poll.return_value = None  # Still running
        
        # Mock balancer process
        mock_balancer = Mock()
        mock_balancer.poll.return_value = None  # Still running
        
        cli_module.worker_processes = [mock_worker1, mock_worker2]
        cli_module.balancer_process = mock_balancer
        
        # Mock executor context manager
        mock_executor_instance = Mock()
        mock_executor.return_value.__enter__.return_value = mock_executor_instance
        
        # Call signal handler
        signal_handler(signal.SIGINT, None)
        
        # Check that workers were terminated
        mock_worker1.terminate.assert_called_once()
        mock_worker2.terminate.assert_called_once()
        
        # Check that balancer was terminated
        mock_balancer.terminate.assert_called_once()
        
        # Check that sys.exit was called
        mock_exit.assert_called_once_with(0)


class TestMainFunction:
    """Test the main CLI entry point."""
    
    @patch('gllm.bin.cli.start_balancer')
    def test_main_start_balancer_command(self, mock_start_balancer):
        """Test main function with start-balancer command."""
        mock_start_balancer.return_value = 0
        
        test_args = [
            "gllm", "start-balancer", 
            "--port", "7000",
            "--ip", "0.0.0.0"
        ]
        
        with patch.object(sys, 'argv', test_args):
            result = main()
        
        assert result == 0
        mock_start_balancer.assert_called_once()
        
        # Check the parsed arguments
        call_args = mock_start_balancer.call_args[0][0]
        assert call_args.port == 7000
        assert call_args.ip == "0.0.0.0"
        assert call_args.existing_workers is None
        assert call_args.load_model is None
    
    @patch('gllm.bin.cli.start_cluster')
    def test_main_start_cluster_command(self, mock_start_cluster):
        """Test main function with start-cluster command."""
        mock_start_cluster.return_value = 0
        
        test_args = [
            "gllm", "start-cluster",
            "--port", "7000",
            "--workers", "0", "1:2",
            "--worker-port-range-start", "8000"
        ]
        
        with patch.object(sys, 'argv', test_args):
            result = main()
        
        assert result == 0
        mock_start_cluster.assert_called_once()
        
        # Check the parsed arguments
        call_args = mock_start_cluster.call_args[0][0]
        assert call_args.port == 7000
        assert call_args.workers == ["0", "1:2"]
        assert call_args.worker_port_range_start == 8000
    
    def test_main_no_command(self):
        """Test main function with no command (shows help)."""
        test_args = ["gllm"]
        
        with patch.object(sys, 'argv', test_args):
            result = main()
        
        assert result == 0
    
    @patch('gllm.bin.cli.start_balancer')
    def test_main_with_existing_workers_and_model(self, mock_start_balancer):
        """Test main function with existing workers and model loading."""
        mock_start_balancer.return_value = 0
        
        test_args = [
            "gllm", "start-balancer",
            "--port", "7000",
            "--existing-workers", "http://localhost:8000", "http://localhost:8100",
            "--load-model", "meta-llama/Llama-2-7b-hf"
        ]
        
        with patch.object(sys, 'argv', test_args):
            result = main()
        
        assert result == 0
        
        # Check the parsed arguments
        call_args = mock_start_balancer.call_args[0][0]
        assert call_args.existing_workers == ["http://localhost:8000", "http://localhost:8100"]
        assert call_args.load_model == "meta-llama/Llama-2-7b-hf"


class TestBalancerStart:
    """Test the start_balancer function."""
    
    @patch('gllm.bin.cli.signal.signal')
    @patch('gllm.bin.cli.start_balancer_process')
    @patch('gllm.bin.cli.register_workers_with_balancer')
    @patch('gllm.bin.cli.load_model_on_workers')
    @patch('gllm.bin.cli.time.sleep')
    def test_start_balancer_success_with_workers_and_model(
        self, mock_sleep, mock_load_model, mock_register, mock_start_process, mock_signal
    ):
        """Test successful balancer start with workers and model loading."""
        
        # Mock successful process start
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_start_process.return_value = mock_process
        
        # Mock successful worker registration
        mock_register.return_value = (2, 0)  # 2 successful, 0 failed
        
        # Mock successful model loading
        mock_load_model.return_value = (1, 0)  # 1 successful, 0 failed
        
        # Create test arguments
        args = argparse.Namespace(
            ip="0.0.0.0",
            port=7000,
            existing_workers=["http://localhost:8000", "http://localhost:8100"],
            load_model="meta-llama/Llama-2-7b-hf"
        )
        
        # Mock the polling loop to exit after one iteration
        def mock_poll_side_effect():
            # First call returns None (running), second call raises KeyboardInterrupt
            calls = [None]
            def poll():
                if calls:
                    calls.pop()
                    return None
                raise KeyboardInterrupt()
            return poll
        
        mock_process.poll = mock_poll_side_effect()
        
        # Mock signal handler
        with patch('gllm.bin.cli.signal_handler') as mock_signal_handler:
            result = start_balancer(args)
        
        assert result == 0
        mock_start_process.assert_called_once_with("0.0.0.0", 7000)
        mock_register.assert_called_once()
        mock_load_model.assert_called_once()
        mock_signal_handler.assert_called_once()
    
    @patch('gllm.bin.cli.start_balancer_process')
    def test_start_balancer_process_start_failure(self, mock_start_process):
        """Test balancer start when process fails to start."""
        mock_start_process.return_value = None
        
        args = argparse.Namespace(
            ip="0.0.0.0",
            port=7000,
            existing_workers=None,
            load_model=None
        )
        
        result = start_balancer(args)
        
        assert result == 1
    
    def test_start_balancer_load_model_without_workers(self):
        """Test that load-model without workers raises an error."""
        args = argparse.Namespace(
            ip="0.0.0.0",
            port=7000,
            existing_workers=None,
            load_model="meta-llama/Llama-2-7b-hf"
        )
        
        with pytest.raises(ValueError, match="--load-model requires --existing-workers"):
            start_balancer(args)


class TestClusterStart:
    """Test the start_cluster function."""
    
    @patch('gllm.bin.cli.signal.signal')
    @patch('gllm.bin.cli.parse_worker_definitions')
    @patch('gllm.bin.cli.start_worker')
    @patch('gllm.bin.cli.start_balancer_process')
    @patch('gllm.bin.cli.register_workers_with_balancer')
    @patch('gllm.bin.cli.load_model_on_workers')
    @patch('gllm.bin.cli.time.sleep')
    def test_start_cluster_success(
        self, mock_sleep, mock_load_model, mock_register, mock_start_balancer,
        mock_start_worker, mock_parse_workers, mock_signal
    ):
        """Test successful cluster startup."""
        
        # Mock worker parsing
        mock_parse_workers.return_value = [[0], [1, 2]]
        
        # Mock successful worker starts
        mock_worker1 = Mock()
        mock_worker1.poll.return_value = None
        mock_worker2 = Mock()
        mock_worker2.poll.return_value = None
        mock_start_worker.side_effect = [mock_worker1, mock_worker2]
        
        # Mock successful balancer start
        mock_balancer = Mock()
        mock_balancer.poll.return_value = None
        mock_start_balancer.return_value = mock_balancer
        
        # Mock successful registrations and model loading
        mock_register.return_value = (2, 0)
        mock_load_model.return_value = (1, 0)
        
        args = argparse.Namespace(
            port=7000,
            ip="0.0.0.0",
            workers=["0", "1:2"],
            worker_port_range_start=8000,
            load_model="meta-llama/Llama-2-7b-hf"
        )
        
        # Mock the polling loop to exit after one iteration
        def mock_poll_side_effect():
            calls = [None]
            def poll():
                if calls:
                    calls.pop()
                    return None
                raise KeyboardInterrupt()
            return poll
        
        mock_balancer.poll = mock_poll_side_effect()
        
        with patch('gllm.bin.cli.signal_handler') as mock_signal_handler:
            result = start_cluster(args)
        
        assert result == 0
        
        # Verify worker starts
        assert mock_start_worker.call_count == 2
        mock_start_worker.assert_any_call([0], 8000, 8010, 0, "http://localhost:7000")
        mock_start_worker.assert_any_call([1, 2], 8100, 8110, 1, "http://localhost:7000")
        
        # Verify balancer start
        mock_start_balancer.assert_called_once_with("0.0.0.0", 7000)
        
        # Verify registrations and model loading
        assert mock_register.call_count == 2  # existing workers + new workers
        mock_load_model.assert_called_once()
    
    @patch('gllm.bin.cli.parse_worker_definitions')
    def test_start_cluster_invalid_workers(self, mock_parse_workers):
        """Test cluster start with invalid worker definitions."""
        mock_parse_workers.side_effect = ValueError("Invalid worker definition")
        
        args = argparse.Namespace(
            workers=["invalid"],
            port=7000,
            ip="0.0.0.0",
            worker_port_range_start=8000,
            load_model=None
        )
        
        result = start_cluster(args)
        
        assert result == 1
    
    @patch('gllm.bin.cli.parse_worker_definitions')
    def test_start_cluster_no_workers(self, mock_parse_workers):
        """Test cluster start with no worker configurations."""
        mock_parse_workers.return_value = []
        
        args = argparse.Namespace(
            workers=[],
            port=7000,
            ip="0.0.0.0",
            worker_port_range_start=8000,
            load_model=None
        )
        
        result = start_cluster(args)
        
        assert result == 1
    
    @patch('gllm.bin.cli.signal.signal')
    @patch('gllm.bin.cli.parse_worker_definitions')
    @patch('gllm.bin.cli.start_worker')
    def test_start_cluster_worker_start_failure(
        self, mock_start_worker, mock_parse_workers, mock_signal
    ):
        """Test cluster start when worker fails to start."""
        mock_parse_workers.return_value = [[0], [1]]
        
        # First worker succeeds, second fails
        mock_worker1 = Mock()
        mock_worker1.poll.return_value = None  # Running
        mock_start_worker.side_effect = [mock_worker1, None]
        
        args = argparse.Namespace(
            workers=["0", "1"],
            port=7000,
            ip="0.0.0.0",
            worker_port_range_start=8000,
            load_model=None
        )
        
        result = start_cluster(args)
        
        assert result == 1
        # Verify direct cleanup was called (not signal handler cleanup)
        mock_worker1.terminate.assert_called_once()


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_worker_list_registration(self):
        """Test registering empty worker list."""
        successful, failed = register_workers_with_balancer([], "http://localhost:7000")
        assert successful == 0
        assert failed == 0
    
    def test_parse_gpu_specification_edge_cases(self):
        """Test GPU specification parsing edge cases."""
        # Test single range
        assert parse_gpu_specification("5:5") == [5]
        
        # Test empty list (should not happen in practice but test for robustness)
        with pytest.raises(ValueError):
            parse_gpu_specification("")
    
    @patch('gllm.bin.cli.time.sleep')
    @patch('gllm.bin.cli.requests.post')
    def test_register_workers_with_timeout(self, mock_post, mock_sleep):
        """Test worker registration with timeout."""
        mock_post.side_effect = requests.Timeout("Request timed out")
        
        workers = ["http://localhost:8000"]
        balancer_url = "http://localhost:7000"
        
        successful, failed = register_workers_with_balancer(workers, balancer_url)
        
        assert successful == 0
        assert failed == 1


if __name__ == "__main__":
    pytest.main([__file__]) 