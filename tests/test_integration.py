import pytest
import json
import threading
import time
from unittest.mock import patch, MagicMock, Mock
import requests
from flask import Flask
import sys
import os

# Add the gllm package to the path so we can import it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gllm import balancer
from gllm.consts import Endpoints


@pytest.fixture
def mock_worker_server():
    """Create a mock worker server for integration testing."""
    app = Flask(__name__)
    
    @app.route('/health')
    def health():
        return {"status": "healthy"}, 200
    
    @app.route('/chat/completions', methods=['POST'])
    def chat_completions():
        return {
            "choices": [{"message": {"content": "Mock response"}}]
        }, 200
    
    @app.route('/completions', methods=['POST'])
    def completions():
        return {
            "choices": [{"text": "Mock completion"}]
        }, 200
    
    @app.route('/load_model', methods=['POST'])
    def load_model():
        return {"status": "loaded"}, 200
    
    @app.route('/release_gpus', methods=['POST'])
    def release_gpus():
        return {"status": "released"}, 200
    
    return app


@pytest.mark.integration
class TestBalancerIntegration:
    """Integration tests for the balancer service."""
    
    def setup_method(self):
        """Reset balancer state before each test."""
        balancer.model = ""
        balancer.worker_queue_size = balancer.WithLock({})
        balancer.app.config['TESTING'] = True
        self.client = balancer.app.test_client()
    
    def test_full_worker_lifecycle(self):
        """Test the complete lifecycle of adding workers and making requests."""
        # 1. Add a worker
        worker_data = {"address": "http://mock-worker:8000"}
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps(worker_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # 2. Verify worker is registered
        assert "http://mock-worker:8000" in balancer.worker_queue_size.entity
        assert balancer.worker_queue_size.entity["http://mock-worker:8000"] == 0
        
        # 3. Mock the worker's response for chat completions
        with patch('gllm.balancer.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = '{"choices": [{"message": {"content": "Integration test response"}}]}'
            mock_post.return_value = mock_response
            
            # 4. Make a chat completion request
            chat_data = {
                "messages": [{"role": "user", "content": "Test message"}],
                "model": "test-model",
                "max_tokens": 100
            }
            
            response = self.client.post(
                Endpoints.CHAT_COMPLETIONS,
                data=json.dumps(chat_data),
                content_type='application/json'
            )
            
            # 5. Verify the response
            assert response.status_code == 200
            assert "Integration test response" in response.data.decode()
            
            # 6. Verify the correct worker was called
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == "http://mock-worker:8000" + Endpoints.CHAT_COMPLETIONS
    
    @patch('gllm.balancer.requests.post')
    def test_load_balancing_with_multiple_workers(self, mock_post):
        """Test load balancing behavior with multiple workers."""
        # Add multiple workers
        workers = [
            "http://worker1:8000",
            "http://worker2:8000",
            "http://worker3:8000"
        ]
        
        for worker in workers:
            worker_data = {"address": worker}
            response = self.client.post(
                Endpoints.ADD_WORKER,
                data=json.dumps(worker_data),
                content_type='application/json'
            )
            assert response.status_code == 200
        
        # Set different loads for workers
        balancer.worker_queue_size.entity[workers[0]] = 2  # Busy
        balancer.worker_queue_size.entity[workers[1]] = 0  # Free
        balancer.worker_queue_size.entity[workers[2]] = 1  # Somewhat busy
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"text": "Response"}]}'
        mock_post.return_value = mock_response
        
        # Make multiple requests and verify load balancing
        completion_data = {
            "model": "test-model",
            "prompt": "Test",
            "max_tokens": 50
        }
        
        called_workers = []
        
        for i in range(3):
            mock_post.reset_mock()
            response = self.client.post(
                Endpoints.COMPLETIONS,
                data=json.dumps(completion_data),
                content_type='application/json'
            )
            assert response.status_code == 200
            
            # Track which worker was called
            args, kwargs = mock_post.call_args
            called_workers.append(args[0])
            
            # Simulate the request finishing by updating queue sizes
            # (In reality, this happens automatically in the balancer)
        
        # The first request should go to worker2 (least busy)
        assert called_workers[0].startswith("http://worker2:8000")
    
    @patch('gllm.balancer.requests.get')
    @patch('gllm.balancer.requests.post')
    def test_health_check_integration(self, mock_post, mock_get):
        """Test health check with real worker setup."""
        # Add workers
        workers = ["http://worker1:8000", "http://worker2:8000"]
        
        for worker in workers:
            worker_data = {"address": worker}
            response = self.client.post(
                Endpoints.ADD_WORKER,
                data=json.dumps(worker_data),
                content_type='application/json'
            )
            assert response.status_code == 200
        
        # Mock healthy workers
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Check health
        response = self.client.get(Endpoints.HEALTH)
        assert response.status_code == 200
        
        # Verify health was checked for all workers
        assert mock_get.call_count == len(workers)
        
        # Verify the correct health endpoints were called
        called_urls = [call[0][0] for call in mock_get.call_args_list]
        for worker in workers:
            assert worker + Endpoints.HEALTH in called_urls
    
    @patch('gllm.balancer.requests.post')
    def test_model_loading_integration(self, mock_post):
        """Test model loading across multiple workers."""
        # Add workers
        workers = ["http://worker1:8000", "http://worker2:8000"]
        
        for worker in workers:
            worker_data = {"address": worker}
            response = self.client.post(
                Endpoints.ADD_WORKER,
                data=json.dumps(worker_data),
                content_type='application/json'
            )
            assert response.status_code == 200
        
        # Mock successful model loading
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        # Load model
        load_data = {
            "model_path": "/path/to/test/model",
            "force_reload": True
        }
        
        response = self.client.post(
            Endpoints.LOAD_MODEL,
            data=json.dumps(load_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        assert balancer.model == "/path/to/test/model"
        
        # Verify model loading was requested on all workers
        assert mock_post.call_count == len(workers)
        
        # Verify the correct load model endpoints were called
        called_urls = [call[0][0] for call in mock_post.call_args_list]
        for worker in workers:
            assert worker + Endpoints.LOAD_MODEL in called_urls
    
    @pytest.mark.slow
    def test_concurrent_request_handling(self):
        """Test that the balancer can handle concurrent requests properly."""
        # Add workers
        workers = ["http://worker1:8000", "http://worker2:8000"]
        
        for worker in workers:
            worker_data = {"address": worker}
            response = self.client.post(
                Endpoints.ADD_WORKER,
                data=json.dumps(worker_data),
                content_type='application/json'
            )
            assert response.status_code == 200
        
        with patch('gllm.balancer.requests.post') as mock_post:
            # Mock slow response to simulate processing time
            def slow_response(*args, **kwargs):
                time.sleep(0.1)  # Simulate processing time
                response = MagicMock()
                response.status_code = 200
                response.text = '{"choices": [{"text": "Concurrent response"}]}'
                return response
            
            mock_post.side_effect = slow_response
            
            # Make concurrent requests
            results = []
            errors = []
            
            def make_request(request_id):
                try:
                    completion_data = {
                        "model": "test-model",
                        "prompt": f"Request {request_id}",
                        "max_tokens": 50
                    }
                    
                    response = self.client.post(
                        Endpoints.COMPLETIONS,
                        data=json.dumps(completion_data),
                        content_type='application/json'
                    )
                    results.append((request_id, response.status_code))
                except Exception as e:
                    errors.append((request_id, str(e)))
            
            # Start multiple threads
            threads = []
            num_requests = 10
            
            for i in range(num_requests):
                thread = threading.Thread(target=make_request, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == num_requests
            assert all(status_code == 200 for _, status_code in results)
            
            # Verify that requests were distributed among workers
            # (The exact distribution depends on timing, but all should succeed)
            assert mock_post.call_count == num_requests
    
    def test_error_propagation(self):
        """Test that errors from workers are properly propagated."""
        # Add a worker
        worker_data = {"address": "http://error-worker:8000"}
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps(worker_data),
            content_type='application/json'
        )
        assert response.status_code == 200
        
        # Mock worker error
        with patch('gllm.balancer.requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 500
            mock_response.text = "Worker internal error"
            mock_post.return_value = mock_response
            
            # Make a request that should fail
            completion_data = {
                "model": "test-model",
                "prompt": "This should fail",
                "max_tokens": 50
            }
            
            response = self.client.post(
                Endpoints.COMPLETIONS,
                data=json.dumps(completion_data),
                content_type='application/json'
            )
            
            # Verify error is propagated
            assert response.status_code == 500
    
    def test_worker_registration_with_existing_model(self):
        """Test that workers receive the current model when registering."""
        # Load a model first
        load_data = {
            "model_path": "/existing/model",
            "force_reload": False
        }
        
        with patch('gllm.balancer.requests.post'):
            response = self.client.post(
                Endpoints.LOAD_MODEL,
                data=json.dumps(load_data),
                content_type='application/json'
            )
            assert response.status_code == 200
        
        # Now add a worker - it should receive the existing model
        worker_data = {"address": "http://new-worker:8000"}
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps(worker_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        assert response.data.decode() == "/existing/model" 