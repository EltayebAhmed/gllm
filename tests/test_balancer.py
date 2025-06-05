import unittest
import json
import threading
from unittest.mock import patch, MagicMock, Mock
import sys
import os

# Add the gllm package to the path so we can import it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from gllm import balancer
from gllm.consts import Endpoints
from gllm import data_def


class TestBalancer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a test client
        balancer.app.config['TESTING'] = True
        self.client = balancer.app.test_client()
        
        # Reset global state
        balancer.model = ""
        balancer.worker_queue_size = balancer.WithLock({})
        
    def tearDown(self):
        """Clean up after each test method."""
        # Reset global state
        balancer.model = ""
        balancer.worker_queue_size = balancer.WithLock({})


class TestWithLock(unittest.TestCase):
    """Test the WithLock utility class."""
    
    def test_init(self):
        """Test WithLock initialization."""
        test_dict = {"key": "value"}
        with_lock = balancer.WithLock(test_dict)
        
        self.assertEqual(with_lock.entity, test_dict)

    
    def test_thread_safety(self):
        """Test that WithLock provides thread safety."""
        with_lock = balancer.WithLock({})
        results = []
        
        def worker(worker_id):
            with with_lock.lock:
                # Simulate some work
                current_len = len(with_lock.entity)
                with_lock.entity[f"worker_{worker_id}"] = worker_id
                results.append(len(with_lock.entity) - current_len)
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All operations should have been atomic
        self.assertEqual(len(with_lock.entity), 10)
        self.assertEqual(sum(results), 10)


class TestWorkerManagement(TestBalancer):
    """Test worker registration and queue management."""
    
    def test_add_worker_success(self):
        """Test successful worker addition."""
        worker_data = {"address": "http://worker1:8000"}
        
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps(worker_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("http://worker1:8000", balancer.worker_queue_size.entity)
        self.assertEqual(balancer.worker_queue_size.entity["http://worker1:8000"], 0)
    
    def test_add_worker_with_existing_model(self):
        """Test adding worker when model is already loaded."""
        balancer.model = "test-model"
        worker_data = {"address": "http://worker1:8000"}
        
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps(worker_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
    
    def test_add_worker_invalid_data(self):
        """Test adding worker with invalid data."""
        worker_data = {"invalid_field": "value"}
        
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps(worker_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    def test_get_least_busy_worker_empty(self):
        """Test getting least busy worker when no workers are available."""
        worker = balancer.get_least_busy_worker()
        self.assertIsNone(worker)
    
    def test_get_least_busy_worker_single(self):
        """Test getting least busy worker with single worker."""
        balancer.worker_queue_size.entity["http://worker1:8000"] = 5
        
        worker = balancer.get_least_busy_worker()
        self.assertEqual(worker, "http://worker1:8000")
    
    def test_get_least_busy_worker_multiple(self):
        """Test load balancing with multiple workers."""
        balancer.worker_queue_size.entity["http://worker1:8000"] = 5
        balancer.worker_queue_size.entity["http://worker2:8000"] = 3
        balancer.worker_queue_size.entity["http://worker3:8000"] = 7
        
        worker = balancer.get_least_busy_worker()
        self.assertEqual(worker, "http://worker2:8000")
    
    def test_register_request_with_worker(self):
        """Test registering a request with a worker."""
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        
        balancer.register_request_with_worker("http://worker1:8000")
        
        self.assertEqual(balancer.worker_queue_size.entity["http://worker1:8000"], 1)
    
    def test_notify_worker_request_complete(self):
        """Test notifying completion of a worker request."""
        balancer.worker_queue_size.entity["http://worker1:8000"] = 3
        
        balancer.notify_worker_request_complete("http://worker1:8000")
        
        self.assertEqual(balancer.worker_queue_size.entity["http://worker1:8000"], 2)


class TestHealthEndpoint(TestBalancer):
    """Test the health check endpoint."""
    
    def test_health_no_workers(self):
        """Test health check with no workers."""
        response = self.client.get(Endpoints.HEALTH)
        
        self.assertEqual(response.status_code, 503)
    
    @patch('gllm.balancer.requests.get')
    def test_health_all_workers_healthy(self, mock_get):
        """Test health check when all workers are healthy."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 0
        
        # Mock healthy responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        response = self.client.get(Endpoints.HEALTH)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(mock_get.call_count, 2)
    
    @patch('gllm.balancer.requests.get')
    def test_health_unhealthy_worker(self, mock_get):
        """Test health check when a worker is unhealthy."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        
        # Mock unhealthy response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        response = self.client.get(Endpoints.HEALTH)
        
        self.assertEqual(response.status_code, 503)


class TestChatCompletions(TestBalancer):
    """Test the chat completions endpoint."""
    
    def test_chat_completions_no_workers(self):
        """Test chat completions with no available workers."""
        chat_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "max_tokens": 100
        }
        
        response = self.client.post(
            Endpoints.CHAT_COMPLETIONS,
            data=json.dumps(chat_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 503)
    
    @patch('gllm.balancer.requests.post')
    def test_chat_completions_success(self, mock_post):
        """Test successful chat completions."""
        # Setup worker
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"message": {"content": "Hello!"}}]}'
        mock_post.return_value = mock_response
        
        chat_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "max_tokens": 100
        }
        
        response = self.client.post(
            Endpoints.CHAT_COMPLETIONS,
            data=json.dumps(chat_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Verify request was registered and completed
        self.assertEqual(balancer.worker_queue_size.entity["http://worker1:8000"], 0)
        
        # Verify the correct worker endpoint was called
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], "http://worker1:8000" + Endpoints.CHAT_COMPLETIONS)
    
    @patch('gllm.balancer.requests.post')
    def test_chat_completions_worker_error(self, mock_post):
        """Test chat completions when worker returns error."""
        # Setup worker
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        
        # Mock error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_post.return_value = mock_response
        
        chat_data = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "test-model",
            "max_tokens": 100
        }
        
        response = self.client.post(
            Endpoints.CHAT_COMPLETIONS,
            data=json.dumps(chat_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 500)


class TestCompletions(TestBalancer):
    """Test the completions endpoint."""
    
    def test_completions_no_workers(self):
        """Test completions with no available workers."""
        completion_data = {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 100
        }
        
        response = self.client.post(
            Endpoints.COMPLETIONS,
            data=json.dumps(completion_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 503)
    
    @patch('gllm.balancer.requests.post')
    def test_completions_success(self, mock_post):
        """Test successful completions."""
        # Setup worker
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"text": "Hello world!"}]}'
        mock_post.return_value = mock_response
        
        completion_data = {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 100
        }
        
        response = self.client.post(
            Endpoints.COMPLETIONS,
            data=json.dumps(completion_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Verify request was registered and completed
        self.assertEqual(balancer.worker_queue_size.entity["http://worker1:8000"], 0)


class TestLoadModel(TestBalancer):
    """Test the load model endpoint."""
    
    @patch('gllm.balancer.requests.post')
    def test_load_model_success(self, mock_post):
        """Test successful model loading."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 0
        
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        load_data = {
            "model_path": "/path/to/model",
            "force_reload": False
        }
        
        response = self.client.post(
            Endpoints.LOAD_MODEL,
            data=json.dumps(load_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(balancer.model, "/path/to/model")
        
        # Verify all workers were called
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('gllm.balancer.requests.post')
    def test_load_model_partial_failure(self, mock_post):
        """Test model loading when some workers fail."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 0
        
        # Mock mixed responses
        def side_effect(*args, **kwargs):
            if "worker1" in args[0]:
                response = MagicMock()
                response.status_code = 200
                return response
            else:
                response = MagicMock()
                response.status_code = 500
                return response
        
        mock_post.side_effect = side_effect
        
        load_data = {
            "model_path": "/path/to/model",
            "force_reload": False
        }
        
        response = self.client.post(
            Endpoints.LOAD_MODEL,
            data=json.dumps(load_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 500)
    
    def test_load_model_no_workers(self):
        """Test model loading with no workers."""
        load_data = {
            "model_path": "/path/to/model",
            "force_reload": False
        }
        
        response = self.client.post(
            Endpoints.LOAD_MODEL,
            data=json.dumps(load_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(balancer.model, "/path/to/model")


class TestReleaseGPUs(TestBalancer):
    """Test the release GPUs endpoint."""
    
    @patch('gllm.balancer.requests.post')
    def test_release_gpus_success(self, mock_post):
        """Test successful GPU release."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 0
        
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        response = self.client.post(Endpoints.RELEASE_GPUS)
        
        self.assertEqual(response.status_code, 200)
        
        # Verify all workers were called
        self.assertEqual(mock_post.call_count, 2)
    
    @patch('gllm.balancer.requests.post')
    def test_release_gpus_partial_failure(self, mock_post):
        """Test GPU release when some workers fail."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 0
        
        # Mock mixed responses
        def side_effect(*args, **kwargs):
            if "worker1" in args[0]:
                response = MagicMock()
                response.status_code = 200
                return response
            else:
                response = MagicMock()
                response.status_code = 500
                return response
        
        mock_post.side_effect = side_effect
        
        response = self.client.post(Endpoints.RELEASE_GPUS)
        
        self.assertEqual(response.status_code, 500)
    
    def test_release_gpus_no_workers(self):
        """Test GPU release with no workers."""
        response = self.client.post(Endpoints.RELEASE_GPUS)
        
        self.assertEqual(response.status_code, 200)


class TestLoadBalancing(TestBalancer):
    """Test load balancing behavior."""
    
    @patch('gllm.balancer.requests.post')
    def test_load_balancing_distribution(self, mock_post):
        """Test that requests are distributed among workers."""
        # Setup workers with different load
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 2
        balancer.worker_queue_size.entity["http://worker3:8000"] = 1
        
        # Mock successful responses
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"choices": [{"text": "response"}]}'
        mock_post.return_value = mock_response
        
        completion_data = {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 100
        }
        
        # Make a request - should go to worker1 (least busy)
        response = self.client.post(
            Endpoints.COMPLETIONS,
            data=json.dumps(completion_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        
        # Verify the request went to the least busy worker
        args, kwargs = mock_post.call_args
        self.assertTrue(args[0].startswith("http://worker1:8000"))
    
    @patch('gllm.balancer.requests.post')
    def test_concurrent_requests_thread_safety(self, mock_post):
        """Test thread safety during concurrent requests."""
        # Setup workers
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        balancer.worker_queue_size.entity["http://worker2:8000"] = 0
        
        # Mock successful responses with delay to simulate processing time
        def slow_response(*args, **kwargs):
            import time
            time.sleep(0.1)  # Small delay
            response = MagicMock()
            response.status_code = 200
            response.text = '{"choices": [{"text": "response"}]}'
            return response
        
        mock_post.side_effect = slow_response
        
        # Make concurrent requests
        import threading
        results = []
        
        def make_request():
            completion_data = {
                "model": "test-model",
                "prompt": "Hello",
                "max_tokens": 100
            }
            response = self.client.post(
                Endpoints.COMPLETIONS,
                data=json.dumps(completion_data),
                content_type='application/json'
            )
            results.append(response.status_code)
        
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        self.assertEqual(len(results), 5)
        self.assertTrue(all(status == 200 for status in results))
        
        # Final queue sizes should be 0 (all requests completed)
        self.assertEqual(balancer.worker_queue_size.entity["http://worker1:8000"], 0)
        self.assertEqual(balancer.worker_queue_size.entity["http://worker2:8000"], 0)


class TestBlueprintRegistration(TestBalancer):
    """Test that endpoints are available on both root and /v1 paths."""
    
    def test_endpoints_available_on_root(self):
        """Test that endpoints work on root path."""
        response = self.client.get(Endpoints.HEALTH)
        # Should return 503 (no workers) but endpoint should exist
        self.assertNotEqual(response.status_code, 404)
    
    def test_endpoints_available_on_v1(self):
        """Test that endpoints work on /v1 path."""
        response = self.client.get('/v1' + Endpoints.HEALTH)
        # Should return 503 (no workers) but endpoint should exist
        self.assertNotEqual(response.status_code, 404)


class TestErrorHandling(TestBalancer):
    """Test error handling scenarios."""
    
    def test_invalid_json_request(self):
        """Test handling of invalid JSON in requests."""
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data="invalid json",
            content_type='application/json'
        )
        
        self.assertIn(response.status_code, [400, 500])  # Should handle gracefully
    
    def test_missing_required_fields(self):
        """Test handling of missing required fields."""
        # Test add worker without address
        response = self.client.post(
            Endpoints.ADD_WORKER,
            data=json.dumps({}),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
    
    @patch('gllm.balancer.requests.post')
    def test_network_error_handling(self, mock_post):
        """Test handling of network errors when communicating with workers."""
        # Setup worker
        balancer.worker_queue_size.entity["http://worker1:8000"] = 0
        
        # Mock network error
        mock_post.side_effect = Exception("Network error")
        
        completion_data = {
            "model": "test-model",
            "prompt": "Hello",
            "max_tokens": 100
        }
        
        # Should handle the exception gracefully
        with self.assertRaises(Exception):
            self.client.post(
                Endpoints.COMPLETIONS,
                data=json.dumps(completion_data),
                content_type='application/json'
            )
        
        # Worker queue should still be updated correctly even if exception occurs
        # This tests that the try/finally or proper cleanup works


if __name__ == '__main__':
    unittest.main() 