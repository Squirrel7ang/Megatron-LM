import unittest
import json
import tempfile
import os
import sys

# Ensure the core module is discoverable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.master.parser.cluster_parser import ClusterParser

class TestClusterParser(unittest.TestCase):
    def setUp(self):
        """Sets up a temporary valid cluster configuration."""
        self.valid_data = {
            "cluster_metadata": {
                "name": "TianShu Cluster",
                "total_nodes": 1,
                "master_address": "10.31.10.210",
                "master_port": 12345,
                "network_type": "Ethernet",
                "is_heterogeneous": False
            },
            "hardware_library": {
                "TianGai-150": {"memory_capacity_gb": 32, "theoretical_tflops": 200}
            },
            "nodes": [
                {
                    "node_id": 0, "hostname": "u210", "ip": "10.31.10.210",
                    "gpus": [
                        {"global_id": 0, "local_id": 0, "type": "TianGai-150"},
                        {"global_id": 1, "local_id": 1, "type": "TianGai-150"}
                    ]
                }
            ]
        }
        self.temp_file = self._create_temp_json(self.valid_data)

    def tearDown(self):
        """Cleans up the temporary file."""
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def _create_temp_json(self, data: dict) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
        return path

    def test_valid_parsing(self):
        """Validates that parsing returns the expected context structure."""
        parser = ClusterParser(self.temp_file)
        cluster = parser.parse()
        
        self.assertEqual(cluster.cluster_name, "TianShu Cluster")
        self.assertEqual(len(cluster.nodes), 1)
        self.assertEqual(cluster.nodes[0].gpus[0].type, "TianGai-150")
        self.assertEqual(cluster.nodes[0].gpus[0].memory_capacity_gb, 32)

    def test_invalid_gpu_type(self):
        """Checks if parser raises error for undefined GPU types."""
        data = self.valid_data.copy()
        data["nodes"][0]["gpus"][0]["type"] = "Unknown-Hardware"
        path = self._create_temp_json(data)
        
        with self.assertRaises(ValueError):
            ClusterParser(path).parse()
        os.remove(path)

    def test_duplicate_global_id(self):
        """Checks if parser detects global_id collisions."""
        data = self.valid_data.copy()
        data["nodes"][0]["gpus"][1]["global_id"] = 0
        path = self._create_temp_json(data)
        
        with self.assertRaisesRegex(ValueError, "Duplicate global_id"):
            ClusterParser(path).parse()
        os.remove(path)

if __name__ == '__main__':
    unittest.main()