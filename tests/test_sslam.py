import unittest
import yaml
import os
from ml_collections import ConfigDict

# Add root of the repository to sys.path to allow direct import of project modules
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.settings import load_config, get_model_from_config
# We import these to potentially check config instance types, though direct instantiation isn't the goal here.
from models.sslam import MaeImageClassificationConfig 

class TestSSLAM(unittest.TestCase):

    def setUp(self):
        self.config_path = "configs/config_sslam.yaml"
        self.model_type = "sslam_mae_classification"

    def test_load_sslam_config(self):
        """
        Tests if the SSLAM configuration file (configs/config_sslam.yaml) can be loaded
        and if it contains expected basic structure.
        """
        self.assertTrue(os.path.exists(self.config_path), f"Config file not found at {self.config_path}")
        
        config = load_config(self.model_type, self.config_path)
        
        self.assertIsNotNone(config, "Loaded config should not be None.")
        self.assertIsInstance(config, ConfigDict, "Loaded config should be a ConfigDict.")
        
        self.assertIn("model", config, "Config should have a 'model' section.")
        self.assertIn("model_name", config.model, "'model' section should have 'model_name'.")
        self.assertEqual(config.model.model_name, self.model_type,
                         f"config.model.model_name should be '{self.model_type}'.")
        
        # Example check for another key, can be expanded
        self.assertIn("model_path", config.model, "'model' section should have 'model_path'.")

    def test_instantiate_sslam_model_raises_not_implemented(self):
        """
        Tests if attempting to instantiate the SSLAM model via get_model_from_config
        correctly raises a NotImplementedError.
        """
        self.assertTrue(os.path.exists(self.config_path), f"Config file not found at {self.config_path}")

        with self.assertRaisesRegex(NotImplementedError, 
                                     "SSLAM model \(sslam_mae_classification\) is defined but not yet fully implemented"):
            get_model_from_config(self.model_type, self.config_path)

if __name__ == '__main__':
    unittest.main()
