import tensorflow as tf
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class GPUManager:
    def __init__(self, memory_growth: bool = True, mixed_precision: bool = False):
        """
        Initialize GPU manager
        
        Args:
            memory_growth: Enable dynamic memory growth to avoid OOM
            mixed_precision: Enable mixed precision training for speed and memory efficiency
        """
        self.memory_growth = memory_growth
        self.mixed_precision = mixed_precision
        self.gpu_available = False
        self.device_info = None
    
    def detect_gpu(self) -> bool:
        """
        Detect GPU availability
        
        Returns:
            True if GPU is available
        """
        gpus = tf.config.list_physical_devices('GPU')
        self.gpu_available = len(gpus) > 0
        
        if self.gpu_available:
            logger.info(f"GPU detected: {len(gpus)} GPU(s) available")
            for i, gpu in enumerate(gpus):
                logger.info(f"  GPU {i}: {gpu}")
        else:
            logger.info("No GPU detected. Will use CPU for training.")
        
        return self.gpu_available
    
    def configure_memory_growth(self) -> None:
        """
        Configure GPU memory growth to prevent OOM errors
        """
        if not self.gpu_available:
            logger.info("GPU not available. Skipping memory growth configuration.")
            return
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            logger.info(f"Dynamic memory growth enabled for {len(gpus)} GPU(s)")
            logger.info("GPU will allocate memory as needed, reducing OOM risk")
            
        except RuntimeError as e:
            logger.error(f"Failed to configure memory growth: {str(e)}")
    
    def set_memory_limit(self, limit_gb: int = None) -> None:
        """
        Set GPU memory limit
        
        Args:
            limit_gb: Memory limit in GB (None for no limit)
        """
        if not self.gpu_available or limit_gb is None:
            return
        
        try:
            gpus = tf.config.list_physical_devices('GPU')
            
            for gpu in gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit_gb * 1024)]
                )
            
            logger.info(f"GPU memory limit set to {limit_gb} GB")
            
        except RuntimeError as e:
            logger.error(f"Failed to set memory limit: {str(e)}")
    
    def enable_mixed_precision(self) -> None:
        """
        Enable mixed precision training for better performance and lower memory usage
        """
        if not self.mixed_precision:
            return
        
        try:
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            logger.info("Mixed precision training enabled (float16)")
            logger.info("This reduces memory usage and speeds up training")
            
        except Exception as e:
            logger.warning(f"Failed to enable mixed precision: {str(e)}")
            logger.info("Continuing with standard precision")
    
    def initialize(self, memory_limit_gb: int = None) -> str:
        """
        Initialize GPU configuration
        
        Args:
            memory_limit_gb: Optional memory limit in GB
            
        Returns:
            Device type used ('GPU' or 'CPU')
        """
        logger.info("Initializing GPU/CPU configuration...")
        
        self.detect_gpu()
        
        if self.gpu_available:
            self.configure_memory_growth()
            
            if memory_limit_gb:
                self.set_memory_limit(memory_limit_gb)
            
            if self.mixed_precision:
                self.enable_mixed_precision()
            
            device_type = 'GPU'
            logger.info("GPU training mode enabled")
            
        else:
            device_type = 'CPU'
            logger.info("CPU training mode enabled")
        
        return device_type
    
    def get_device_info(self) -> dict:
        """
        Get device information
        
        Returns:
            Dictionary with device information
        """
        info = {
            'gpu_available': self.gpu_available,
            'num_gpus': len(tf.config.list_physical_devices('GPU')) if self.gpu_available else 0,
            'num_cpus': len(tf.config.list_physical_devices('CPU')),
            'device_type': 'GPU' if self.gpu_available else 'CPU'
        }
        
        if self.gpu_available:
            gpus = tf.config.list_physical_devices('GPU')
            info['gpus'] = [str(gpu) for gpu in gpus]
        
        return info
    
    def print_device_info(self) -> None:
        """
        Print device information
        """
        info = self.get_device_info()
        
        print("\n" + "=" * 60)
        print("Device Configuration")
        print("=" * 60)
        print(f"GPU Available: {info['gpu_available']}")
        print(f"Number of GPUs: {info['num_gpus']}")
        print(f"Number of CPUs: {info['num_cpus']}")
        print(f"Device Type: {info['device_type']}")
        
        if info['gpu_available']:
            print("\nGPU Details:")
            for gpu in info['gpus']:
                print(f"  {gpu}")
        
        print("\nMemory Configuration:")
        print("  Dynamic memory growth: Enabled")
        print("  Mode: Allocate memory as needed")
        print("=" * 60 + "\n")
