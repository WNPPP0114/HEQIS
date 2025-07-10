from abc import ABC, abstractmethod
import random, torch, numpy as np
from utils.util import setup_device
import os

class MAABase(ABC):
    """
    Abstract base class for the MAA framework,
    defining core method interfaces.
    All subclasses must implement the following methods.
    """

    def __init__(self, N_pairs, batch_size, num_epochs,
                 generator_names, discriminators_names,
                 ckpt_dir, output_dir,
                 initial_learning_rate = 2e-4,
                 train_split = 0.8,
                 precise = torch.float32,
                 do_distill_epochs: int = 1,
                 cross_finetune_epochs: int = 5,
                 device = None,
                 seed=None,
                 ckpt_path="auto",):
        """
        Initialize necessary hyperparameters.

        :param N_pairs: Number of generators or discriminators
        :param batch_size: Mini-batch size
        :param num_epochs: Scheduled training epochs
        :param initial_learning_rate: Initial learning rate
        :param generators: Recommended to be an iterable object, including generators with different features
        :param discriminators: Recommended to be an iterable object, can be the same discriminator
        :param ckpt_path: Checkpoints for each model
        :param output_path: Output path for visualization, loss function logs, etc.
        """

        self.N = N_pairs
        self.initial_learning_rate = initial_learning_rate
        self.generator_names = generator_names
        self.discriminators_names = discriminators_names
        self.ckpt_dir = ckpt_dir
        self.ckpt_path = ckpt_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.train_split = train_split
        self.seed = seed
        self.do_distill_epochs = do_distill_epochs
        self.cross_finetune_epochs = cross_finetune_epochs
        self.device = device
        self.precise = precise

        self.set_seed(self.seed)  # Initialize random seed
        self.device = setup_device(device)
        print("Running Device:", self.device)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print("Output directory created! ")

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
            print("Checkpoint directory created! ")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @abstractmethod
    def process_data(self):
        """Data preprocessing, including reading, cleaning, splitting, etc."""
        pass

    @abstractmethod
    def init_model(self):
        """Model structure initialization"""
        pass

    @abstractmethod
    def init_dataloader(self):
        """Initialize data loaders for training and evaluation"""
        pass

    @abstractmethod
    def init_hyperparameters(self):
        """Initialize hyperparameters required for training"""
        pass

    @abstractmethod
    def train(self):
        """Execute the training process"""
        pass

    @abstractmethod
    def save_models(self):
        """Execute the training process"""
        pass

    @abstractmethod
    def distill(self):
        """Execute the knowledge distillation process"""
        pass

    @abstractmethod
    def visualize_and_evaluate(self):
        """Evaluate model performance and visualize results"""
        pass

    @abstractmethod
    def init_history(self):
        """Initialize the metric recording structure during training"""
        pass