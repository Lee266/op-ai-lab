from src.datasets.mnist.mnist_data import get_mnist_dataset

train_data_loader = get_mnist_dataset(train=True, batch_size=64)