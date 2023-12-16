import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

class ShowImages:
  """ データセットやデータローダーから画像を表示するためのクラス.

  Methods:
    - show_images: データセットから複数の画像を表示するメソッド.
    - show_images_loader: データローダーから複数の画像を表示するメソッド.
  """
  def __init__(self) -> None:
    print("Active ShowImages instance created")

  # 画像の表示
  def imshow(self, img, ax):
      npimg = img.numpy()
      ax.imshow(np.transpose(npimg, (1, 2, 0)))

  def show_images(self, dataset, num_images_to_display:int=20) -> None:
    """データセットから複数の画像を表示するメソッド.

    Args:
      - dataset: 表示するデータセット.
      - num_images_to_display (int): 表示する画像の数.
    """
    print(f"Shape: {dataset[0].shape}")
    _, axes = plt.subplots(1, num_images_to_display, figsize=(num_images_to_display, 2))

    for i in range(num_images_to_display):
        image, label = dataset[i]
        axes[i].set_title(f"Label: {label}")
        self.imshow(image, axes[i])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()

  def show_images_loader(self, dataset_lodaer:DataLoader, num_images_to_display:int=20) -> None:
    """データローダーから複数の画像を表示するメソッド.

    Args:
      - dataset_loader (DataLoader): 表示するデータローダー.
      - num_images_to_display (int): 表示する画像の数.
    """
    _, axes = plt.subplots(1, num_images_to_display, figsize=(num_images_to_display, 2))
    images, labels = next(iter(dataset_lodaer))
    print(f"Shape: {images[0].shape}")

    for i in range(num_images_to_display):
      axes[i].set_title(f"Label: {labels[i]}")
      self.imshow(images[i], axes[i])
      axes[i].axis('off')

    plt.tight_layout()
    plt.show()
