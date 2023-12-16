from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_mnist_dataset(train:bool=True, transform=None, download:bool=True, batch_size:int=64):
    """ MNISTデータセットを取得する関数

    Args:
        train (bool): トレーニングデータセット(True)またはテストデータセット(False)の取得を指定します。
        transform (torchvision.transforms.Transform): データ変換を指定します。
        download (bool): データセットが存在しない場合にダウンロードするかどうかを指定します。
        batch_size (int): データローダーのバッチサイズを指定します。

    Returns:
        torch.utils.data.DataLoader: 指定された条件で構築されたMNISTデータセットを含むデータローダー
    """

    # データ変換が指定されていない場合は、デフォルトの変換を使用
    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 1チャンネルを3チャンネルに変換
            transforms.Resize((224, 224)),  # ViTモデルは大きな画像が必要
            transforms.ToTensor()
        ])

    # データセットの取得
    mnist_dataset = datasets.MNIST(
        root='/usr/src/ai-lab/src/datasets/mnist/data',
        train=train,
        transform=transform,
        download=download
    )

    # データローダーの作成
    data_loader = DataLoader(
        dataset=mnist_dataset,
        batch_size=batch_size,
        shuffle=True if train else False
    )

    return data_loader
