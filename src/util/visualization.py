import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from gradcam.utils import visualize_cam
from util.path_functions import PathFunctions
from matplotlib.backends.backend_pdf import PdfPages

FIXED_DIRECTORY = '/usr/src/ai-lab/src'
path_functions_instance = PathFunctions()

class Visualization:

  def __init__(self) -> None:
    print("Active Visualization instance created")

  def position_embedding(self, checkpointPath, model):
    checkpoint = torch.load(path_functions_instance.absolutePath(checkpointPath))
    model.load_state_dict(checkpoint)
    model.eval()
    # モデルから位置埋め込みを読み込む
    # N:パッチ数+クラストークン、D:次元数
    pos_embed = model.state_dict()['pos_embed'] # shape:(1, N, D)
    H_and_W = int(np.sqrt(pos_embed.shape[1]-1)) # クラストークン分を引いて平方根をとる
    # パッチ間のコサイン類似度を求め可視化
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, pos_embed.shape[1]):
        sim = F.cosine_similarity(pos_embed[0, i:i+1], pos_embed[0, 1:], dim=1)
        sim = sim.reshape((H_and_W, H_and_W)).detach().cpu().numpy()
        ax = fig.add_subplot(H_and_W, H_and_W, i)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.imshow(sim)

  # Attention Weightを取得するための関数
  def extract(self, pre_model, target, inputs):
      feature = None
      def forward_hook(module, inputs, outputs):
          # 順伝搬の出力を features というグローバル変数に記録する
          global blocks
          blocks = outputs.detach()
      # コールバック関数を登録する
      handle = target.register_forward_hook(forward_hook) # 推論する
      pre_model.eval()
      pre_model(inputs)
      # コールバック関数を解除する
      handle.remove()
      return blocks

  def attention_map(self, model, checkpointPath:str, imagePath:str, imageSize:int=224, checkpointNeedModel:bool=False):
      checkpoint = torch.load(path_functions_instance.absolutePath(checkpointPath))
      if checkpointNeedModel:
        model.load_state_dict(checkpoint["model"])
      else:
        model.load_state_dict(checkpoint)
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model.to(device)
      model.eval()

      # 画像ファイルを読み込み
      BaseImage = Image.open(path_functions_instance.absolutePath(imagePath)).convert('RGB')
      normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
      transform = transforms.Compose([
        transforms.Resize(imageSize),
        transforms.CenterCrop(imageSize),
        transforms.ToTensor(),
        normalize
      ])
      invTrans = transforms.Compose([
        transforms.Normalize(
          mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
          std=[1/0.229, 1/0.224, 1/0.255]),
      ])
      img = transform(BaseImage)
      img = img.view(1, *img.shape)
      print(f"BaseImageInfo: (Size: {BaseImage.size}, Mode: {BaseImage.mode})")
      print(f"TransformImageInfo: {img.shape}")

      # blockごと(Transformer Encoderのlayer)のAttention Weightを取得 # L:層数, H:ヘッド数、N:パッチ数+クラストークン
      attention_weight = []
      for i in range(len(model.blocks)):
        target_module = model.blocks[i].attn.attn_drop
        features = self.extract(model, target_module, img.to(device)) # shape: (1, H, N, N)
        attention_weight.append([features.to('cpu').detach().numpy().copy()])
      attention_weight = np.squeeze(np.concatenate(attention_weight), axis=1) # shape: (L, H, N, N)

      print(f"attetionWeightInfo(L, H, N, N): {attention_weight.shape}")

      # ヘッド方向に平均
      mean_head = np.mean(attention_weight, axis=1) # shape: (L, N, N)
      print(f"attetionWeightInfo(L, N, N): {mean_head.shape}")
      # NxNの単位行列を加算
      mean_head = mean_head + np.eye(mean_head.shape[1])
      # 正規化
      mean_head = mean_head / mean_head.sum(axis=(1, 2))[:, np.newaxis, np.newaxis] # 層方向に乗算
      v = mean_head[-1]
      for n in range(1, len(mean_head)):
          v = np.matmul(v, mean_head[-1 - n])
      print(f"v.shape: {v.shape}")
      # クラストークンと各パッチトークン間とのAttention Weightから、
      # 入力画像サイズまで正規化しながらリサイズしてAttention Mapを生成
      mask = v[0, 1:].reshape(14, 14)
      attention_map = cv2.resize(mask / mask.max(), (img.shape[2], img.shape[3]))[..., np.newaxis]
      # attention_map = cv2.resize(v, (imageSize, imageSize), interpolation=cv2.INTER_LINEAR)
      inv_tensor = invTrans(img)[0]
      # Attention MapとAttentionをかけた画像を生成
      mask = torch.from_numpy(attention_map.astype(np.float32))
      _, result = visualize_cam(mask, img.to(device))
      
      return [inv_tensor, attention_map, result]
  
  def show_attention_map(self, imagesList:list, pdfPath:str='', savePdf:bool=True):
      if imagesList:
        with PdfPages(path_functions_instance.absolutePath(pdfPath)) as pdf:
          # グラフを作成
          plt.figure(figsize=[20,20], tight_layout=True)
          maxRow=len(imagesList)
          maxLen=len(imagesList[0])
          imageIndex = 0

          for idx, result in enumerate(imagesList):
              input_image, attention_map, image_with_attention = result
              # 入力画像
              plt.subplot(maxRow, maxLen, 1+imageIndex)
              plt.imshow(input_image.permute(1,2,0), vmin=0, vmax=1)
              plt.title("Input Image")
              plt.axis("off")
              # Attention Mapの画像
              plt.subplot(maxRow, maxLen, 2+imageIndex)
              plt.imshow(attention_map, cmap='jet', vmin=0, vmax=1)
              plt.title("Attention Map")
              plt.axis("off")
              # Attention Mapをかけた画像
              plt.subplot(maxRow, maxLen, 3+imageIndex)
              plt.imshow(image_with_attention.detach().cpu().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
              plt.title("Input Image with Attention")
              plt.axis("off")
              imageIndex += maxLen
          if savePdf: 
              pdf.savefig(bbox_inches='tight')
      else:
          print("Error: noimag")
