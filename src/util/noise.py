import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Noise:
  def __init__(self) -> None:
        print("Active Time instance created")
  
  def overlay_images(self, background_path:str, overlay_path:str, output_path:str, alpha=0.5, show_preview=False):
      """2つの画像を重ねて結果を保存する関数

      Args:
          background_path (str): 背景画像へのパス。
          overlay_path (str): 重ねる画像へのパス。
          output_path (str): 結果を保存するパス。
          alpha (float, optional): 重ねる画像の透明度。デフォルトは 0.5。
      """
      # 背景画像の読み込み
      background = cv2.imread(background_path)
      background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

      # オーバーレイ画像の読み込み
      overlay = cv2.imread(overlay_path)
      overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

      # 画像を重ねる
      result = cv2.addWeighted(background, 1 - alpha, overlay, alpha, 0)

      # 結果の表示
      if show_preview:
        plt.imshow(result)
        plt.axis('off')
        plt.show()

      # 結果の保存
      result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
      cv2.imwrite(output_path, result)