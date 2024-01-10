from flask import Blueprint, jsonify, request
import timm
import torch
import os
import logging
from PIL import Image
import torchvision.transforms as transforms

ai_routes = Blueprint('ai_routes', __name__)
logging.basicConfig(level=logging.DEBUG)
# 画像のアップロード先フォルダ
UPLOAD_FOLDER = './uploaded_images'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_image(model_type, weight_name):
    try:
        uploaded_file = request.files['image']
        
        # 画像をPNG形式で保存
        img_path = os.path.join(UPLOAD_FOLDER, f"{uploaded_file.filename.split('.')[0]}.png")
        uploaded_file.save(img_path)
        img = Image.open(uploaded_file)
        img.save(img_path, format='PNG')
        # 画像の前処理
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Grayscale(num_output_channels=3),
        ])
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)

        # モデルの作成と重みの読み込み
        model = timm.create_model(model_type, pretrained=False, num_classes=10, img_size=224)
        weight_path = f"./src/dev/result/weight/mnist-{weight_name}.pth"
        try:
            model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        except Exception as e:
            return jsonify({'error': f'Failed to load the model weights: {str(e)}'}), 500
        
        # 推論モードでの出力取得
        model.eval()
        with torch.no_grad():
            output = model(img_tensor)
        logging.debug(output)

        # ソフトマックス関数を適用し、パーセンテージに変換
        softmax_output = torch.nn.functional.softmax(output, dim=-1)
        percentages = (softmax_output * 100).round().tolist()[0]
        
        # 予測結果の取得
        _, indices = torch.topk(output, k=output.size(-1), dim=-1)
        predictions = [{'label': str(idx.item()), 'value': int(percentages[idx])} for idx in indices[0]]
        result = {'prediction': predictions}
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': f'Error during inference: {str(e)}'}), 500

@ai_routes.route('/mnist', methods=['GET'])
def predict_mnist():
    return jsonify({'message': 'This is a POST request to /mnist'})

@ai_routes.route('/mnist/vit', methods=['POST'])
def predict_vit():
    return predict_image('vit_tiny_patch16_224', 'vit')

@ai_routes.route('/mnist/swin', methods=['POST'])
def predict_swin():
    return predict_image('swin_tiny_patch4_window7_224', 'swin')

@ai_routes.route('/mnist/vgg', methods=['POST'])
def predict_vgg():
    return predict_image('vgg19', 'vgg19')

@ai_routes.route('/mnist/resnet', methods=['POST'])
def predict_resnet():
    return predict_image('resnet101', 'resnet')

# 例外ハンドリング
@ai_routes.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404