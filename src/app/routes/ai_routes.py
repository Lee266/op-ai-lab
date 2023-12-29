from flask import Blueprint, jsonify
import timm
import torch
import os

ai_routes = Blueprint('ai_routes', __name__)

@ai_routes.route('/mnist/vit', methods=['GET'])
def predict():
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=10, img_size=224)
    current_directory = os.getcwd()
    print(f"{current_directory}hello")
    # 重みの読み込み（適切なパスを指定してください）
    weight_path = "./src/dev/result/weight/mnist-vit.pth"
    try:
        # ファイルから重みを読み込む
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    except Exception as e:
        return jsonify({'error': f'Failed to load the model weights: {str(e)}'}), 500
    dummy_input = torch.rand(1, 3, 224, 224)
    # 推論
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
        print(output)

    # 推論結果を JSON 形式で返す
    result = {'prediction': output.tolist()}  # 仮のレスポンス例

    return jsonify(result)

# 他にもエンドポイントがあればここに追加

# 例外ハンドリング
@ai_routes.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404