import time
import os
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import joblib  # もしくはpickleを使用することもできます
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import io  # ioモジュールをインポート


# アップロードされた画像を保存するディレクトリ（フルパス）
UPLOAD_FOLDER = os.path.abspath('uploads')  
# アップロードした画像の有効期限（秒）
IMAGE_EXPIRATION_TIME = 3600  # 1時間

app = Flask(__name__)
# モデルの読み込み
model = joblib.load('bananaDetect2.pkl')
# uploads フォルダの絶対パスを設定
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# モデルのデバイスを設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 1. 画像の前処理
def preprocess_image(image):
    # FileStorageをファイルとして保存
    temp_image_path = os.path.join(UPLOAD_FOLDER, 'temp_image.jpg')
    image.save(temp_image_path)
    # FileStorageをPIL Imageに変換
    image = Image.open(temp_image_path)
    transform = T.Compose([
        T.ToTensor(),  # 画像をテンソルに変換
    ])
    return transform(image).unsqueeze(0)  # バッチ次元を追加

# 2. モデルに画像を渡して推論
def predict(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)

    return predictions

# 3. バウンディングボックスを描画
def draw_boxes(image, predictions, threshold=0.5):
    draw = ImageDraw.Draw(image)
    best_score = 0.0
    best_box = None
    for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        if score > threshold and score > best_score:
            best_score = score
            best_box = box

    if best_box is not None:
        best_box = [round(i, 2) for i in best_box.tolist()]  # バウンディングボックスの座標を整数に変換
        draw.rectangle(best_box, outline="red", width=3)
    
    return image



# アップロードされた画像を処理して結果を表示するルート
@app.route('/', methods=['GET', 'POST'])
def upload_and_display_image():
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image:
                # 画像の前処理
                image_copy = image.stream.read()  # アップロードされた画像を読み込み
                image_stream = io.BytesIO(image_copy)  # バッファに読み込んだ画像を保存
                 # 画像をPIL Imageとして開く
                pil_image = Image.open(image_stream)

                image_tensor = preprocess_image(pil_image)
                # モデルに画像を渡して推論
                predictions = predict(model, image_tensor)
                # バウンディングボックスを描画
                result_image = draw_boxes(pil_image.copy(), predictions[0], threshold=0.5)
                # 画像を保存
                result_image_path = os.path.join(UPLOAD_FOLDER,'result.jpg')
                result_image.save(result_image_path)
                # 結果の画像を表示
                return render_template('index.html', image_path='uploads/result.jpg')

    # 画像がアップロードされていない場合やPOSTリクエストがない場合は、通常のページを表示
    return render_template('index.html', image_path=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # アップロードされた画像を表示
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def generate_unique_filename(filename):
    secure_name = secure_filename(filename)
    unique_filename = secure_name  # ここでは一意のファイル名として元のファイル名を使用
    return unique_filename

if __name__ == '__main__':
    app.run(debug=True)
