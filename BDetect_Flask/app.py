#import all necessary libraries
import os
from flask import Flask, render_template, request, send_from_directory,jsonify
from werkzeug.utils import secure_filename
import uuid
import joblib  # もしくはpickleを使用することもできます
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import io  # ioモジュールをインポート
from flask_cors import CORS  # CORSをインポート

# ここでFlaskアプリを作成
app = Flask(__name__)
# アップロードされた画像を保存するディレクトリ（フルパス）
UPLOAD_FOLDER = os.path.abspath('static/uploads')  
# モデルの読み込み
model = joblib.load('bananaDetect2.pkl')
# uploads フォルダの絶対パスを設定
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# モデルのデバイスを設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CORSを有効にする
CORS(app)
# static フォルダを提供フォルダとして指定します
static_folder = 'static'


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
def draw_boxes(image, predictions, threshold=0.5,outline_color="red", outline_width=3, outer_color = 'rgba(211, 211, 211, 128)'):
    draw = ImageDraw.Draw(image)
    best_score = 0.0
    best_box = None
    for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
        if score > threshold and score > best_score: #予測結果が複数出た場合、最もスコアが高いものだけを使う
            best_score = score
            best_box = box

    if best_box is not None:
        best_box = [round(i, 2) for i in best_box.tolist()]  # バウンディングボックスの座標を整数に変換
        draw.rectangle(best_box, outline="red", width=3)

        # バウンディングボックスで切り取り
        left, top, right, bottom = map(int, best_box)
        cropped_image = image.crop((left, top, right, bottom))

        cropped_image.save('static/uploads/cropped_result.jpg')  # 切り取り画像を別のファイルに保存
    return image



# ランディングページ
@app.route('/banana', methods=['GET'])
def upload_and_display_image():
    # 通常のページを表示
    return render_template('index.html', image_path=None)


# 画像アップロード&Object Detect POST API
@app.route('/upload', methods=['POST'])
def upload_image():
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


            #これより下に、predictBrixModelにより糖度の予測を行う。
            #画像をboundigboxで切り取って保存。それを前処理してモデルに渡してpredictされる数値をdataに格納して返す。

            data = {
                'image_path': '/uploads/result.jpg',
                'predict_brix': 8
            }
            # dataをjsonにして返す
            return jsonify(data)


    # 画像がアップロードされていない場合やPOSTリクエストがない場合は、通常のページを表示
    return jsonify({'error': 'No image uploaded'}), 400

# /uploads/<filename> へのアクセスでアップロードされたファイルを返すルート
#画像にアクセスして表示するにはこの設定が必要。
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(static_folder, f'uploads/{filename}')


# appを実行する
if __name__ == '__main__':
    app.run(debug=True)
