# import time
# import os
# from flask import Flask, render_template, request, send_from_directory
# from werkzeug.utils import secure_filename
# import uuid
# import joblib  # もしくはpickleを使用することもできます
# from PIL import Image, ImageDraw
# import torch
# import torchvision.transforms as T
# import io  # ioモジュールをインポート


# # アップロードされた画像を保存するディレクトリ（フルパス）
# UPLOAD_FOLDER = os.path.abspath('uploads')  
# # アップロードした画像の有効期限（秒）
# IMAGE_EXPIRATION_TIME = 3600  # 1時間

# app = Flask(__name__)
# # モデルの読み込み
# model = joblib.load('bananaDetect2.pkl')
# # uploads フォルダの絶対パスを設定
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# # モデルのデバイスを設定
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # 1. 画像の前処理
# def preprocess_image(image):
#     # FileStorageをファイルとして保存
#     temp_image_path = os.path.join(UPLOAD_FOLDER, 'temp_image.jpg')
#     image.save(temp_image_path)
#     # FileStorageをPIL Imageに変換
#     image = Image.open(temp_image_path)
#     transform = T.Compose([
#         T.ToTensor(),  # 画像をテンソルに変換
#     ])
#     return transform(image).unsqueeze(0)  # バッチ次元を追加

# # 2. モデルに画像を渡して推論
# def predict(model, image_tensor):
#     model.eval()
#     with torch.no_grad():
#         image_tensor = image_tensor.to(device)
#         predictions = model(image_tensor)

#     return predictions

# # 3. バウンディングボックスを描画
# def draw_boxes(image, predictions, threshold=0.5,outline_color="red", outline_width=3, outer_color = 'rgba(211, 211, 211, 128)'):
#     draw = ImageDraw.Draw(image)
#     best_score = 0.0
#     best_box = None
#     for score, label, box in zip(predictions['scores'], predictions['labels'], predictions['boxes']):
#         if score > threshold and score > best_score: #予測結果が複数出た場合、最もスコアが高いものだけを使う
#             best_score = score
#             best_box = box

#     if best_box is not None:
#         best_box = [round(i, 2) for i in best_box.tolist()]  # バウンディングボックスの座標を整数に変換
#         draw.rectangle(best_box, outline="red", width=3)

#         # # バウンディングボックスの座標を取得
#         # x, y, x_max, y_max = best_box.tolist()
#         # # バウンディングボックスのサイズを計算
#         # width = x_max - x
#         # height = y_max - y

#         # # # 外側の四角形を描画（内部は塗りつぶさない）
#         # # outer_x = 0  # 外側の四角形の左上隅のx座標
#         # # outer_y = 0  # 外側の四角形の左上隅のy座標
#         # # outer_x_max = image.width  # 外側の四角形の右下隅のx座標
#         # # outer_y_max = image.height  # 外側の四角形の右下隅のy座標

#         # # # 外側の四角形を描画（半透明の色を使用）
#         # # draw.rectangle([outer_x, outer_y, x, outer_y_max], fill=outer_color)  # 左側の部分を塗りつぶし
#         # # draw.rectangle([x_max, outer_y, outer_x_max, outer_y_max], fill=outer_color)  # 右側の部分を塗りつぶし
#         # # draw.rectangle([x, outer_y, x_max, y], fill=outer_color)  # 上側の部分を塗りつぶし
#         # # draw.rectangle([x, y_max, x_max, outer_y_max], fill=outer_color)  # 下側の部分を塗りつぶし

#         # # 内側の四角形を描画（内部は塗りつぶし）
#         # draw.rectangle([x, y, x_max, y_max], outline=outline_color, width=outline_width)

#     return image



# # アップロードされた画像を処理して結果を表示するルート
# @app.route('/', methods=['GET', 'POST'])
# def upload_and_display_image():
#     if request.method == 'POST':
#         if 'image' in request.files:
#             image = request.files['image']
#             if image:
#                 # 画像の前処理
#                 image_copy = image.stream.read()  # アップロードされた画像を読み込み
#                 image_stream = io.BytesIO(image_copy)  # バッファに読み込んだ画像を保存
#                  # 画像をPIL Imageとして開く
#                 pil_image = Image.open(image_stream)

#                 image_tensor = preprocess_image(pil_image)
#                 # モデルに画像を渡して推論
#                 predictions = predict(model, image_tensor)
#                 # バウンディングボックスを描画
#                 result_image = draw_boxes(pil_image.copy(), predictions[0], threshold=0.5)
#                 # 画像を保存
#                 result_image_path = os.path.join(UPLOAD_FOLDER,'result.jpg')
#                 result_image.save(result_image_path)
#                 # 結果の画像を表示
#                 return render_template('index.html', image_path='uploads/result.jpg')

#     # 画像がアップロードされていない場合やPOSTリクエストがない場合は、通常のページを表示
#     return render_template('index.html', image_path=None)

# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     # アップロードされた画像を表示
#     return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# def generate_unique_filename(filename):
#     secure_name = secure_filename(filename)
#     unique_filename = secure_name  # ここでは一意のファイル名として元のファイル名を使用
#     return unique_filename

# if __name__ == '__main__':
#     app.run(debug=True)


#API POSTの場合
import time
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


# アップロードされた画像を保存するディレクトリ（フルパス）
UPLOAD_FOLDER = os.path.abspath('static/uploads')  
# アップロードした画像の有効期限（秒）
IMAGE_EXPIRATION_TIME = 3600  # 1時間

app = Flask(__name__)
# モデルの読み込み
model = joblib.load('bananaDetect2.pkl')
# uploads フォルダの絶対パスを設定
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# モデルのデバイスを設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CORSを有効にする
CORS(app)


# ここからは、バックエンド
# 画像の編集と保存の関数
def edit_and_save_image(image):
    # ここに画像の編集処理を追加
    # 例: 画像に矩形を描画
    draw = ImageDraw.Draw(image)
    draw.rectangle([(50, 50), (200, 200)], outline="red", width=5)

    # 保存
    edited_image_path = os.path.join(UPLOAD_FOLDER, 'edited_image.jpg')
    image.save(edited_image_path)



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


# ここの部分を変えたい
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
            # 保存した画像のパスを返す
            return '/uploads/result.jpg'
            # return 'result.jpg'

    # 画像がアップロードされていない場合やPOSTリクエストがない場合は、通常のページを表示
    return jsonify({'error': 'No image uploaded'}), 400



# static フォルダを提供フォルダとして指定します
static_folder = 'static'
# /uploads/<filename> へのアクセスでアップロードされたファイルを返すルート
#画像にアクセスして表示するにはこの設定が必要。
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(static_folder, f'uploads/{filename}')


if __name__ == '__main__':
    app.run(debug=True)
