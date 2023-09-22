#import all necessary libraries
import os
from flask import Flask, render_template, request, send_from_directory,jsonify
from werkzeug.utils import secure_filename
import uuid
import joblib  
from PIL import Image, ImageDraw
import torch
import torchvision.transforms as T
import io  
from flask_cors import CORS  

# set Flask App
app = Flask(__name__)
# path to save all images.
UPLOAD_FOLDER = os.path.abspath('static/uploads')  
# get models 
model_d= joblib.load('bananaDetect3.pkl')
model_c=joblib.load('bananaClassify4.pkl')
# set absolute path
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# set device of model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# validate CORS
CORS(app)
# static folder
static_folder = 'static'




# 1. depicting boundinig box func
def draw_boxes(image, best_box,outline_color="red", outline_width=3, outer_color = 'rgba(211, 211, 211, 128)'):
    draw = ImageDraw.Draw(image)
    if best_box is not None:
        best_box = [round(i, 2) for i in best_box.tolist()]  # interept bounding box into Integer.
        draw.rectangle(best_box, outline="red", width=3)

        # crop bounding image
        left, top, right, bottom = map(int, best_box)
        cropped_image = image.crop((left, top, right, bottom))

        cropped_image.save('static/uploads/cropped_result.jpg')  # save crop image
    return image



# 2. preprocessing of image(before crop)
def preprocess_image(image):
    transform_d = T.Compose([
        T.ToTensor(),  # transfer image into tensor
    ])
    return transform_d(image).unsqueeze(0)  




# 3. predict bounding box from image
def detect_bounding_box(model, image_tensor, threshold=0.5):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model(image_tensor)

    # find high score bounding box. and return only one.
    best_score = 0.0
    best_box = None
    for score, label, box in zip(predictions[0]['scores'], predictions[0]['labels'], predictions[0]['boxes']):
        if score > threshold and score > best_score:
            best_score = score
            best_box = box

    return best_box



#4. preprocessing of image（after crop）
def preprocess_cropped_image(image):
    transform = T.Compose([
        T.Resize((224, 224)),  # resize of image
        T.RandomHorizontalFlip(),  # rondom inversion
        T.RandomRotation(10),  # random rotation
        T.ToTensor(),  # trasnfert image into tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])
    return transform(image).unsqueeze(0) 

# 5. predict banana status judge.
def classify_banana(model, image):    
    # input image to model, and get prediction.
    with torch.no_grad():
        outputs = model(image)
    _, predicted_class = outputs.max(1)
    
    return predicted_class.item()



# landing page
@app.route('/banana', methods=['GET'])
def upload_and_display_image():
    # default page
    return render_template('index.html', image_path=None)


# Image Upload & Object Detect &Predict Status POST API
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' in request.files:
        image = request.files['image']
        if image:
            # preprocessing
            image_copy = image.stream.read()  # read uploaded image
            image_stream = io.BytesIO(image_copy)
            # open image sa PIL
            pil_image = Image.open(image_stream)

            # save original image as temp_image.jpg
            temp_image_path = os.path.join(UPLOAD_FOLDER, 'temp_image.JPG')
            pil_image.save(temp_image_path)


            image_tensor = preprocess_image(pil_image)
            # input image into model and get bounding box.
            best_box = detect_bounding_box(model_d, image_tensor)
            # depict bounding box
            result_image = draw_boxes(pil_image.copy(), best_box)
            # save image as result.jpg
            result_image_path = os.path.join(UPLOAD_FOLDER,'result.jpg')
            result_image.save(result_image_path)


            # open temp_image.jpg
            image_path1 = os.path.join(UPLOAD_FOLDER, 'temp_image.JPG')
            pil_image1 = Image.open(image_path1)
            # detect bounding box.
            image_tensor1 = preprocess_image(pil_image1)
            bounding_box1 = detect_bounding_box(model_d, image_tensor1)
        
            if bounding_box1 is not None:
                
                x1, y1, x2, y2 = map(int, bounding_box1.tolist())
                cropped_image = pil_image.crop((x1, y1, x2, y2))
                        
                # apply preprocessing
                cropped_image_tensor = preprocess_cropped_image(cropped_image)
                        
                # predict status
                predicted_class = classify_banana(model_c, cropped_image_tensor)
                        
            # return image_path & predict as json.
            data = {
                'image_path': '/uploads/result.jpg',
                'predict': predicted_class
            }
            return jsonify(data)

    return jsonify({'error': 'No image uploaded'}), 400

# /uploads/<filename>
#画像にアクセスして表示するにはこの設定が必要。
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(static_folder, f'uploads/{filename}')


# run app
if __name__ == '__main__':
    app.run(debug=True)
