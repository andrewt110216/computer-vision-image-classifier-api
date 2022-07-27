# Imports
from flask import Flask, jsonify, request
import io
import json
from PIL import Image
import torch
from torchvision import models
import torchvision.transforms as T

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# set pretrained computer vision model from pytorch
# possible options: resnet101, googlenet
model = models.googlenet(pretrained=True)
try:
    imagenet_index = json.load(open('imagenet_class_index.json'))
except:
    imagenet_index = json.load(open('project/imagenet_class_index.json'))

def allowed_file(filename: str) -> bool:
    """Check that the filename is in the set of allowed extensions"""
    if '.' in filename:
        extension = filename.split('.')[1].lower()
        return extension in ALLOWED_EXTENSIONS
    return False

# set return string for home page request
@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        with open('index.html') as f:
            html_index = f.read()
    except:
        with open('project/index.html') as f:
            html_index = f.read()
    return html_index

def image_transformation(image_bytes: bytes) -> torch.tensor:
    # create transformer
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_transformations = T.Compose([T.Resize(255), T.ToTensor(), normalize])

    # convert to PIL image object with only 3 channels (RGB)
    uploaded_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    return img_transformations(uploaded_image).unsqueeze(0)

def prediction(image_bytes: bytes) -> dict[dict]:
    tensor = image_transformation(image_bytes)

    model_output = model.forward(tensor)
    _, indices = torch.sort(model_output, descending=True)
    percentages = torch.nn.functional.softmax(model_output, dim=1)[0]
    top_5 = {}
    for i in range(5):
        idx = indices[0][i].item()
        class_id, class_name = imagenet_index[str(idx)]
        confidence = percentages[idx].item()
        top_5[i+1] = {'class_id': class_id,
                    'class name': class_name,
                    'confidence %': f'{confidence:.3%}'}
    return top_5

# handle inbound image
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']

        # if file exists and has allowed extension
        if file and allowed_file(file.filename):
            # convert file to bytes
            image_bytes = file.read()
            top_5 = prediction(image_bytes)
            return jsonify(top_5)

        return 'Error: was unable to process the image file through the model. Note that the file must be a JPEG.'
        
    elif request.method == 'GET':
        try:
            with open('predict.html') as f:
                html_predict = f.read()
        except:
            with open('project/predict.html') as f:
                html_predict = f.read()
        return html_predict

if __name__ == "__main__":
    app.run(debug=True)
