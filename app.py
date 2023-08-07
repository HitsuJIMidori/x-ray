from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import os
from PIL import Image
import numpy as np
from werkzeug.utils import secure_filename

model_file = "VGG19 02.h5"
model = load_model(model_file)

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def makePredictions(path):
    '''
    Method to predict if the uploaded image is healthy or pneumonic
    '''
    img = Image.open(path)
    img_d = img.resize((224, 224))
    if len(np.array(img_d).shape) < 3:
        rgbimg = Image.new("RGB", img_d.size)
        rgbimg.paste(img_d)
    else:
        rgbimg = img_d
    rgbimg = np.array(rgbimg, dtype=np.float64)
    rgbimg = rgbimg.reshape((1, 224, 224, 3))
    predictions = model.predict(rgbimg)
    a = int(np.argmax(predictions))
    if a == 1:
        a = "pneumonic"
    else:
        a = "healthy"
    return a
@app.route('/page')
def page():
    return render_template('page.html')



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'img' not in request.files:
            return render_template('home.html', filename="unnamed.png", message="Please upload a file")
        
        f = request.files['img']
        filename = secure_filename(f.filename)
        
        if f.filename == '':
            return render_template('home.html', filename="unnamed.png", message="No file selected")
        
        if not ('jpeg' in f.filename or 'png' in f.filename or 'jpg' in f.filename):
            return render_template('home.html', filename="unnamed.png", message="Please upload an image with .png or .jpg/.jpeg extension")
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(file_path)
        
        predictions = makePredictions(file_path)
        return render_template('home.html', filename=f.filename, message=predictions, show=True)
    
    return render_template('home.html', filename='unnamed.png')

if __name__ == "_main_":
    app.run(debug=True)
    
   