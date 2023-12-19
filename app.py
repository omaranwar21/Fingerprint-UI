from flask import Flask, render_template, request, jsonify
from PIL import Image
# from main import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  

@app.route('/model')
def model():
    return render_template('model.html') 
 
@app.route('/prediction')
def prediction():
    return render_template('prediction.html') 

@app.route('/processing', methods=['POST'])
def processing():
        # Get the uploaded image from the request
    uploaded_file = request.files['image']

# Open an existing image file
# existing_image = Image.open("existing_image.jpg")

# Save the image with a new name or format
    uploaded_file.save("new_image.png")  # You can change the file format and file name as needed

    # print(uploaded_file)

    # Process the image (replace this with your actual processing logic)
    # Run("F_L3C.jpg")

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

# Add more routes for other HTML files if needed

if __name__ == '__main__':
    app.run(debug=True)
