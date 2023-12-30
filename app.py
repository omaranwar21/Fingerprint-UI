from flask import Flask, render_template, request, send_file
from PIL import Image
from main import *
import classifier_fingerprint as FR
import hand_classifier as HC
import name_classifier as NC

app = Flask(__name__)

 
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("prediction.html")

@app.route('/model')
def model():
    return render_template('model.html') 
 
@app.route("/submit", methods = ['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        img = request.files['my_image']
        img_path = "static/" + img.filename	
        img.save(img_path)
        gender_pred = FR.prediction(img_path)
        hand_pred = HC.Hand_prediction(img_path)
        name_pred = NC.Name_prediction(img_path)
    return render_template('prediction.html', prediction = gender_pred,hand_pred=hand_pred,name_pred=name_pred, img_path = img_path) 

@app.route('/processing', methods=['POST'])
def processing():
        # Get the uploaded image from the request
    uploaded_file = request.files['image']

# Open an existing image file
# existing_image = Image.open("existing_image.jpg")

# Save the image with a new name or format
    uploaded_file.save("input.jpg")  # You can change the file format and file name as needed

    # print(uploaded_file)

    # Process the image (replace this with your actual processing logic)
    Run("input.jpg")
    minext("Application_enhance.jpg")
    # return
    return send_file("extracted_image.jpg", mimetype='image/jpeg')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

# Add more routes for other HTML files if needed

if __name__ == '__main__':
    app.run(debug=True)
