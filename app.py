# app.py

from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from werkzeug.utils import secure_filename
import shutil

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the Flask application
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your actual secret key
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max upload size: 16MB

# Define model classes in the same order as your model's output layer
model_classes = ['earphones', 'home_appliances', 'laptop', 'phone']

# Define a mapping from model class to brand
model_to_brand = {
    'earphones': 'Apple',
    'home_appliances': 'Samsung',
    'laptop': 'Google',
    'phone': 'LG'
}

# Define category directories
category_dirs = {
    'earphones': os.path.join(UPLOAD_FOLDER, 'earphones'),
    'home_appliances': os.path.join(UPLOAD_FOLDER, 'home_appliances'),
    'laptop': os.path.join(UPLOAD_FOLDER, 'laptop'),
    'phone': os.path.join(UPLOAD_FOLDER, 'phone')
}

# Ensure category directories exist
for dir_path in category_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Load the trained model
model_path = os.path.join('models', r'C:\Users\HP\phone_classifier\models\smartphone_model_classifier_with_brands.keras')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at path: {model_path}")
model = tf.keras.models.load_model(model_path)

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def classify_image(img_path):
    """Preprocess the image and classify it using the loaded model."""
    try:
        # Preprocess the image to match the input format of the model
        img = load_img(img_path, target_size=(224, 224))  # Ensure size is 224x224
        img_array = img_to_array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)

        # Get the predicted class index
        class_idx = np.argmax(predictions[0])

        # Map the predicted class index to the corresponding model class
        model_class = model_classes[class_idx]

        # Get the brand from the model class
        brand = model_to_brand.get(model_class, "Unknown brand")

        return brand, model_class
    except Exception as e:
        print(f"Error during classification: {e}")
        return None, None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)
        
        file = request.files['file']
        
        # If user does not select a file, return an error message
        if file.filename == '':
            flash('No file selected for uploading.')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # Save the uploaded file temporarily
                file.save(temp_path)
                
                # Classify the uploaded image
                brand, model_class = classify_image(temp_path)
                
                if brand and model_class:
                    # Define the destination directory based on classification
                    dest_dir = category_dirs.get(model_class)
                    
                    if dest_dir:
                        # Define the destination path
                        dest_path = os.path.join(dest_dir, filename)
                        
                        # Move the file to the destination directory
                        shutil.move(temp_path, dest_path)
                        
                        # Prepare the image URL for display
                        image_url = url_for('static', filename=f"uploads/{model_class}/{filename}")
                        
                        # Retrieve all images for gallery display
                        laptop_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'laptop'))
                        phone_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'phone'))
                        earphones_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'earphones'))
                        home_appliances_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'home_appliances'))
                        
                        return render_template('upload.html', 
                                               message=f'Image successfully uploaded and classified as {brand} - {model_class}.',
                                               image_url=image_url,
                                               brand=brand,
                                               model_class=model_class,
                                               laptop_images=laptop_images,
                                               phone_images=phone_images,
                                               earphones_images=earphones_images,
                                               home_appliances_images=home_appliances_images)
                    else:
                        # If the model_class is not recognized, remove the file
                        os.remove(temp_path)
                        flash('Classification resulted in an unknown category.')
                        return redirect(request.url)
                else:
                    # Remove the file if classification failed
                    os.remove(temp_path)
                    flash('Error in classification.')
                    return redirect(request.url)
            
            except Exception as e:
                # Remove the file in case of any exception
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                print(f"Error: {e}")
                flash(f"An error occurred while processing the file: {e}")
                return redirect(request.url)
        else:
            flash('Allowed file types are png, jpg, jpeg, gif.')
            return redirect(request.url)
    
    # For GET request, retrieve images from each category to display in the gallery
    laptop_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'laptop'))
    phone_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'phone'))
    earphones_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'earphones'))
    home_appliances_images = os.listdir(os.path.join(app.config['UPLOAD_FOLDER'], 'home_appliances'))
    
    return render_template('upload.html',
                           laptop_images=laptop_images,
                           phone_images=phone_images,
                           earphones_images=earphones_images,
                           home_appliances_images=home_appliances_images)

# Route to serve uploaded images is handled by Flask's static folder

if __name__ == '__main__':
    # Ensure 'uploads' directory exists within 'static'
    for category in model_classes:
        os.makedirs(os.path.join(UPLOAD_FOLDER, category), exist_ok=True)
    
    # Run the Flask app
    app.run(debug=True)
