from flask import Flask, render_template, request, jsonify, send_file
import os
import uuid
import cv2
import numpy as np
import io
from ml.predict import predict_growth

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # Our existing ML logic
            result = predict_growth(filepath)
            
            # DO NOT remove the file immediately so we can highlight it later
            # os.remove(filepath)
            
            return jsonify({
                'success': True,
                'growth': result['label'],
                'probabilities': result['probabilities'],
                'filename': filename
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': str(e)}), 500

@app.route('/gzones/<filename>')
def highlight_page(filename):
    return render_template('gzones.html', filename=filename)

@app.route('/api/highlight_image/<filename>')
def highlight_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File not found", 404

    try:
        # Robust image reading (supports unicode paths & strange formats)
        with open(filepath, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if img is None:
            return "Failed to decode uploaded image.", 500

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Broadened range for green color to catch dark/pale vegetation
        lower_green = np.array([25, 20, 20])
        upper_green = np.array([95, 255, 255])
        
        # Threshold the HSV image
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Highlight green areas in the original image
        highlighted = img.copy()
        highlighted[mask > 0] = [0, 255, 0] # BGR for pure Green
        
        # Encode and send
        _, buffer = cv2.imencode('.jpg', highlighted)
        io_buf = io.BytesIO(buffer.tobytes())
        io_buf.seek(0)
        return send_file(io_buf, mimetype='image/jpeg', as_attachment=False, download_name="highlight.jpg")
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
