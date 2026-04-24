from flask import Flask, render_template, request, redirect, url_for, send_file
import os, uuid, cv2, numpy as np, io
from ml.predict import predict_growth

app = Flask(__name__, template_folder='templates')
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index2.html', result=None)

@app.route('/predict2', methods=['POST'])
def predict2():
    file = request.files.get('file')
    if not file or file.filename == '':
        return render_template('index2.html', result=None, error='No file selected.')
    filename = str(uuid.uuid4()) + '_' + file.filename
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    try:
        result = predict_growth(filepath)
        return render_template('index2.html', result=result, filename=filename)
    except Exception as e:
        return render_template('index2.html', result=None, error=str(e))

@app.route('/gzones2/<filename>')
def gzones2(filename):
    return render_template('gzones2.html', filename=filename)

@app.route('/api/highlight_image/<filename>')
def highlight_image(filename):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return 'File not found', 404
    try:
        with open(filepath, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return 'Failed to decode image', 500
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([25, 20, 20]), np.array([95, 255, 255]))
        highlighted = img.copy()
        highlighted[mask > 0] = [0, 255, 0]
        _, buffer = cv2.imencode('.jpg', highlighted)
        buf = io.BytesIO(buffer.tobytes())
        buf.seek(0)
        return send_file(buf, mimetype='image/jpeg')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)
