from flask import Flask, render_template_string, request, jsonify
import tensorflow as tf
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)

# Load model
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def index():
    return render_template_string('''
<!DOCTYPE html>
<html>
<head>
    <title>手写数字识别</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; }
        #canvas { border: 2px solid #333; background: white; }
        button { padding: 10px 20px; font-size: 16px; margin: 10px; }
        #result { font-size: 24px; margin-top: 20px; }
    </style>
</head>
<body>
    <h1>✏️ 手写数字识别</h1>
    <canvas id="canvas" width="280" height="280"></canvas><br>
    <button onclick="clearCanvas()">清空</button>
    <button onclick="recognize()">识别</button>
    <div id="result"></div>
    
    <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        
        var isDrawing = false;
        
        canvas.addEventListener('mousedown', function(e) {
            isDrawing = true;
            ctx.beginPath();
            ctx.moveTo(e.offsetX, e.offsetY);
        });
        
        canvas.addEventListener('mousemove', function(e) {
            if (isDrawing) {
                ctx.lineTo(e.offsetX, e.offsetY);
                ctx.stroke();
            }
        });
        
        canvas.addEventListener('mouseup', function() {
            isDrawing = false;
        });
        
        function clearCanvas() {
            ctx.fillStyle = 'white';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').innerHTML = '';
        }
        
        function recognize() {
            var dataURL = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('result').innerHTML = 
                    '预测数字: ' + data.digit + ' (置信度: ' + data.confidence + '%)';
            });
        }
    </script>
</body>
</html>
''')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    image = Image.open(io.BytesIO(base64.b64decode(image_data))).convert('L')
    image = image.resize((28, 28))
    image = np.array(image).astype('float32') / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    prediction = model.predict(image, verbose=0)
    digit = int(np.argmax(prediction))
    confidence = float(prediction[0][digit] * 100)
    
    return jsonify({'digit': digit, 'confidence': round(confidence, 2)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)