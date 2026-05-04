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
    <title>手写数字识别系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 40px 20px; 
            background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
            min-height: 100vh;
        }
        .main-header {
            text-align: center;
            margin-bottom: 40px;
        }
        .main-title {
            font-size: 36px;
            font-weight: 700;
            background: linear-gradient(135deg, #1e3a8a 0%, #7c3aed 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }
        .student-card {
            display: inline-flex;
            align-items: center;
            gap: 15px;
            background: white;
            padding: 12px 24px;
            border-radius: 30px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            font-size: 14px;
            color: #64748b;
        }
        .student-icon {
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #1e3a8a, #7c3aed);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
        }
        .tabs-container {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 30px;
        }
        .tab-btn {
            padding: 14px 32px;
            font-size: 16px;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: white;
            color: #64748b;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            position: relative;
            overflow: hidden;
        }
        .tab-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #1e3a8a, #7c3aed);
            transition: left 0.3s ease;
        }
        .tab-btn:hover::before, .tab-btn.active::before {
            left: 0;
        }
        .tab-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .tab-btn.active {
            background: linear-gradient(135deg, #1e3a8a, #7c3aed);
            color: white;
            box-shadow: 0 4px 20px rgba(30, 58, 138, 0.3);
        }
        .tab-content {
            display: none;
            animation: fadeIn 0.3s ease;
        }
        .tab-content.active {
            display: block;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .content-wrapper {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
        }
        .panel {
            background: white;
            border-radius: 20px;
            padding: 30px;
            box-shadow: 0 4px 25px rgba(0,0,0,0.06);
        }
        .panel-title {
            font-size: 20px;
            font-weight: 600;
            color: #1e293b;
            margin-bottom: 25px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .panel-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, #10b981, #059669);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 18px;
        }
        .upload-zone {
            border: 2px dashed #cbd5e1;
            border-radius: 16px;
            padding: 50px 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: #fafafa;
        }
        .upload-zone:hover {
            border-color: #7c3aed;
            background: #f5f3ff;
            transform: scale(1.02);
        }
        .upload-zone.dragover {
            border-color: #7c3aed;
            background: #f5f3ff;
            box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.1);
        }
        .upload-text {
            color: #64748b;
            font-size: 16px;
            margin-bottom: 10px;
        }
        .upload-hint {
            color: #94a3b8;
            font-size: 14px;
        }
        #file-input { display: none; }
        #uploaded-image {
            max-width: 100%;
            max-height: 250px;
            border-radius: 12px;
            margin-top: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .canvas-container {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: inset 0 2px 10px rgba(0,0,0,0.05);
        }
        #canvas {
            display: block;
            background: white;
            cursor: crosshair;
        }
        .btn-group {
            display: flex;
            gap: 12px;
            margin-top: 20px;
        }
        .btn {
            flex: 1;
            padding: 14px 24px;
            font-size: 15px;
            font-weight: 600;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .btn-primary {
            background: linear-gradient(135deg, #1e3a8a, #7c3aed);
            color: white;
            box-shadow: 0 4px 15px rgba(30, 58, 138, 0.3);
        }
        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(30, 58, 138, 0.4);
        }
        .btn-secondary {
            background: #f1f5f9;
            color: #64748b;
        }
        .btn-secondary:hover {
            background: #e2e8f0;
        }
        .result-section {
            text-align: center;
        }
        .prediction-box {
            width: 160px;
            height: 160px;
            margin: 0 auto 20px;
            background: linear-gradient(135deg, #1e3a8a, #7c3aed);
            border-radius: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 10px 40px rgba(30, 58, 138, 0.3); }
            50% { box-shadow: 0 15px 50px rgba(30, 58, 138, 0.4); }
        }
        .prediction-digit {
            font-size: 80px;
            font-weight: 800;
            color: white;
        }
        .confidence-text {
            font-size: 18px;
            color: #64748b;
            margin-bottom: 30px;
        }
        .confidence-value {
            font-weight: 700;
            color: #1e3a8a;
        }
        .top3-section {
            margin-bottom: 30px;
        }
        .top3-title {
            font-size: 16px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 15px;
        }
        .top3-list {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .top3-item {
            display: flex;
            align-items: center;
            padding: 12px 16px;
            background: #f8fafc;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .top3-item:hover {
            background: #f1f5f9;
            transform: translateX(5px);
        }
        .top3-rank {
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            font-size: 14px;
            margin-right: 15px;
        }
        .rank-1 { background: linear-gradient(135deg, #fbbf24, #f59e0b); color: white; }
        .rank-2 { background: linear-gradient(135deg, #9ca3af, #6b7280); color: white; }
        .rank-3 { background: linear-gradient(135deg, #d97706, #b45309); color: white; }
        .top3-digit {
            font-size: 24px;
            font-weight: 700;
            color: #1e293b;
            margin-right: auto;
        }
        .top3-confidence {
            font-size: 14px;
            color: #64748b;
        }
        .prob-section {
            margin-top: 20px;
        }
        .prob-title {
            font-size: 16px;
            font-weight: 600;
            color: #475569;
            margin-bottom: 15px;
        }
        .prob-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 10px;
        }
        .prob-item {
            text-align: center;
            padding: 12px 8px;
            background: #f8fafc;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        .prob-item:hover {
            background: #e0e7ff;
            transform: translateY(-3px);
        }
        .prob-digit {
            font-size: 18px;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 8px;
        }
        .prob-bar-container {
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin-bottom: 5px;
        }
        .prob-bar-fill {
            height: 100%;
            background: linear-gradient(90deg, #1e3a8a, #7c3aed);
            border-radius: 4px;
            transition: width 0.5s ease;
            min-width: 2px;
        }
        .prob-value {
            font-size: 12px;
            color: #64748b;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e2e8f0;
            color: #94a3b8;
            font-size: 14px;
        }
        .model-info {
            display: inline-flex;
            gap: 20px;
            margin-top: 10px;
        }
        .model-badge {
            padding: 6px 16px;
            background: #e0e7ff;
            color: #4338ca;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <header class="main-header">
        <h1 class="main-title">✏️ 手写数字识别系统</h1>
        <div class="student-card">
            <div class="student-icon">👤</div>
            <span>陈晶晶 | 学号：112304260112 | 班级：数据1231</span>
        </div>
    </header>
    
    <div class="tabs-container">
        <button class="tab-btn active" onclick="showTab('tab1')">
            📤 实验二：上传图片
        </button>
        <button class="tab-btn" onclick="showTab('tab2')">
            ✏️ 实验三：手写画板
        </button>
    </div>
    
    <!-- 实验二：上传图片 -->
    <div id="tab1" class="tab-content active">
        <div class="content-wrapper">
            <div class="panel">
                <h2 class="panel-title">
                    <span class="panel-icon">📁</span>
                    上传手写数字图片
                </h2>
                <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()" ondragover="event.preventDefault()" ondrop="handleDrop(event)">
                    <div class="upload-text">点击或拖拽图片到此处</div>
                    <div class="upload-hint">支持 JPG、PNG 等格式</div>
                </div>
                <img id="uploaded-image" style="display:none">
                <input type="file" id="file-input" accept="image/*" style="display:none" onchange="handleFile()">
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="recognizeImage()">🔍 识别图片</button>
                    <button class="btn btn-secondary" onclick="clearUpload()">🗑️ 清空</button>
                </div>
            </div>
            
            <div class="panel">
                <h2 class="panel-title">
                    <span class="panel-icon">📊</span>
                    识别结果
                </h2>
                <div class="result-section" id="result1">
                    <div class="prediction-box">
                        <span class="prediction-digit">?</span>
                    </div>
                    <div class="confidence-text">置信度：<span class="confidence-value">0%</span></div>
                    
                    <div class="top3-section">
                        <div class="top3-title">🏆 Top-3 预测</div>
                        <div class="top3-list" id="top3-list1">
                            <div class="top3-item"><div class="top3-rank rank-1">1</div><span class="top3-digit">-</span><span class="top3-confidence">-</span></div>
                            <div class="top3-item"><div class="top3-rank rank-2">2</div><span class="top3-digit">-</span><span class="top3-confidence">-</span></div>
                            <div class="top3-item"><div class="top3-rank rank-3">3</div><span class="top3-digit">-</span><span class="top3-confidence">-</span></div>
                        </div>
                    </div>
                    
                    <div class="prob-section">
                        <div class="prob-title">📈 概率分布</div>
                        <div class="prob-grid" id="prob-grid1">
                            <div class="prob-item"><div class="prob-digit">0</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">1</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">2</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">3</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">4</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">5</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">6</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">7</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">8</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">9</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <!-- 实验三：手写画板 -->
    <div id="tab2" class="tab-content">
        <div class="content-wrapper">
            <div class="panel">
                <h2 class="panel-title">
                    <span class="panel-icon">✏️</span>
                    在画板上书写数字
                </h2>
                <div class="canvas-container">
                    <canvas id="canvas" width="350" height="350"></canvas>
                </div>
                <div class="btn-group">
                    <button class="btn btn-primary" onclick="recognizeSketch()">🔍 识别手写</button>
                    <button class="btn btn-secondary" onclick="clearCanvas()">🗑️ 清空画板</button>
                </div>
            </div>
            
            <div class="panel">
                <h2 class="panel-title">
                    <span class="panel-icon">🎯</span>
                    识别结果
                </h2>
                <div class="result-section" id="result2">
                    <div class="prediction-box">
                        <span class="prediction-digit">?</span>
                    </div>
                    <div class="confidence-text">置信度：<span class="confidence-value">0%</span></div>
                    
                    <div class="top3-section">
                        <div class="top3-title">🏆 Top-3 预测</div>
                        <div class="top3-list" id="top3-list2">
                            <div class="top3-item"><div class="top3-rank rank-1">1</div><span class="top3-digit">-</span><span class="top3-confidence">-</span></div>
                            <div class="top3-item"><div class="top3-rank rank-2">2</div><span class="top3-digit">-</span><span class="top3-confidence">-</span></div>
                            <div class="top3-item"><div class="top3-rank rank-3">3</div><span class="top3-digit">-</span><span class="top3-confidence">-</span></div>
                        </div>
                    </div>
                    
                    <div class="prob-section">
                        <div class="prob-title">📈 概率分布</div>
                        <div class="prob-grid" id="prob-grid2">
                            <div class="prob-item"><div class="prob-digit">0</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">1</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">2</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">3</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">4</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">5</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">6</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">7</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">8</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                            <div class="prob-item"><div class="prob-digit">9</div><div class="prob-bar-container"><div class="prob-bar-fill"></div></div><div class="prob-value">0%</div></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <footer class="footer">
        <div>基于卷积神经网络 (CNN) 实现</div>
        <div class="model-info">
            <span class="model-badge">训练准确率：99.09%</span>
            <span class="model-badge">验证准确率：99.04%</span>
        </div>
    </footer>
    
    <script>
        var canvas = document.getElementById('canvas');
        var ctx = canvas.getContext('2d');
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#1e293b';
        ctx.lineWidth = 18;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        
        var isDrawing = false;
        var lastX = 0;
        var lastY = 0;
        
        canvas.addEventListener('mousedown', function(e) {
            isDrawing = true;
            [lastX, lastY] = [e.offsetX, e.offsetY];
        });
        
        canvas.addEventListener('mousemove', function(e) {
            if (isDrawing) {
                ctx.beginPath();
                ctx.moveTo(lastX, lastY);
                ctx.lineTo(e.offsetX, e.offsetY);
                ctx.stroke();
                [lastX, lastY] = [e.offsetX, e.offsetY];
            }
        });
        
        canvas.addEventListener('mouseup', function() { isDrawing = false; });
        canvas.addEventListener('mouseout', function() { isDrawing = false; });
        
        function showTab(tabId) {
            document.querySelectorAll('.tab-btn').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            event.target.classList.add('active');
            document.getElementById(tabId).classList.add('active');
        }
        
        function handleFile() {
            var file = document.getElementById('file-input').files[0];
            if (file) {
                var reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('uploaded-image').src = e.target.result;
                    document.getElementById('uploaded-image').style.display = 'block';
                    document.getElementById('upload-zone').style.display = 'none';
                }
                reader.readAsDataURL(file);
            }
        }
        
        function handleDrop(e) {
            e.preventDefault();
            var file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                var reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('uploaded-image').src = e.target.result;
                    document.getElementById('uploaded-image').style.display = 'block';
                    document.getElementById('upload-zone').style.display = 'none';
                }
                reader.readAsDataURL(file);
            }
        }
        
        function recognizeImage() {
            var img = document.getElementById('uploaded-image');
            if (!img.src) {
                alert('请先上传图片');
                return;
            }
            var canvasTemp = document.createElement('canvas');
            var ctxTemp = canvasTemp.getContext('2d');
            canvasTemp.width = 28;
            canvasTemp.height = 28;
            ctxTemp.drawImage(img, 0, 0, 28, 28);
            var dataURL = canvasTemp.toDataURL('image/png');
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                showResults('1', data);
            });
        }
        
        function recognizeSketch() {
            var dataURL = canvas.toDataURL('image/png');
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: dataURL })
            })
            .then(response => response.json())
            .then(data => {
                showResults('2', data);
            });
        }
        
        function showResults(tabNum, data) {
            document.querySelector('#result' + tabNum + ' .prediction-digit').textContent = data.digit;
            document.querySelector('#result' + tabNum + ' .confidence-value').textContent = data.confidence + '%';
            
            var top3Items = document.querySelectorAll('#top3-list' + tabNum + ' .top3-item');
            top3Items.forEach((item, index) => {
                item.querySelector('.top3-digit').textContent = data.top3[index].digit;
                item.querySelector('.top3-confidence').textContent = data.top3[index].confidence.toFixed(2) + '%';
            });
            
            var probItems = document.querySelectorAll('#prob-grid' + tabNum + ' .prob-item');
            probItems.forEach((item, index) => {
                var prob = data.probabilities[index];
                item.querySelector('.prob-bar-fill').style.width = (prob * 100) + '%';
                item.querySelector('.prob-value').textContent = (prob * 100).toFixed(1) + '%';
            });
        }
        
        function clearUpload() {
            document.getElementById('file-input').value = '';
            document.getElementById('uploaded-image').src = '';
            document.getElementById('uploaded-image').style.display = 'none';
            document.getElementById('upload-zone').style.display = 'block';
            resetResults('1');
        }
        
        function clearCanvas() {
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            resetResults('2');
        }
        
        function resetResults(tabNum) {
            document.querySelector('#result' + tabNum + ' .prediction-digit').textContent = '?';
            document.querySelector('#result' + tabNum + ' .confidence-value').textContent = '0%';
            
            var top3Items = document.querySelectorAll('#top3-list' + tabNum + ' .top3-item');
            top3Items.forEach(item => {
                item.querySelector('.top3-digit').textContent = '-';
                item.querySelector('.top3-confidence').textContent = '-';
            });
            
            var probItems = document.querySelectorAll('#prob-grid' + tabNum + ' .prob-item');
            probItems.forEach(item => {
                item.querySelector('.prob-bar-fill').style.width = '0%';
                item.querySelector('.prob-value').textContent = '0%';
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
    
    # 转换为numpy数组
    img_array = np.array(image)
    
    # 反转颜色：手写是黑色在白色上，MNIST是白色在黑色上
    img_array = 255 - img_array
    
    # 二值化处理，提高对比度
    threshold = 128
    img_array[img_array < threshold] = 0
    img_array[img_array >= threshold] = 255
    
    # 找到数字的边界框
    rows = np.any(img_array > 0, axis=1)
    cols = np.any(img_array > 0, axis=0)
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        
        # 裁剪到边界框
        img_array = img_array[ymin:ymax+1, xmin:xmax+1]
        
        # 计算缩放比例，保持宽高比
        height, width = img_array.shape
        max_dim = max(height, width)
        scale = 20 / max_dim
        
        # 缩放
        new_height, new_width = int(height * scale), int(width * scale)
        img_array = np.array(Image.fromarray(img_array).resize((new_width, new_height)))
    
    # 创建28x28的画布，居中放置数字
    final_img = np.zeros((28, 28), dtype=np.float32)
    h, w = img_array.shape
    
    # 计算居中位置
    y_offset = (28 - h) // 2
    x_offset = (28 - w) // 2
    
    # 放置数字
    final_img[y_offset:y_offset+h, x_offset:x_offset+w] = img_array
    
    # 归一化
    final_img = final_img / 255.0
    
    # 添加batch和channel维度
    image = np.expand_dims(final_img, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    prediction = model.predict(image, verbose=0)[0]
    digit = int(np.argmax(prediction))
    confidence = float(prediction[digit] * 100)
    
    top3_indices = np.argsort(prediction)[::-1][:3]
    top3 = [{'digit': int(i), 'confidence': float(prediction[i] * 100)} for i in top3_indices]
    
    return jsonify({
        'digit': digit,
        'confidence': round(confidence, 2),
        'top3': top3,
        'probabilities': prediction.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)