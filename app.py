import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 加载模型
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    if image is None:
        return None
    
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    
    image = 255 - image
    threshold = 128
    image[image < threshold] = 0
    image[image >= threshold] = 255
    
    rows = np.any(image > 0, axis=1)
    cols = np.any(image > 0, axis=0)
    
    if np.any(rows) and np.any(cols):
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        image = image[ymin:ymax+1, xmin:xmax+1]
        
        height, width = image.shape
        max_dim = max(height, width)
        scale = 20 / max_dim
        new_height, new_width = int(height * scale), int(width * scale)
        image = np.array(Image.fromarray(image.astype(np.uint8)).resize((new_width, new_height)))
    
    final_img = np.zeros((28, 28), dtype=np.float32)
    h, w = image.shape
    y_offset = (28 - h) // 2
    x_offset = (28 - w) // 2
    final_img[y_offset:y_offset+h, x_offset:x_offset+w] = image
    
    final_img = final_img / 255.0
    final_img = np.expand_dims(final_img, axis=0)
    final_img = np.expand_dims(final_img, axis=-1)
    
    return final_img

def predict(image):
    processed_image = preprocess_image(image)
    if processed_image is None:
        return "?", 0.0, [], [0.1]*10
    
    prediction = model.predict(processed_image, verbose=0)[0]
    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100
    
    top3_indices = np.argsort(prediction)[::-1][:3]
    top3_results = [{"数字": int(i), "置信度(%)": float(prediction[i]*100)} for i in top3_indices]
    
    return str(digit), round(confidence, 2), top3_results, prediction.tolist()

with gr.Blocks(title="手写数字识别系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✏️ 手写数字识别系统")
    gr.Markdown("### 陈晶晶 | 学号：112304260112 | 班级：数据1231")
    
    with gr.Tabs():
        with gr.Tab("📤 实验二：上传图片"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 上传手写数字图片")
                    image_input = gr.Image(label="上传图片", type="numpy")
                    with gr.Row():
                        recognize_btn = gr.Button("🔍 识别", variant="primary")
                        clear_btn = gr.Button("🗑️ 清空")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 预测结果")
                    result_digit = gr.Textbox(label="预测数字", scale=2, placeholder="?")
                    result_confidence = gr.Textbox(label="置信度")
                    top3_table = gr.DataFrame(headers=["排名", "数字", "置信度(%)"], label="Top-3 预测")
                    probability_bar = gr.BarPlot(x=[0,1,2,3,4,5,6,7,8,9], y=[10]*10, label="概率分布")
        
        with gr.Tab("✏️ 实验三：手写画板"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 在画板上书写数字")
                    sketchpad = gr.Sketchpad(label="手写画板", brush_radius=8)
                    with gr.Row():
                        predict_btn = gr.Button("🔍 识别", variant="primary")
                        erase_btn = gr.Button("🗑️ 清空")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 预测结果")
                    sketch_result_digit = gr.Textbox(label="预测数字", scale=2, placeholder="?")
                    sketch_result_confidence = gr.Textbox(label="置信度")
                    sketch_top3_table = gr.DataFrame(headers=["排名", "数字", "置信度(%)"], label="Top-3 预测")
                    sketch_probability_bar = gr.BarPlot(x=[0,1,2,3,4,5,6,7,8,9], y=[10]*10, label="概率分布")
    
    def handle_upload_recognize(image):
        if image is None:
            return "?", 0.0, [], gr.BarPlot.update(y=[10]*10)
        digit, confidence, top3, probabilities = predict(image)
        top3_with_rank = [[i+1, item["数字"], round(item["置信度(%)"], 2)] for i, item in enumerate(top3)]
        return digit, f"{confidence}%", top3_with_rank, gr.BarPlot.update(y=probabilities)
    
    def handle_sketch_predict(image):
        if image is None:
            return "?", 0.0, [], gr.BarPlot.update(y=[10]*10)
        digit, confidence, top3, probabilities = predict(image)
        top3_with_rank = [[i+1, item["数字"], round(item["置信度(%)"], 2)] for i, item in enumerate(top3)]
        return digit, f"{confidence}%", top3_with_rank, gr.BarPlot.update(y=probabilities)
    
    recognize_btn.click(handle_upload_recognize, inputs=image_input, outputs=[result_digit, result_confidence, top3_table, probability_bar])
    clear_btn.click(lambda: (None, "?", 0.0, [], gr.BarPlot.update(y=[10]*10)), outputs=[image_input, result_digit, result_confidence, top3_table, probability_bar])
    predict_btn.click(handle_sketch_predict, inputs=sketchpad, outputs=[sketch_result_digit, sketch_result_confidence, sketch_top3_table, sketch_probability_bar])
    erase_btn.click(lambda: (None, "?", 0.0, [], gr.BarPlot.update(y=[10]*10)), outputs=[sketchpad, sketch_result_digit, sketch_result_confidence, sketch_top3_table, sketch_probability_bar])
    
    gr.Markdown("---")
    gr.Markdown("### 🤖 模型信息")
    gr.Markdown("卷积神经网络 (CNN) | 训练准确率：99.09% | 验证准确率：99.04%")

demo.launch()