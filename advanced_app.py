import gradio as gr
import tensorflow as tf
import numpy as np

# 加载模型
model = tf.keras.models.load_model('model.h5')

def preprocess_image(image):
    """预处理图片：转换为灰度、调整大小、归一化"""
    if image is None:
        return None
    if len(image.shape) == 3:
        image = np.mean(image, axis=2)
    image = np.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    return image

def predict(image):
    """预测数字并返回结果"""
    processed_image = preprocess_image(image)
    if processed_image is None:
        return "?", 0.0, [], [0.1]*10
    
    prediction = model.predict(processed_image, verbose=0)[0]
    digit = np.argmax(prediction)
    confidence = prediction[digit] * 100
    top3_indices = np.argsort(prediction)[::-1][:3]
    top3_results = [{"数字": int(i), "置信度(%)": float(prediction[i]*100)} for i in top3_indices]
    
    return str(digit), round(confidence, 2), top3_results, prediction.tolist()

# 创建界面
with gr.Blocks(title="手写数字识别系统") as demo:
    gr.Markdown("# 🎯 手写数字识别系统")
    gr.Markdown("### 学生信息：李金彪 | 学号：112304260132 | 班级：数据1231")
    
    with gr.Tabs():
        # 实验二：上传图片
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
                    with gr.Row():
                        result_digit = gr.Textbox(label="预测数字", scale=2, placeholder="?")
                        result_confidence = gr.Textbox(label="置信度")
                    top3_table = gr.DataFrame(headers=["排名", "数字", "置信度(%)"], label="Top-3 预测")
                    probability_bar = gr.BarPlot(x=[0,1,2,3,4,5,6,7,8,9], y=[10]*10, label="概率分布")
        
        # 实验三：手写画板
        with gr.Tab("✏️ 实验三：手写画板"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 在画板上书写数字")
                    sketchpad = gr.Sketchpad(label="手写画板")
                    with gr.Row():
                        predict_btn = gr.Button("🔍 识别", variant="primary")
                        erase_btn = gr.Button("🗑️ 清空")
                with gr.Column(scale=1):
                    gr.Markdown("#### 预测结果")
                    with gr.Row():
                        sketch_result_digit = gr.Textbox(label="预测数字", scale=2, placeholder="?")
                        sketch_result_confidence = gr.Textbox(label="置信度")
                    sketch_top3_table = gr.DataFrame(headers=["排名", "数字", "置信度(%)"], label="Top-3 预测")
                    sketch_probability_bar = gr.BarPlot(x=[0,1,2,3,4,5,6,7,8,9], y=[10]*10, label="概率分布")
    
    # 绑定事件处理函数
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
    gr.Markdown("### 📖 使用说明")
    gr.Markdown("**实验二**：点击上传区域，选择手写数字图片进行识别")
    gr.Markdown("**实验三**：使用鼠标在画板上书写数字，点击识别按钮获取结果")
    gr.Markdown("### 🤖 模型信息")
    gr.Markdown("模型类型：卷积神经网络 (CNN) | 训练准确率：99.09% | 验证准确率：99.04%")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=5000)