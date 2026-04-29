import gradio as gr
import tensorflow as tf
import numpy as np

# 加载训练好的模型
model = tf.keras.models.load_model('model.h5')

def predict_digit(image):
    # 处理输入图片
    image = np.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    # 预测
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    confidence = prediction[0][digit] * 100
    
    return f"预测结果: {digit} (置信度: {confidence:.2f}%)"

# 创建Gradio界面
with gr.Blocks(title="手写数字识别") as demo:
    gr.Markdown("# 📝 手写数字识别系统")
    gr.Markdown("上传一张手写数字图片，系统将自动识别数字")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="numpy", label="上传图片")
            submit_btn = gr.Button("识别")
        with gr.Column():
            output_text = gr.Textbox(label="识别结果")
    
    submit_btn.click(predict_digit, inputs=input_image, outputs=output_text)
    
    gr.Markdown("---")
    gr.Markdown("### 使用说明：")
    gr.Markdown("1. 点击上传按钮选择手写数字图片")
    gr.Markdown("2. 图片应为28x28像素的灰度图像")
    gr.Markdown("3. 点击识别按钮获取预测结果")

if __name__ == "__main__":
    demo.launch(share=True)