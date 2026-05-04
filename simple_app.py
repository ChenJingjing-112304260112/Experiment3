import gradio as gr
import tensorflow as tf
import numpy as np

# 加载模型
model = tf.keras.models.load_model('model.h5')

def predict_digit(image):
    if image is None:
        return "?", ""
    
    # 处理图片
    image = np.mean(image, axis=2) if len(image.shape) == 3 else image
    image = np.resize(image, (28, 28))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)
    
    # 预测
    prediction = model.predict(image, verbose=0)
    digit = np.argmax(prediction)
    confidence = prediction[0][digit] * 100
    
    return str(digit), f"{confidence:.2f}%"

# 创建界面
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="手写数字"),
    outputs=[
        gr.Textbox(label="预测结果"),
        gr.Textbox(label="置信度")
    ],
    title="手写数字识别",
    description="在画板上书写数字，系统将识别"
)

demo.launch(share=True)