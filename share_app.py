import gradio as gr
import tensorflow as tf
import numpy as np

# 加载模型
model = tf.keras.models.load_model('model.h5')

def predict_digit(image):
    if image is None:
        return "?"
    
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
    
    return f"数字: {digit} (置信度: {confidence:.2f}%)"

# 创建界面
demo = gr.Interface(
    fn=predict_digit,
    inputs=gr.Sketchpad(label="手写画板"),
    outputs=gr.Textbox(label="预测结果"),
    title="✏️ 手写数字识别",
    description="在画板上书写数字(0-9)，点击提交按钮识别"
)

# 启动应用并生成共享链接
demo.launch(share=True)