import gradio as gr
import tensorflow as tf
import numpy as np

print("Gradio version:", gr.__version__)
print("TensorFlow version:", tf.__version__)

# 加载模型
try:
    model = tf.keras.models.load_model('model.h5')
    print("模型加载成功")
except Exception as e:
    print("模型加载失败:", e)

def predict_digit(image):
    if image is None:
        return "?"
    
    try:
        image = np.mean(image, axis=2) if len(image.shape) == 3 else image
        image = np.resize(image, (28, 28))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        prediction = model.predict(image, verbose=0)
        digit = np.argmax(prediction)
        return str(digit)
    except Exception as e:
        print("预测错误:", e)
        return "错误"

# 创建界面
try:
    demo = gr.Interface(
        fn=predict_digit,
        inputs=gr.Sketchpad(),
        outputs=gr.Textbox(label="预测结果"),
        title="手写数字识别"
    )
    print("界面创建成功")
    
    demo.launch(server_name="0.0.0.0", server_port=7860)
except Exception as e:
    print("启动失败:", e)