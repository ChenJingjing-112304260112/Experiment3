import sys
print("Python版本:", sys.version)

try:
    import gradio as gr
    print("Gradio导入成功, 版本:", gr.__version__)
except Exception as e:
    print("Gradio导入失败:", e)
    sys.exit(1)

try:
    import tensorflow as tf
    print("TensorFlow导入成功, 版本:", tf.__version__)
except Exception as e:
    print("TensorFlow导入失败:", e)
    sys.exit(1)

try:
    import numpy as np
    print("NumPy导入成功, 版本:", np.__version__)
except Exception as e:
    print("NumPy导入失败:", e)
    sys.exit(1)

# 测试模型加载
try:
    model = tf.keras.models.load_model('model.h5')
    print("模型加载成功")
    print("模型结构:", model.summary())
except Exception as e:
    print("模型加载失败:", e)
    sys.exit(1)

# 定义预测函数
def predict_digit(image):
    if image is None:
        return "请输入数字"
    
    try:
        image = np.mean(image, axis=2) if len(image.shape) == 3 else image
        image = np.resize(image, (28, 28))
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        
        prediction = model.predict(image, verbose=0)
        digit = np.argmax(prediction)
        confidence = prediction[0][digit] * 100
        return f"预测: {digit} (置信度: {confidence:.2f}%)"
    except Exception as e:
        print("预测错误:", e)
        return f"错误: {str(e)}"

# 创建界面
print("\n正在创建界面...")
try:
    demo = gr.Interface(
        fn=predict_digit,
        inputs=gr.Sketchpad(label="手写数字"),
        outputs=gr.Textbox(label="预测结果"),
        title="手写数字识别系统",
        description="在画板上书写数字(0-9)，系统将识别"
    )
    print("界面创建成功")
    
    print("\n正在启动服务器...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
except Exception as e:
    print("启动失败:", e)
    import traceback
    traceback.print_exc()