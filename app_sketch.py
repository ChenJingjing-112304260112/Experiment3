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
    
    # 获取Top-3预测结果
    top3_indices = np.argsort(prediction[0])[::-1][:3]
    top3_results = [(int(i), float(prediction[0][i]*100)) for i in top3_indices]
    
    return digit, confidence, top3_results, prediction[0]

# 创建Gradio界面
with gr.Blocks(title="手写数字识别系统") as demo:
    gr.Markdown("# ✏️ 交互式手写数字识别")
    gr.Markdown("在下方画板上手写数字，系统将实时识别")
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(label="手写画板", shape=(280, 280))
            with gr.Row():
                clear_btn = gr.Button("清空画板")
                predict_btn = gr.Button("识别")
        with gr.Column():
            output_digit = gr.Number(label="预测数字", scale=2)
            output_confidence = gr.Number(label="置信度 (%)")
            top3_output = gr.DataFrame(
                headers=["数字", "置信度 (%)"],
                label="Top-3 预测结果"
            )
            probability_plot = gr.BarPlot(
                x=[0,1,2,3,4,5,6,7,8,9],
                y=[0.1]*10,
                label="概率分布",
                title="各数字概率"
            )
    
    def update_plot(probabilities):
        return gr.BarPlot.update(
            y=probabilities.tolist(),
            x=[0,1,2,3,4,5,6,7,8,9]
        )
    
    def handle_predict(image):
        if image is None:
            return None, None, None, update_plot(np.zeros(10))
        
        digit, confidence, top3, probabilities = predict_digit(image)
        return digit, confidence, top3, update_plot(probabilities)
    
    predict_btn.click(handle_predict, inputs=sketchpad, outputs=[output_digit, output_confidence, top3_output, probability_plot])
    clear_btn.click(lambda: None, None, sketchpad)
    
    gr.Markdown("---")
    gr.Markdown("### 使用说明：")
    gr.Markdown("1. 使用鼠标在画板上书写数字（0-9）")
    gr.Markdown("2. 点击识别按钮获取预测结果")
    gr.Markdown("3. 点击清空画板重新书写")

if __name__ == "__main__":
    demo.launch(share=True)