import gradio as gr

def greet(name):
    return f"Hello, {name}!"

with gr.Blocks() as demo:
    name = gr.Textbox(label="Name")
    output = gr.Textbox(label="Output")
    gr.Button("Greet").click(greet, name, output)

# 启动应用，使用share=True生成公开链接
demo.launch(share=True)
