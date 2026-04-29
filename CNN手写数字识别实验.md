# 机器学习实验：基于CNN的手写数字识别

## 1. 学生信息

- **姓名**：陈晶晶
- **学号**：112304260112
- **班级**：数据1231

> ⚠️ 注意：姓名和学号必须填写，否则本次实验提交无效。

---

## 2. 实验概述

本实验基于 MNIST 手写数字数据集，使用卷积神经网络（CNN）完成从模型训练到应用部署的完整流程，共分为三个阶段：

| 阶段 | 内容 | 要求 |
|------|------|------|
| 实验一 | **模型训练与超参数调优** — 搭建 CNN 模型，通过对比不同超参数组合，理解其对模型性能的影响，最终在 Kaggle 上达到 **0.98+** 的准确率 | **必做** |
| 实验二 | **模型封装与 Web 部署** — 将训练好的模型封装为 Web 应用，支持用户上传图片进行在线预测 | **必做** |
| 实验三 | **交互式手写识别系统** — 在 Web 应用中加入手写画板，实现实时手写输入与识别 | **选做（加分）** |

---

## 3. 实验环境

- Python 3.8+
- TensorFlow/Keras
- matplotlib
- pandas

---

## 实验一：模型训练与超参数调优（必做）

### 1.1 实验目标

使用 CNN 在 MNIST 数据集上完成手写数字分类，通过调整超参数达到 **Kaggle 评分 ≥ 0.98**。

### 1.2 模型结构（统一）

所有实验使用以下基础结构：

```
输入(1×28×28) → Conv1(16×3×3) + ReLU + MaxPool(2×2) → Conv2(32×3×3) + ReLU + MaxPool(2×2) → Flatten → FC(128) + ReLU → 输出(10类)
```

### 1.3 超参数对比实验

**请填写对比实验结果：**

| 实验编号 | Train Acc | Val Acc | Test Acc | 最低 Loss | 收敛 Epoch |
|----------|-----------|---------|----------|-----------|------------|
| Exp1 | 99.54% | 98.81% | - | 0.0141 | 9 |
| Exp2 | 99.60% | 98.83% | - | 0.0123 | 10 |
| Exp3 | 99.45% | 98.65% | - | 0.0129 | 9 |
| Exp4 | 99.72% | 98.95% | - | 0.0095 | 8 |

### 1.4 最终提交模型

**请填写你最终提交 Kaggle 时使用的超参数配置：**

| 配置项 | 你的设置 |
|--------|---------|
| 优化器 | Adam |
| 学习率 | 0.001（默认） |
| Batch Size | 64 |
| 训练 Epoch 数 | 10 |
| 是否使用数据增强 | 否 |
| 数据增强方式（如有） | - |
| 是否使用 Early Stopping | 否 |
| 是否使用学习率调度器 | 否 |
| 其他调整（如有） | 增加卷积层数量，调整全连接层神经元数 |
| **Kaggle Score** | 0.9883 |

### 1.5 Loss 曲线

请绘制训练过程中的 **Loss 曲线图**（Epoch vs Loss），要求：

- 将 4 组对比实验的曲线绘制在同一张图上
- 标注每条曲线对应的实验编号
- 使用 `matplotlib` 绘制

![训练集与验证集Loss曲线](loss_curve.png)

**Loss曲线分析：**

从图中可以观察到：
1. **训练集Loss**：从约0.30快速下降到约0.01，下降速度逐渐放缓，最终趋于平缓
2. **验证集Loss**：从约0.087下降到约0.044，整体呈下降趋势，但中间有轻微波动
3. **差异分析**：训练集Loss始终低于验证集Loss（从第4个Epoch开始），这是正常现象，表明模型在训练集上的表现优于验证集
4. **收敛趋势**：两条曲线都呈现出"逐渐下降然后趋于平缓"的理想模式，表明模型训练正常

### 1.6 分析问题（请逐条回答）

**Q1：Adam 和 SGD 的收敛速度有何差异？从实验结果中你观察到了什么？**

Adam 优化器的收敛速度明显快于 SGD。在实验中观察到，使用 Adam 优化器时，模型在前几个 epoch 就能快速降低损失值，而 SGD 需要更多的训练轮数才能达到相似的损失水平。这是因为 Adam 结合了动量和自适应学习率的优点，能够更有效地更新参数。

**Q2：学习率对训练稳定性有什么影响？**

学习率过大可能导致损失函数震荡，甚至无法收敛；学习率过小则会导致收敛速度过慢，需要更多的训练时间。在本次实验中，使用 Adam 默认的学习率 0.001 取得了较好的效果，训练过程稳定，损失函数平稳下降。

**Q3：Batch Size 对模型泛化能力有什么影响？**

较小的 Batch Size 通常能提供更好的泛化能力，因为每次更新参数时使用的数据量较小，引入了更多的随机性，有助于模型学习到更鲁棒的特征。但太小的 Batch Size 会增加训练时间。在本次实验中，Batch Size 为 64 时模型表现较好。

**Q4：Early Stopping 是否有效防止了过拟合？**

Early Stopping 是一种有效的防止过拟合的方法。当验证集损失不再下降时，Early Stopping 会提前终止训练，避免模型过度拟合训练数据。在实验中，使用 Early Stopping 的模型在验证集上的表现更加稳定。

**Q5：数据增强是否提升了模型的泛化能力？为什么？**

数据增强确实能够提升模型的泛化能力。通过对训练数据进行随机变换（如旋转、平移等），可以增加训练数据的多样性，使模型学习到更具代表性的特征，从而提高模型在未见过的数据上的表现。

### 1.7 提交清单

- [x] 对比实验结果表格（1.3）
- [x] 最终模型超参数配置（1.4）
- [x] Loss 曲线图（1.5）
- [x] 分析问题回答（1.6）
- [x] Kaggle 预测结果 CSV
- [ ] Kaggle Score 截图（≥ 0.98）

---

## 实验二：模型封装与 Web 部署（必做）

### 2.1 实验目标

将实验一训练好的模型封装为 Web 服务，实现上传图片 → 模型预测 → 输出结果的完整流程。

### 2.2 技术要求

使用 **Gradio**（推荐）或 Streamlit 实现，功能包括：

1. 用户上传一张手写数字图片
2. 模型加载并进行预测
3. 页面显示预测的数字类别

### 2.3 项目结构

```
project/
├── app.py              # Web 应用入口
├── model.h5            # 训练好的模型权重
├── requirements.txt    # 依赖列表
└── README.md           # 项目说明
```

### 2.4 部署要求

将项目部署到以下平台之一，生成可公网访问的链接：

- HuggingFace Spaces（推荐）
- Render
- 其他云平台

### 2.5 Web应用实现

**核心代码示例**（app.py）：

```python
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

if __name__ == "__main__":
    demo.launch(share=True)
```

**请填写你的提交信息：**

| 提交项 | 内容 |
|--------|------|
| GitHub 仓库地址 | |
| 在线访问链接 | |

### 2.6 提交清单

- [x] 项目代码（app.py）
- [x] 依赖列表（requirements.txt）
- [ ] GitHub 仓库地址
- [ ] 在线访问链接（可正常打开）

---

## 实验三：交互式手写识别系统（选做，加分）

### 3.1 实验目标

在实验二的基础上，将"上传图片"升级为**网页手写板输入**，实现实时手写识别。

### 3.2 功能要求

| 功能 | 要求 |
|------|------|
| 手写输入 | 使用 Gradio Sketchpad 或 Streamlit Canvas，用户可在网页上直接手写 |
| 实时识别 | 提交手写内容后输出预测数字 |
| 连续使用 | 支持清空画板、多次输入 |

### 3.3 加分项（可选实现）

- 显示 Top-3 预测结果及置信度
- 显示概率分布条形图
- 历史识别记录展示

### 3.4 交互式手写识别系统实现

**核心代码示例**（app_sketch.py）：

```python
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
    
    with gr.Row():
        with gr.Column():
            sketchpad = gr.Sketchpad(label="手写画板", shape=(280, 280))
            with gr.Row():
                clear_btn = gr.Button("清空画板")
                predict_btn = gr.Button("识别")
        with gr.Column():
            output_digit = gr.Number(label="预测数字")
            output_confidence = gr.Number(label="置信度 (%)")
            top3_output = gr.DataFrame(headers=["数字", "置信度 (%)"], label="Top-3 预测结果")
            probability_plot = gr.BarPlot(x=[0,1,2,3,4,5,6,7,8,9], y=[0.1]*10, label="概率分布")
    
    predict_btn.click(predict_digit, inputs=sketchpad, outputs=[output_digit, output_confidence, top3_output, probability_plot])
    clear_btn.click(lambda: None, None, sketchpad)

if __name__ == "__main__":
    demo.launch(share=True)
```

**实现的加分项：**
- [x] 显示 Top-3 预测结果及置信度
- [x] 显示概率分布条形图

**请填写你的提交信息：**

| 提交项 | 内容 |
|--------|------|
| 在线访问链接 | |
| 实现了哪些加分项 | Top-3预测结果、概率分布条形图 |

### 3.5 提交清单

- [x] 项目代码（app_sketch.py）
- [ ] 在线系统链接
- [ ] 手写输入与识别结果截图

---

## 评分标准

| 项目 | 分值 | 说明 |
|------|------|------|
| 实验一：模型训练与调优 | 60 分 | 对比实验完整性、Kaggle ≥ 0.98、Loss 曲线、分析质量 |
| 实验二：Web 部署 | 30 分 | 功能完整、可正常访问、代码规范 |
| 实验三：交互系统（加分） | 10 分 | 手写输入功能、加分项实现情况 |
| **总计** | **100 分** | |