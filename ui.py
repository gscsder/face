# coding   : utf-8
# @Time    : 2024/7/26
# @Author  : Gscsd
# @File    : ui.py
# @Software: PyCharm

import gradio as gr
import numpy as np
import cv2


def answer(message, history):
    history = history or []
    message = message.lower()
    if message == "你好":
        response = "你好，有什么可以帮到你吗?"

    elif message == "你是谁":
        response = "我是虚拟数字人幻静，你可以叫我小静或者静静。"

    elif message == "你能做什么":
        response = "我可以陪你聊天，回答你的问题，我还可以做很多很多事情！"

    else:
        response = "你的这个问题超出了我的理解范围，等我学习后再来回答你。或者你可以问我其他问题，能回答的我尽量回答你！"

    history.append((message, response))

    return history, history


def gray_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray


with gr.Blocks() as demo:
    # 设置tab选项卡
    with gr.Tab("图像灰度处理"):
        # Blocks特有组件，设置所有子组件按垂直排列
        # 垂直排列是默认情况，不加也没关系
        with gr.Column():
            img_input = gr.Image()
            img_output = gr.Image()
            img_button = gr.Button("灰度化")
    with gr.Tab("对话框"):
        # Blocks特有组件，设置所有子组件按水平排列
        with gr.Row():
            state = gr.State([])
            chatbot = gr.Chatbot(label="消息记录")
            txt = gr.Textbox(show_label=False, placeholder="请输入你的问题")
    # 设置折叠内容
    img_button.click(gray_image, inputs=img_input, outputs=img_output)
    txt.submit(fn=answer, inputs=[txt, state], outputs=[chatbot, state])

demo.launch()


