# coding   : utf-8
# @Time    : 2024/7/26
# @Author  : Gscsd
# @File    : main.py
# @Software: PyCharm
import numpy as np
import gradio as gr
import cv2
from detect import FaceRecognition

# 修正numpy版本兼容问题
np.int = int
old_lst = np.linalg.lstsq
np.linalg.lstsq = lambda a, b, rcond=None: old_lst(a, b, rcond)

if __name__ == '__main__':
    img = cv2.imdecode(np.fromfile('cj.jpg', dtype=np.uint8), -1)
    face_recognitio = FaceRecognition()
    # 人脸注册
    result = face_recognitio.register(img, user_name='cj')


    def draw_img(img, th):
        img_ = face_recognitio.draw_img(img)
        results = face_recognitio.recognition(img, th)
        return img_, results[0]


    # for result in results:
    #     print("识别结果：{}".format(result))

    def add_name(img, name):
        res = face_recognitio.register(img, name)
        if res == "success":
            return f"录入成功：{name}"
        return res


    with gr.Blocks() as demo:
        # 设置tab选项卡
        with gr.Tab("人脸检测"):
            # Blocks特有组件，设置所有子组件按垂直排列
            # 垂直排列是默认情况，不加也没关系
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(label="输入图")
                    sl = gr.Slider(minimum=0.1, maximum=10, step=0.1, value=1, label="阈值")
                with gr.Column():
                    img_output = gr.Image(label="输出图")
                    t_output = gr.Textbox(label="人名")
            img_button = gr.Button("检测")
        with gr.Tab("录入人脸"):
            with gr.Row():
                with gr.Column():
                    img_input2 = gr.Image(label="输入图")
                    t_input2 = gr.Textbox(label="人名")
                with gr.Column():
                    # img_output2 = gr.Image(label="输出图")
                    t_output2 = gr.Textbox()
            img_button2 = gr.Button("录入")
        # 设置折叠内容
        img_button.click(draw_img, inputs=[img_input, sl], outputs=[img_output, t_output])
        img_button2.click(add_name, inputs=[img_input2, t_input2], outputs=t_output2)
        # txt.submit(fn=answer, inputs=[txt, state], outputs=[chatbot, state])

    demo.launch(share=True)
