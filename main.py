# coding   : utf-8
# @Time    : 2024/7/26
# @Author  : Gscsd
# @File    : main.py
# @Software: PyCharm
import numpy as np
import gradio as gr
import os
from bin.detect import FaceRecognition
from httpx._config import Timeout
import httpx

# 修正numpy版本兼容问题
np.int = int
old_lst = np.linalg.lstsq
np.linalg.lstsq = lambda a, b, rcond=None: old_lst(a, b, rcond)
# 解决httpx超时问题
httpx._config.DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=30.0)

os.environ['GRADIO_TEMP_DIR'] = './tmp'

if __name__ == '__main__':
    face_recognition = FaceRecognition()
    css = """
    .custom_img_row {
    max-height: 70vh !important;
    overflow-y: scroll !important;
}
    """


    def detect_img(img, th):
        results = face_recognition.recognition(img, th / 100)
        if not results:
            return (img, []), "未检测到人脸"

        unknown_num = 0
        for i in results:
            if i.name == "未知":
                unknown_num = unknown_num + 1
                i.name += str(unknown_num)
        img_ = face_recognition.draw_img(results)
        # 选出识别到的人名，位置均以0替代
        return ((img_, [((0, 0, 0, 0), i.name) for i in results]),
                "，".join([f"{i.name}（{i.similarity:.2%}）" for i in results]))


    def add_name(img, name):
        if not name:
            return "必须输入人名"
        return face_recognition.register(img, name)


    def compare_img(img1, img2):
        return face_recognition.check_compare(img1, img2)


    with gr.Blocks(css=css, delete_cache=(120, 1800)) as app:
        # 设置tab选项卡
        with gr.Tab("人脸检测"):
            with gr.Row():
                slider = gr.Slider(minimum=60, maximum=95, step=1, value=70, label="相似度阈值（%）", show_label=True)
                t_result = gr.Textbox(label="检测结果")
            with gr.Row(elem_classes="custom_img_row") as row:
                img_input = gr.Image(label="输入图像")
                img_output = gr.AnnotatedImage()

            img_button = gr.Button("检测")
        with gr.Tab("录入人脸"):
            with gr.Row():
                img_input2 = gr.Image(label="输入图像")
                with gr.Column():
                    t_input2 = gr.Textbox(label="人名")
                    t_output2 = gr.Textbox(label="状态")
            img_button2 = gr.Button("录入")
        with gr.Tab("人脸对比"):
            with gr.Row():
                img_input3 = gr.Image(label="输入图像1")
                img_input4 = gr.Image(label="输入图像2")
            t_output3 = gr.Textbox(label="状态/相似度")
            img_button3 = gr.Button("对比")
        img_button.click(detect_img, inputs=[img_input, slider], outputs=[img_output, t_result])
        img_button2.click(add_name, inputs=[img_input2, t_input2], outputs=t_output2)
        img_button3.click(compare_img, inputs=[img_input3, img_input4], outputs=t_output3)

    app.launch(server_name="0.0.0.0", server_port=5000)
