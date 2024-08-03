# coding   : utf-8
# @Time    : 2024/8/3
# @Author  : Gscsd
# @File    : ui.py
# @Software: PyCharm
import os

import cv2
import gradio as gr

from .detect import FaceRecognition
from .config import delete_cache_time, face_dir, temp_dir, pic_size

os.environ['GRADIO_TEMP_DIR'] = temp_dir
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


def add_name(img_origin, img_crop, name: str, arg1, arg2, arg3) -> str:
    if not name:
        return "必须输入人名"
    img_crop = img_crop["composite"]
    # 未修改id和gender则设为None
    arg1 = None if arg1 == 10 ** 7 - 1 else arg1
    arg2 = None if arg2 == - 1 else arg2
    r = face_recognition.register(img_crop, name.strip(), arg1, arg2, arg3)
    if isinstance(r, str):
        return r
    # 保存录入图片，缩略图设置尺寸为160
    thumbnail = cv2.resize(img_crop, (pic_size, pic_size))
    cv2_write(thumbnail, f"{r.id}.jpg")
    cv2_write(img_origin, f"{r.id}_hd.jpg")
    return f"录入【{r.name}】成功"


def cv2_write(img, name):
    if not os.path.exists(face_dir):
        os.makedirs(face_dir)
    # 适配中文路径写入，注意颜色模式转换
    cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))[1].tofile(os.path.join(face_dir, name))


def compare_img(img1, img2):
    return face_recognition.check_compare(img1, img2)


def create_tab1():
    with gr.Tab("人脸检测"):
        with gr.Row():
            slider = gr.Slider(minimum=60, maximum=95, step=1, value=70, label="相似度阈值（%）", show_label=True)
            result = gr.Textbox(label="检测结果")
        with gr.Row(elem_classes="custom_img_row"):
            img_input = gr.Image(label="输入图像")
            img_output = gr.AnnotatedImage(label="标注图像")
        button = gr.Button("检测")
        button.click(detect_img, inputs=[img_input, slider], outputs=[img_output, result])


def create_tab2():
    with gr.Tab("录入人脸"):
        with gr.Row():
            img_origin = gr.Image(label="输入图像")
            img_crop = gr.ImageEditor(label="裁剪图像", crop_size="1:1", image_mode="RGB",
                                      layers=False, brush=False, eraser=False)
            img_origin.upload(lambda x: x, img_origin, img_crop)
        with gr.Row():
            name = gr.Textbox(label="人名", max_lines=1)
            status = gr.Textbox(label="状态")
        with gr.Accordion("配置信息", open=False):
            with gr.Row():
                id_ = gr.Number(minimum=10 ** 7 - 1, maximum=10 ** 8 - 1, value=10 ** 7 - 1, precision=0,
                                label="ID（可选）",
                                info="个人唯一id，八位数字")
                gender = gr.Number(minimum=-1, value=-1, maximum=1, precision=0, label="性别（可选）",
                                   info="性别：0 - 女，1 - 男")
                source = gr.Textbox(label="来源（可选）", max_lines=1, info="简短描述录入来源",
                                    value="手动录入")
        button = gr.Button("录入")
        button.click(add_name, inputs=[img_origin, img_crop, name, id_, gender, source], outputs=status)


def create_tab3():
    with gr.Tab("人脸对比"):
        with gr.Row():
            img1 = gr.Image(label="输入图像1")
            img2 = gr.Image(label="输入图像2")
        output = gr.Textbox(label="状态/相似度")
        button = gr.Button("对比")
        button.click(compare_img, inputs=[img1, img2], outputs=output)


with gr.Blocks(css=css, delete_cache=delete_cache_time) as app:
    create_tab1()
    create_tab2()
    create_tab3()
