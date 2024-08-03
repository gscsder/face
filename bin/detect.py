# coding   : utf-8
# @Time    : 2024/7/26
# @Author  : Gscsd
# @File    : detect.py
# @Software: PyCharm
import cv2
import numpy as np
import onnxruntime
from PIL import Image, ImageDraw, ImageFont
from insightface.app import FaceAnalysis
from .db import Database, Face, Person, calc_similarity

providers = ['CPUExecutionProvider']


class FaceRecognition:
    def __init__(self, gpu_id=0, threshold=0.6, det_thresh=0.6, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.image = None
        self.faces = None
        self.gpu_id = gpu_id
        self.providers = ['CPUExecutionProvider']
        self.db = Database()
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.check_gpu()
        self.model = FaceAnalysis(providers=self.providers)
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)

    def check_gpu(self):
        if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
            self.providers.insert(0, 'CUDAExecutionProvider')

    def draw_img(self, persons: list[Face]):

        dimg = self.image.copy()
        for i in range(len(self.faces)):
            face = self.faces[i]
            box = face.bbox.astype(int)
            # 线宽200分之一取整
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), int(dimg.shape[0] / 200))
            if face.kps is not None:
                kps = face.kps.astype(int)
                for ll in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if ll == 0 or ll == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[ll][0], kps[ll][1]), 1, color,
                               2)
            # 字体大小40分之一取整，偏移至左上方
            font_size = int(dimg.shape[0] / 40 + 1)
            dimg = self.put_chn_text(dimg, f"{persons[i].name}（{persons[i].similarity:.2%}）",
                                     (box[0], box[1] - int(font_size * 1.5)), font_size)

        return dimg

    @staticmethod
    def put_chn_text(img, text: str, position: tuple[float, float], font_size: int):

        font = ImageFont.truetype('STZHONGS.TTF', font_size, encoding="utf-8")
        img_pil = Image.fromarray(img[..., ::-1])  # 转成 PIL 格式
        draw = ImageDraw.Draw(img_pil)  # 创建绘制对象
        draw.text(xy=position, text=text,
                  font=font, fill=(0, 255, 0))
        return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)  # 再转成 OpenCV 的格式，记住 OpenCV 中通道排布是 BGR

    def recognition(self, image, threshold=None) -> list[Face]:
        threshold = self.threshold if threshold is None else threshold
        persons = self.feature(image)
        results = []

        for p in persons:
            # 开始人脸识别
            r = self.db.search_by_embedding(p.embedding)
            if r and r[0].similarity > threshold:
                results.append(r[0])
            else:
                results.append(Face(10, p.dict()))
        return results

    def feature(self, image, name="未知") -> list[Person]:
        self.image = cv2.imread(image) if isinstance(image, str) else image
        self.faces = self.model.get(self.image)
        # 抽取归一化后特征向量
        return [Person(name=name, gender=1 if i.gender else 0, embedding=i.normed_embedding) for i in self.faces]

    def check_compare(self, img1, img2) -> str:
        """
        检测是否可以比较相似度，都必须只有一张人脸
        :return:
        """
        p1, p2 = self.feature(img1), self.feature(img2)
        if (n := len(p1) + len(p2)) < 2:
            return "未检测到人脸"
        elif n == 2 and len(p1) == 1:
            return f"相似度：{self.face_compare(p1[0].embedding, p2[0].embedding):.2%}"
        else:
            return "人脸数量过多"

    @staticmethod
    def face_compare(embedding1, embedding2) -> float:
        # 比较相似度
        diff = np.subtract(embedding1, embedding2)
        dist = np.sum(np.square(diff))
        return calc_similarity(dist)

    def register(self, image, user_name, id_=None, gender=None, source=None, threshold=None) -> str | Face:
        threshold = self.threshold if threshold is None else threshold
        faces = self.recognition(image, threshold=threshold)
        if not faces:
            return '图片检测不到人脸'
        if len(faces) > 1:
            return '一次只能录入一张人脸'

        if faces[0].similarity >= threshold:
            return '该用户已存在'
        else:
            faces[0].name = user_name
            if id_:
                faces[0].id = id_
            if gender:
                faces[0].gender = gender
            if source:
                faces[0].source = source

            res = self.db.insert_one(Person(**faces[0].dict()))
            if res.succ_count:
                return faces[0]
            return "录入失败"
