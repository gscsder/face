# coding   : utf-8
# @Time    : 2024/7/29
# @Author  : Gscsd
# @File    : db.py
# @Software: PyCharm
import hashlib
import random
import time
from typing import Literal

from pydantic import BaseModel, Field
from pymilvus import (connections, FieldSchema,
                      CollectionSchema,
                      DataType,
                      Collection)

from .similarity import calc_similarity


def timestamp_to_digits() -> int:
    current_timestamp = str(time.time())  # 获取当前时间戳并转换为字符串
    encoded = hashlib.sha1(current_timestamp.encode()).hexdigest()  # 使用SHA-1哈希函数进行编码
    six_digits = int(encoded, 16) % (10 ** 6)  # 将编码结果转换为6位数字
    eight_digits_str = f"{six_digits:0<6d}{random.randint(10, 99)}"  # 在低位补0，确保编码为6位数字，额外增加2位随机数字校验
    stamp = int(eight_digits_str)
    return stamp


class Person(BaseModel):
    id: int = Field(default_factory=timestamp_to_digits)
    name: str = "未知"
    gender: Literal[0, 1]
    source: str = "未知"
    embedding: list[float]


class Face(Person):
    similarity: float = .0

    def __init__(self, distance: float, fields: dict):
        data = {"similarity": calc_similarity(distance), **fields}
        # data = {"similarity": calc_similarity(distance), "embedding": embedding, **metadata}
        super().__init__(similarity=calc_similarity(distance), **fields)
        # super().__init__(**data)


class Database:
    def __init__(self, collection_name="face", dim=512):
        self.dim = dim
        self.collection_name = collection_name
        connections.connect(host="127.0.0.1", port=19530, keep_alive=True)
        self.collection = self.get_collection()

    def get_collection(self):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="gender", dtype=DataType.INT32),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=20),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
        ]
        schema = CollectionSchema(fields, enable_dynamic_field=False)
        collection = Collection(self.collection_name, schema)
        index_param = {"index_type": "FLAT", "metric_type": "L2", "params": {}}
        collection.create_index("embedding", index_param)
        collection.load()
        return collection

    def insert_one(self, person: Person):
        r = self.collection.insert([person.dict()])
        self.collection.flush()
        return r

    def insert_many(self, persons: list[Person]):
        r = self.collection.insert([p.dict() for p in persons])
        self.collection.flush()
        return r

    def search_by_embedding(self, embedding: list[float], limit=1) -> list[Face]:
        """
        一次只搜索一个
        :param embedding:
        :param limit:
        :return:
        """
        # 空collection搜索会报错
        if self.collection.is_empty:
            return []
        res = self.collection.search([embedding], anns_field="embedding", param={},
                                     limit=limit, output_fields=["id", "name", "source", "gender", "embedding"])
        return [Face(r.distance, r.fields) for r in res[0]]

    def query_by_id(self, id_: int):
        return self.collection.query(f"id == {id_}",
                                     output_fields=["id", "name", "source", "gender", "embedding"], limit=1)

    @property
    def count(self) -> int:
        return self.collection.num_entities


if __name__ == '__main__':
    db = Database()
    s = [random.random() for _ in range(512)]
    db.insert_one(Person(gender=0, embedding=s))
    print(db.search_by_embedding(s))
