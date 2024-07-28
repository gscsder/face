# coding   : utf-8
# @Time    : 2024/7/27
# @Author  : Gscsd
# @File    : db.py
# @Software: PyCharm
# pip install pymilvus==2.2.7
from pymilvus import connections, CollectionSchema, FieldSchema, DataType, Collection, utility,MilvusClient
MilvusClient()

# 连接数据库
connections.connect(
    alias="default",
    user='minioadmin', password='minioadmin',
    host='localhost', port='19530'
)

# 创建集合
people_id = FieldSchema(
    name='id', dtype=DataType.INT64, is_primary=True
)
people_name = FieldSchema(
    name='name', dtype=DataType.VARCHAR, max_length=200
)
people_metrics = FieldSchema(
    name='metrics', dtype=DataType.FLOAT_VECTOR, dim=3
)
schema = CollectionSchema(fields=[people_id, people_name, people_metrics])
collection = Collection(name='people', schema=schema, using='default', shards_num=2)
# 插入数据
data = [
    [1, 2, 3, 4, 5, 6],
    ['小明','小月','小王','小李','小张','小赵'],
    [[1.8, 75, 25],[1.75, 70, 24],[1.8, 80, 28],[1.78, 78, 30],[1.75, 70, 23],[1.8, 76, 29]]
]
mr = collection.insert(data)
collection.flush()
