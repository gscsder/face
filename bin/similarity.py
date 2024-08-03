# coding   : utf-8
# @Time    : 2024/7/29
# @Author  : Gscsd
# @File    : similarity.py
# @Software: PyCharm

import math

from sympy import symbols, Eq, solve


def sigmoid_para(x1=1, s1=0.95, x2=1.6, s2=0.6) -> tuple[float, float]:
    """
    求解Sigmoid函数未知参数，将x1-x2映射为0.95-0.6的相似度，此区间内分布更密集，可自由更改
    """
    a, b = symbols("a b")
    equation_1 = Eq(1 / (1 + math.e ** (a * x1 + b)), s1)
    equation_2 = Eq(1 / (1 + math.e ** (a * x2 + b)), s2)
    [(a, b)] = solve((equation_1, equation_2), (a, b))
    return a, b


A, B = sigmoid_para()


def calc_similarity(x: float) -> float:
    """
    计算相似度
    """
    return 1 / (1 + math.e ** (A * x + B))


if __name__ == '__main__':
    print(f"x=0.0, similarity={calc_similarity(0.0):.2%}")
    print(f"x=1.5, similarity={calc_similarity(1.5):.2%}")
    print(f"x=3.0, similarity={calc_similarity(3.0):.2%}")
