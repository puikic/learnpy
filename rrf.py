"""多路召回结果融合

实现函数 rrf(rankings, k, c=60)。

背景：同一个查询经过多个检索器，得到多个按相关性从高到低排列的
文档 ID 列表。我们希望综合这些排名，返回最值得推荐的前 k 个文档。

参数说明：
    rankings：多个排名列表，每个文档 ID 都是整数。
    k：希望返回的文档数量，为非负整数。
    c：计分公式中的正数，默认值为 60。

计分规则：
    文档在某个列表中排名第 r，该列表给它贡献 1 / (c + r) 分。
    排名 r 从 1 开始；未出现的文档在该列表中得 0 分。
    一个文档的总分等于它在各列表中得分之和。

返回规则：
    按总分从高到低，返回最多 k 个不重复的文档 ID。
    总分相同时，文档 ID 小的排在前面。

边界约定：
    允许空输入和空列表。
    k=0 时返回 []；k 大于不同文档的数量时，返回全部文档。
    k<0 或 c<=0 时，抛出 ValueError。
    同一列表中若有重复 ID，只按第一次出现的位置计分。
    排名使用原始位置，后续重复项仍占位置。
    例如 [7, 7, 8] 中，文档 8 的排名是 3，不是 2。
    不同列表中的同一文档可以累加得分。
示例：
    rankings = [[1, 2, 3], [2, 1, 4], []]
    k = 3
    c = 60
    返回结果 = [1, 2, 3]

解释：文档 1 和 2 的总分都是 1/61 + 1/62，按 ID 排成 1、2。
文档 3 和 4 的总分都是 1/63，因此前 3 个文档为 [1, 2, 3]。
"""


from collections import defaultdict


def rrf(rankings: list[list[int]], k: int, c: float = 60) -> list[int]:
    if k < 0 or c <= 0:
        raise ValueError("k 必须为非负整数，c 必须为正数")

    docs = defaultdict(int)  # doc_id -> 总分

    for ranking in rankings:
        seen = set()  # 本列表已计分过的文档（重复 ID 只按第一次出现计分）
        for i, doc_id in enumerate(ranking):  # i 用原始位置，重复项仍占位
            if doc_id not in seen:
                seen.add(doc_id)
                docs[doc_id] += 1 / (c + i + 1)  # 排名 r = i + 1

    # 总分降序；同分时 ID 小者在前（对分数取负，让两种要求共用一次排序）
    ordered = sorted(docs, key=lambda d: (-docs[d], d))
    return ordered[:k]  # k=0 得 []；k 超过文档数时自然返回全部
    

