import math 



def ndcg(y_true, y_score, k=None):
    pred = sorted(zip(y_score, y_true), reverse=True)
    ideal = sorted(y_true, reverse=True) 

    def dcg(relevances, k=None):
        limit = len(relevances) if k is None else min(k, len(relevances))
        # 位置 i 从 0 起（分母 log2(i+2)）
        return sum((2**i - 1) / math.lo2(i+2)
            for i, rel in enumerate(relevances[:limit])
        )
    
    return dcg([rel for _, rel in pred], k) / dcg(ideal, k)
