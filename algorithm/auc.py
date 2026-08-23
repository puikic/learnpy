def auc(y_true, y_score):
    positive_scores = [score for label, score in zip(y_true, y_score) if label == 1]
    negative_scores = [score for label, score in zip(y_true, y_score) if label == 0]

    count = 0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                count += 1
            elif positive == negative:
                count += 0.5

    return count / (len(positive_scores) * len(negative_scores))

# 按分数排序后逐组扫描，用“当前组正样本数 × 此前已出现的负样本数”批量累加赢分（1分），
# 并用“组内正负样本数乘积 × 0.5”批量累加平局分
def auc_fast(y_true, y_score):
    # 1. 将分数和标签配对，按分数从小到大排序
    pairs = sorted(zip(y_score, y_true), key=lambda x: x[0])
    
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    
    # 边界保护（如果没有正样本或负样本，AUC无定义，通常返回0.5）
    if total_pos == 0 or total_neg == 0:
        return 0.5

    auc = 0.0
    # 记录「分数严格小于当前组」的负样本个数
    neg_before = 0  
    i = 0
    n = len(pairs)

    while i < n:
        # 2. 找出所有分数相同的样本（处理平局）
        current_score = pairs[i][0]
        pos_in_group = 0
        neg_in_group = 0

        j = i
        while j < n and pairs[j][0] == current_score:
            if pairs[j][1] == 1:
                pos_in_group += 1
            else:
                neg_in_group += 1
            j += 1

        # 3. 核心统计（完全对应你的暴力逻辑）
        # 情况 A：组内正样本 vs 之前分数更低的负样本 => 相当于 positive > negative，计 1 分
        auc += pos_in_group * neg_before
        
        # 情况 B：组内正样本 vs 组内负样本（分数相等）=> 相当于 positive == negative，计 0.5 分
        auc += pos_in_group * neg_in_group * 0.5

        # 4. 更新：把当前组的负样本归入“之前”的范围
        neg_before += neg_in_group
        i = j

    # 除以总的配对数量
    return auc / (total_pos * total_neg)

if __name__ == "__main__":
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]

    result = auc(labels, scores)
    print(f"AUC: {result:.2f}")  # 0.75
