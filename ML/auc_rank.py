def auc(y_true, y_score):
    # 按分数升序排序，秩从 1 开始（Mann-Whitney U / 秩和公式）
    order = sorted(range(len(y_score)), key=lambda i: y_score[i])

    # 分配秩：分数相等的样本取平均秩，与暴力法 tie 记 0.5 的语义一致
    ranks = [0.0] * len(y_score)
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and y_score[order[j]] == y_score[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2  # [i+1, j] 区间的平均秩
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j

    positive_rank_sum = sum(rank for label, rank in zip(y_true, ranks) if label == 1)
    num_positive = sum(1 for label in y_true if label == 1)
    num_negative = len(y_true) - num_positive

    return (positive_rank_sum - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative)


if __name__ == "__main__":
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]

    result = auc(labels, scores)
    print(f"AUC: {result:.2f}")  # 0.75
