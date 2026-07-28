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


if __name__ == "__main__":
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]

    result = auc(labels, scores)
    print(f"AUC: {result:.2f}")  # 0.75
