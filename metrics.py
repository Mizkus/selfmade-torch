def binary_classification_metrics(prediction, ground_truth):
    TP = sum((prediction == 1) & (ground_truth == 1))
    TN = sum((prediction == 0) & (ground_truth == 0))
    FP = sum((prediction == 1) & (ground_truth == 0))
    FN = sum((prediction == 0) & (ground_truth == 1))

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (TP + TN) / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)

    return accuracy, precision, recall, f1


def multiclass_accuracy(prediction, ground_truth):
    correct_predictions = sum(prediction == ground_truth)
    total_predictions = len(ground_truth)
    accuracy = correct_predictions / total_predictions
    return accuracy
