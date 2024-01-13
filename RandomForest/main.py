import numpy as np
import pandas as pd

from random_forest import RandomForest


def load_data(filename):
    df = pd.read_csv('heart.csv')
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    X, y = shuffled_df.iloc[:, :-1].values, shuffled_df.iloc[:, -1].values
    return X, y


def split_into_folds(X, y, splits):
    inputs = []
    outputs = []
    step = len(X) // splits
    for split in range(splits):
        inputs.append(X[split * step:(split + 1) * step])
        outputs.append(y[split * step:(split + 1) * step])
    return inputs, outputs


def compute_auc(random_forest, X_test, y_test):
    y_pred_proba = random_forest.predict_proba(np.array(X_test))
    roc = []
    for threshold in np.arange(0, 1, 0.01):
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for index in range(len(y_pred_proba)):
            if y_pred_proba[index] >= threshold:
                pred = 1
            else:
                pred = 0

            if pred == 1 and y_test[index] == 1:
                tp += 1
            elif pred == 1 and y_test[index] == 0:
                fp += 1
            elif pred == 0 and y_test[index] == 1:
                fn += 1
            else:
                tn += 1

        if (tp + fn) != 0:
            true_positive_rate = tp / (tp + fn)
        else:
            true_positive_rate = 0

        if (fp + tn) != 0:
            false_positive_rate = fp / (fp + tn)
        else:
            false_positive_rate = 0

        if (tp + fp) != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0

        if (tp + fn) != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0

        roc.append({"threshold": threshold, "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate, "precision": precision, "recall": recall})
    auc = 0
    aurpc = 0
    for index in range(len(roc) - 1):
        auc += abs(roc[index + 1]["false_positive_rate"] - roc[index]["false_positive_rate"]) * abs(
            roc[index]["true_positive_rate"] + roc[index + 1]["true_positive_rate"]) / 2
        aurpc += abs(roc[index + 1]["recall"] - roc[index]["recall"]) * abs(
            roc[index]["precision"] + roc[index + 1]["precision"]) / 2

    return auc, aurpc


def cross_validation(inputs, outputs, splits):
    predictions = []
    for i in range(splits):
        print("Started fold " + str(i))
        X_test, X_train, y_test, y_train = select_folds(i, inputs, outputs, splits)
        random_forest = RandomForest()
        random_forest.fit(np.array(X_train), np.array(y_train))
        false_negatives, false_positives, true_negatives, true_positives = train_and_test(random_forest, X_test, y_test)

        accuracy, f1_score, precision, recall, accuracy_ci_95 = compute_basic_metrics(false_negatives, false_positives, true_negatives,
                                                                      true_positives)
        auc, aurpc = compute_auc(random_forest, X_test, y_test)

        predictions.append(
            {"accuracy": accuracy, "f1_score": f1_score, "precision": precision, "recall": recall, "auc": auc,
             "aurpc": aurpc, "accuracy_ci_95": accuracy_ci_95})

        print({"accuracy": accuracy, "f1_score": f1_score, "precision": precision, "recall": recall, "auc": auc,
               "aurpc": aurpc, "accuracy_ci_95": accuracy_ci_95})
    return predictions


def compute_basic_metrics(false_negatives, false_positives, true_negatives, true_positives):
    accuracy = (true_positives + true_negatives) / (
            true_positives + true_negatives + false_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * precision * recall / (precision + recall)
    accuracy_ci_95 = 1.96 * np.sqrt((accuracy * (1 - accuracy)) /
                                    (true_positives + true_negatives + false_positives + false_negatives))
    return accuracy, f1_score, precision, recall, accuracy_ci_95


def train_and_test(random_forest, X_test, y_test):
    y_pred = random_forest.predict(np.array(X_test))
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for j in range(len(y_pred)):
        if y_pred[j] == 1 and y_test[j] == 1:
            true_positives += 1
        elif y_pred[j] == 1 and y_test[j] == 0:
            false_positives += 1
        elif y_pred[j] == 0 and y_test[j] == 1:
            false_negatives += 1
        else:
            true_negatives += 1
    return false_negatives, false_positives, true_negatives, true_positives


def select_folds(i, inputs, outputs, splits):
    X_train = []
    y_train = []
    for j in range(splits):
        if i != j:
            X_train.extend(inputs[j])
            y_train.extend(outputs[j])
    X_test = inputs[i]
    y_test = outputs[i]
    return X_test, X_train, y_test, y_train


def main():
    X, y = load_data('heart.csv')
    positives_number = y[y == 1].shape[0]
    negatives_number = y[y == 0].shape[0]
    splits = 5
    inputs, outputs = split_into_folds(X, y, splits)
    predictions = cross_validation(inputs, outputs, splits)

    metrics = {"accuracy": 0.0, "f1_score": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0, "aurpc": 0.0, "accuracy_ci_95": 0.0}
    for prediction in predictions:
        for key, value in prediction.items():
            metrics[key] += value

    for key in metrics:
        metrics[key] /= splits

    print('\n\n')
    print('Average metrics for 5 folds:')
    print(metrics)


if __name__ == '__main__':
    main()
