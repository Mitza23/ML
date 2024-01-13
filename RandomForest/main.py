import numpy as np
import pandas as pd

from random_forest import RandomForest

if __name__ == '__main__':
    df = pd.read_csv('heart.csv')
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    splits = 10
    inputs = []
    outputs = []
    for split in range(splits):
        inputs.append(X[split * 30:(split + 1) * 30])
        outputs.append(y[split * 30:(split + 1) * 30])

    for i in range(splits):
        random_forest = RandomForest(n_trees=100, max_depth=15)
        X_train = []
        y_train = []
        for j in range(splits):
            if i != j:
                X_train.extend(inputs[j])
                y_train.extend(outputs[j])
        X_test = inputs[i]
        y_test = outputs[i]
        random_forest.fit(np.array(X_train), np.array(y_train))
        y_pred = random_forest.predict(np.array(X_test))
        correct = 0
        wrong = 0
        for j in range(len(y_pred)):
            if y_pred[j] == y_test[j]:
                correct += 1
            else:
                wrong += 1
        print("Accuracy for split", i + 1, ":", correct / (correct + wrong))



