import numpy as np
from sklearn.utils import resample

from decision_tree import DecisionTree


class RandomForest:
    def __init__(self, n_trees=100, max_depth=15):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sampling (sampling with replacement)
            X_subset, y_subset = resample(X, y, replace=True)

            # Train a decision tree on the subset of data
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_subset, y_subset)

            # Add the trained tree to the forest
            self.trees.append(tree)

    def predict(self, X):
        # Make predictions for each tree and aggregate the results
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        # Use majority voting to get the final predictions
        ensemble_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)
        return ensemble_predictions
