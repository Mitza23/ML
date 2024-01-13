import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = np.unique(y)

        # If only one class in the data or max depth is reached, return the class
        if len(unique_classes) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return unique_classes[0]

        # If no features left, return the majority class
        if num_features == 0:
            return np.bincount(y).argmax()

        # Find the best feature and threshold to split the data
        best_feature, best_threshold = self._find_best_split(X, y)

        # If no split is found, return the majority class
        if best_feature is None:
            return np.bincount(y).argmax()

        # Split the data based on the best feature and threshold
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        # Create a decision node
        decision_node = {
            'feature_index': best_feature,
            'threshold': best_threshold,
            'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
            'right': self._build_tree(X[right_mask], y[right_mask], depth + 1)
        }

        return decision_node

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        if num_samples <= 1:
            return None, None

        # Calculate the Gini impurity for the current node
        current_gini = self._calculate_gini(y)

        best_gini = 1.0
        best_feature = None
        best_threshold = None

        for feature_index in range(num_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                # Split the data
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                # Skip splits with empty nodes
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Calculate the Gini impurity for the split
                gini_left = self._calculate_gini(y[left_mask])
                gini_right = self._calculate_gini(y[right_mask])

                # Calculate the weighted average Gini impurity
                weighted_gini = (np.sum(left_mask) / num_samples) * gini_left + \
                                (np.sum(right_mask) / num_samples) * gini_right

                # Update the best split if the current one is better
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, y):
        num_samples = len(y)
        if num_samples == 0:
            return 0.0

        # Calculate the Gini impurity
        class_counts = np.bincount(y)
        probabilities = class_counts / num_samples
        gini = 1.0 - np.sum(probabilities ** 2)

        return gini

    def predict(self, X):
        return np.array([self._predict_tree(sample, self.tree) for sample in X])

    def _predict_tree(self, x, tree):
        if isinstance(tree, dict):
            # If a decision node, recursively traverse the tree
            if x[tree['feature_index']] <= tree['threshold']:
                return self._predict_tree(x, tree['left'])
            else:
                return self._predict_tree(x, tree['right'])
        else:
            # If a leaf node, return the class label
            return tree
