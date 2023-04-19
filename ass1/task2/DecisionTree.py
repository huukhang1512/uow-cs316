from tkinter import Y
import numpy as np
import math


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left=None,
        right=None,
        value=None,
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=5, n_features=None, criteria='information_gain'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.criteria = criteria
        self.root = None

    @staticmethod
    def _shannon_entropy(y):
        num_entries = len(y)
        label_counts = np.bincount(y)  # Count the occurences of each
        shannon_ent = 0.0

        for label in label_counts:
            prob = float(label_counts[label]) / num_entries
            shannon_ent -= prob * np.log2(prob)
        return shannon_ent

    @staticmethod
    def _information_gain(self, X, y, threshold):
        parent_entropy = self._entropy(y)

        left_idxs, right_idxs = self._split(X, threshold)

        num_entries = len(y)
        num_left_entries = len(left_idxs)
        num_right_entries = len(right_idxs)

        left_entropy = self._entropy(y.iloc[left_idxs])
        right_entropy = self._entropy(y.iloc[right_idxs])

        child_entropy = (num_left_entries/num_entries) * left_entropy + \
            (num_right_entries/num_entries) * right_entropy

        return parent_entropy - child_entropy

    @staticmethod
    def _gini(self, X, y):
        pass

    @staticmethod
    def _gain_ratio(self, X, y):
        pass

    def _calculate_criteria(self, X, y, threshold):
        match self.criteria:
            case 'information-gain':
                return self._information_gain(X, y, threshold)

    # Split the tree base on the given feature index:

    def _split(self, X, threshold):
        # group indices that are smaller than the split threshold
        left_idxs = np.argwhere(X <= threshold)

        # group indices that are larger than the threshold
        right_idxs = np.argwhere(X > threshold)
        return left_idxs, right_idxs

    # Function to build the tree recursively
    def _generate_tree(self, X, y, cur_depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Early stopping
        if (cur_depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            hist = np.bincount(y)
            most_common_label = np.argmax(hist)
            return Node(value=most_common_label)

        # Creating a random group for each recursive tree splitting
        random_feature = np.random.choice(
            n_features, self.n_features, replace=False)

        # Find best split from root
        best_split_idx, best_split_threshold = self._get_best_split(
            X, y, random_feature)

        # Recursively build the left subtree and right subtree
        left_idxs, right_idxs = self._split(
            X.iloc[:, best_split_idx], best_split_threshold)

        left = self._generate_tree(
            X.iloc[left_idxs, :], y.iloc[left_idxs], cur_depth + 1)
        right = self._generate_tree(
            X.iloc[right_idxs, :], y.iloc[right_idxs], cur_depth + 1)

        return Node(best_split_idx, best_split_threshold, left, right)

    def _get_best_split(self, X, y, features):
        max_criteria = float('-inf')
        best_split_idx = None
        best_split_threshold = None
        for feature_idx in features:
            X_feature = X.iloc[:, feature_idx]
            thresholds = np.unique(X_feature)

            for threshold in thresholds:
                # Calculate the criteria for splitting
                criteria = self._calculate_criteria(X_feature, y, threshold)
                # update the gain
                if (criteria > max_criteria):
                    max_criteria = criteria
                    best_split_idx = feature_idx
                    best_split_threshold = threshold

        return best_split_idx, best_split_threshold

    def _dfs(self, x, node: Node):
        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._dfs(x, node.left)
        return self._dfs(x, node.right)

    def fit(self, X, y):
        self.root = self._generate_tree(X, y)

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._dfs(x, self.root))

        return np.array[predictions]
