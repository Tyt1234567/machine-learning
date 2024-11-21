import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class TreeNode:
    def __init__(self, gini, num_samples, num_samples_class, class_label):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_class = num_samples_class
        self.class_label = class_label
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None

        mk = [np.sum(y == k) for k in range(self.num_classes)]
        best_gini = 1.0 - sum((n / m) ** 2 for n in mk)
        best_idx, best_thr = None, None

        for idx in range(self.num_features):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.num_classes
            num_right = mk.copy()

            for i in range(1, m):
                k = classes[i - 1]
                num_left[k] += 1
                num_right[k] -= 1

                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.num_classes))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.num_classes))
                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == k) / m) ** 2 for k in range(self.num_classes))

    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.num_features = X.shape[1]
        self.tree = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        num_samples_class = [np.sum(y == k) for k in range(self.num_classes)]
        class_label = np.argmax(num_samples_class)

        node = TreeNode(
            gini=self.gini(y),
            num_samples=y.size,
            num_samples_class=num_samples_class,
            class_label=class_label,
        )

        if depth < self.max_depth:
            idx, thr = self.best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X):
        return [self.predict_helper(x) for x in X]

    def predict_helper(self, x):
        node = self.tree
        while node.left:
            if x[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.class_label

if __name__ == "__main__":
    # Load data
    iris = load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Decision tree classifier...")
    tree_clf = DecisionTreeClassifier(max_depth=3)
    tree_clf.fit(X_train, y_train)

    print("Prediction...")
    y_pred = tree_clf.predict(X_test)
    tree_clf_acc = accuracy_score(y_test, y_pred)
    print("Test set accuracy:", tree_clf_acc)
