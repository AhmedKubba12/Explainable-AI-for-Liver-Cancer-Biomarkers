import pandas as pd
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Load the CSV file (replace the path with your file path if needed)
data = pd.read_csv('data/selected_genes_500_xg.csv')

# Separate the features and the target variable (label)
X = data.drop(columns=['Symbol'])  # All columns except 'Symbol' are features
y = data['Symbol']  # 'Symbol' is the target variable

# Get the feature names
feature_names = X.columns.tolist()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data (helps with classifier performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the Grey Wolf Optimizer class for feature selection
class GWOFeatureSelection:
    def __init__(self, iterations, pack_size, n_features, target_feature_count=5):
        self.iterations = iterations
        self.pack_size = pack_size
        self.n_features = n_features
        self.target_feature_count = target_feature_count

    # Initialize a wolf with random feature selection (binary vector with exactly target_feature_count features selected)
    def wolf(self):
        wolf = [0] * self.n_features
        selected_indices = random.sample(range(self.n_features), self.target_feature_count)
        for idx in selected_indices:
            wolf[idx] = 1
        return wolf

    # Generate a wolf pack
    def pack(self):
        return [self.wolf() for _ in range(self.pack_size)]

    # Calculate fitness (accuracy) for a wolf (subset of features)
    def fitness(self, wolf):
        selected_features = [i for i in range(len(wolf)) if wolf[i] == 1]
        if len(selected_features) != self.target_feature_count:
            return 0  # Ensure exactly target_feature_count features are selected

        X_train_fs = X_train_scaled[:, selected_features]
        X_test_fs = X_test_scaled[:, selected_features]
        # Model to validate
        clf = SVC(kernel='linear', random_state=42)
        clf.fit(X_train_fs, y_train)
        predictions = clf.predict(X_test_fs)
        return accuracy_score(y_test, predictions)

    # Main loop for GWO optimization
    def hunt(self):
        wolf_pack = self.pack()
        pack_fit = sorted([(self.fitness(wolf), wolf) for wolf in wolf_pack], reverse=True)

        for iteration in range(self.iterations):
            alpha, beta, delta = pack_fit[0][1], pack_fit[1][1], pack_fit[2][1]
            print(f'Iteration {iteration + 1}, Best Fitness: {self.fitness(alpha)}')

            a = 2 * (1 - iteration / self.iterations)

            for i in range(self.pack_size):
                if wolf_pack[i] in [alpha, beta, delta]:
                    continue
                new_position = []
                for j in range(self.n_features):
                    r1, r2 = random.random(), random.random()
                    A1 = a * (2 * r1 - 1)
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - wolf_pack[i][j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = random.random(), random.random()
                    A2 = a * (2 * r1 - 1)
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - wolf_pack[i][j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = random.random(), random.random()
                    A3 = a * (2 * r1 - 1)
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - wolf_pack[i][j])
                    X3 = delta[j] - A3 * D_delta

                    X = (X1 + X2 + X3) / 3
                    new_position.append(X)

                # Apply hard constraint to ensure exactly target_feature_count features are selected
                thresholded_position = [0] * self.n_features
                selected_indices = sorted(range(len(new_position)), key=lambda k: new_position[k], reverse=True)[:self.target_feature_count]
                for idx in selected_indices:
                    thresholded_position[idx] = 1
                wolf_pack[i] = thresholded_position

            pack_fit = sorted([(self.fitness(wolf), wolf) for wolf in wolf_pack], reverse=True)

        return pack_fit[0][1]

# Instantiate the model with target feature count set to 5
gwo = GWOFeatureSelection(iterations=20, pack_size=200, n_features=X_train.shape[1], target_feature_count=35)

# Run the optimization
best_features = gwo.hunt()

# Count the selected features
selected_features_count = sum(best_features)
selected_features = [feature_names[i] for i, bit in enumerate(best_features) if bit == 1]

print("Number of Selected Features:", selected_features_count)
print("Selected Features:", selected_features)
