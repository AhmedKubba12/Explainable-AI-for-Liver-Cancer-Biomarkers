import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import numpy as np

# Step 1: Load the dataset
file_path = 'data/selected_genes_500_xg.csv'  # Update with your actual file path
data = pd.read_csv(file_path)

# Step 2: Separate features (X) and target (y)
X = data.drop(columns=["Symbol"])  # Assuming "Symbol" is the target column
y = data["Symbol"]

# Step 3: Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Step 5: Define the ACO parameters
num_ants = 1
num_features = X_train.shape[1]
num_iterations = 20
alpha = 1.0  # Influence of pheromone
beta = 2.0  # Influence of heuristic
evaporation_rate = 0.01
pheromone_intensity = 1.5

# Initialize pheromone levels
pheromones = np.ones(num_features)

# Heuristic information (you might want to initialize it more intelligently based on feature importance)
heuristic_info = np.ones(num_features)

# Step 6: Define the fitness function for ACO
def evaluate_features(selected_indices):
    # Train an XGBoost classifier on the selected features
    clf = XGBClassifier(eval_metric='mlogloss')

    # Perform 3-fold cross-validation
    scores = cross_val_score(clf, X_train.iloc[:, selected_indices], y_train, cv=3, scoring='accuracy')

    # We want to minimize 1 - accuracy (i.e., maximize accuracy)
    return 1 - scores.mean()

# Step 7: Run the ACO algorithm
best_score = float('inf')
best_features = []

for iteration in range(num_iterations):
    all_solutions = []
    all_scores = []

    for ant in range(num_ants):
        # Select features based on pheromone and heuristic values
        probabilities = (pheromones ** alpha) * (heuristic_info ** beta)
        probabilities /= probabilities.sum()

        selected_indices = np.random.choice(num_features, 15, replace=False, p=probabilities)  # Select 20 features

        # Evaluate the selected features
        score = evaluate_features(selected_indices)

        all_solutions.append(selected_indices)
        all_scores.append(score)

        # Update pheromone trails
        if score < best_score:
            best_score = score
            best_features = selected_indices

    # Update pheromones based on solutions found
    for i in range(num_features):
        pheromones[i] *= (1 - evaporation_rate)  # Evaporation

    for solution, score in zip(all_solutions, all_scores):
        for feature in solution:
            pheromones[feature] += pheromone_intensity / score  # Update pheromones

    print(f"Iteration {iteration + 1}/{num_iterations}, Best score: {1 - best_score}")

# Step 8: Display the best features found and the best score
selected_feature_names = X_train.columns[best_features].tolist()
print(f"Selected {len(best_features)} features from the dataset.")
print(f"Selected feature names: {selected_feature_names}")
print(f"Best cross-validation accuracy: {1 - best_score}")

# Step 9: Train the final XGBoost model with the selected features
clf_final = XGBClassifier(eval_metric='mlogloss')
clf_final.fit(X_train.iloc[:, best_features], y_train)

# Step 10: Test the accuracy on the test set
test_accuracy = clf_final.score(X_test.iloc[:, best_features], y_test)
print(f"Test accuracy with selected features: {test_accuracy}")
