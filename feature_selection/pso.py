import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and preprocess the dataset
data = pd.read_csv("data/selected_genes_500_xg.csv")
target = data["Symbol"].map({"sl": 0, "hgdn": 1, "lgdn": 2, "ehcc": 3, "phcc": 4})
data_predictors = data.drop(columns=["Symbol"])
predictor_names = data_predictors.columns

# Split data
x_train, x_test, y_train, y_test = train_test_split(data_predictors, target, test_size=0.3, random_state=0)

# Train model function
def train_model(x_train, x_test, y_train, y_test, feature_names):
    x_train = x_train[feature_names]
    x_test = x_test[feature_names]
    
    mdl = RandomForestClassifier(random_state=1)
    mdl.fit(x_train, y_train)
    y_hat = mdl.predict(x_test)
    accuracy = accuracy_score(y_test, y_hat)
    return accuracy

# Feature selection fitness
def feature_selection_fitness(features):
    features = [int(x) for x in features]
    selected_features = [i for i, x in enumerate(features) if x == 1]
    
    if not selected_features:  # Avoid empty feature set
        return 0
    
    feature_names = predictor_names[selected_features]
    accuracy = train_model(x_train, x_test, y_train, y_test, feature_names)
    return -accuracy  # PSO minimizes the fitness function

# PSO parameters
num_features = data_predictors.shape[1]
swarm_size = 10
max_iter = 10
inertia = 0.72984
c1 = 2.05
c2 = 2.05
lb = np.zeros(num_features)
ub = np.ones(num_features)

# Initialize particles
particles = np.random.randint(2, size=(swarm_size, num_features))  # Binary initialization
velocities = np.random.uniform(size=(swarm_size, num_features))
pbest_positions = particles.copy()
pbest_scores = np.array([feature_selection_fitness(p) for p in particles])
gbest_position = pbest_positions[pbest_scores.argmin()]
gbest_score = pbest_scores.min()

# PSO algorithm
for iteration in range(max_iter):
    r1 = np.random.uniform(size=(swarm_size, num_features))
    r2 = np.random.uniform(size=(swarm_size, num_features))
    velocities = (inertia * velocities + 
                  c1 * r1 * (pbest_positions - particles) + 
                  c2 * r2 * (gbest_position - particles))
    particles = particles + velocities
    particles = np.clip(particles, lb, ub).astype(int)  # Convert to binary

    # Evaluate fitness
    scores = np.array([feature_selection_fitness(p) for p in particles])

    # Update personal bests
    better_idx = scores < pbest_scores
    pbest_positions[better_idx] = particles[better_idx]
    pbest_scores[better_idx] = scores[better_idx]

    # Update global best
    if scores.min() < gbest_score:
        gbest_position = particles[scores.argmin()]
        gbest_score = scores.min()

    print(f"Iteration {iteration+1}/{max_iter}, Best Score: {-gbest_score:.4f}")

best_features = np.where(gbest_position == 1)[0]

# Select top features
def select_top_features(x_train, x_test, y_train, y_test, feature_names):
    x_train_subset = x_train[feature_names]
    x_test_subset = x_test[feature_names]

    mdl = RandomForestClassifier(random_state=1)
    mdl.fit(x_train_subset, y_train)

    importances = mdl.feature_importances_
    feature_importances = dict(zip(feature_names, importances))

    top_features = sorted(feature_importances, key=feature_importances.get, reverse=True)[:5]
    return top_features

selected_feature_names = predictor_names[best_features]
top_features = select_top_features(x_train, x_test, y_train, y_test, selected_feature_names)

# Print the results
print("Top 5 features:")
for i, feature in enumerate(top_features, start=1):
    print(f"{i}. {feature}")

import joblib

# Train and evaluate the model with the top 5 features
def evaluate_best_features_and_save_model(x_train, x_test, y_train, y_test, top_features, model_filename):
    x_train_best = x_train.loc[:, top_features]
    x_test_best = x_test.loc[:, top_features]

    # Train the model
    mdl = RandomForestClassifier(random_state=1)
    mdl.fit(x_train_best, y_train)
    y_hat = mdl.predict(x_test_best)

    # Save the model
    joblib.dump(mdl, model_filename)  # Save the model to a file
    print(f"Model saved to {model_filename}")

    accuracy = accuracy_score(y_test, y_hat)
    return accuracy

# File name to save the model
model_filename = "models/best_random_forest_model.pkl"

# Evaluate and save the model
best_accuracy = evaluate_best_features_and_save_model(x_train, x_test, y_train, y_test, top_features, model_filename)
print(f"Accuracy with the top 5 feature subset: {best_accuracy:.4f}")

