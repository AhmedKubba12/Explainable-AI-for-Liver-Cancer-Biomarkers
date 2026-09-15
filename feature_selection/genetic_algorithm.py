import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score
import warnings
warnings.filterwarnings("ignore")
data= pd.read_csv("data/selected_genes_500_xg.csv")
target = data.loc[:, "Symbol"]

target = np.where(target == "sl", 0,
         np.where(target == "lgdn", 1,
         np.where(target == "hgdn", 2,
         np.where(target == "ehcc", 3, 
         np.where(target == "phcc", 4,0)))))

data_predictors = data.loc[:, data.columns != 'Symbol']
predictor_names = data_predictors.columns
# Split data
x_train, x_test, y_train, y_test = train_test_split(data_predictors, target, test_size=0.3,random_state=0)

def generate_random_individuals(population_size, num_features, min_features, max_features):
    individuals = np.zeros((population_size, num_features))
    for i in range(population_size):
        num_ones = np.random.randint(min_features, max_features + 1)
        ones_indices = np.random.choice(num_features, num_ones, replace=False)
        individuals[i, ones_indices] = 1
    return individuals

def train_model(x_train, x_test, y_train, y_test, predictor_names):
    # Select the specified predictor features
    x_train = x_train.loc[:, predictor_names]
    x_test = x_test.loc[:, predictor_names]
    
    # Define the Random Forest model with a set of hyperparameters
    mdl = RandomForestClassifier(
        n_estimators=300,          # Number of trees in the forest
        max_depth=15,              # Maximum depth of the trees
        min_samples_split=5,       # Minimum number of samples required to split an internal node
        min_samples_leaf=2,        # Minimum number of samples required to be at a leaf node
        max_features='sqrt',       # Number of features to consider when looking for the best split
        bootstrap=True,            # Whether bootstrap samples are used when building trees
                    # Seed for random number generation
    )
    
    # Train the model
    mdl.fit(x_train, y_train)
    # Make predictions
    y_hat = mdl.predict(x_test)
    # Calculate precision
    prec = precision_score(y_test, y_hat, average='weighted')  # Use 'weighted' for multiclass scenarios
    return prec
def choose_parents(population, accuracy, elite_percent=0.1):
    # Get elite of top 10% which doesn't mutate
    elite_num = int(round((elite_percent * population.shape[0])))
    elite_num = max(elite_num, 2)  # Ensure at least 2 elite individuals
    ind_ac = np.argsort(-accuracy)
    top_perc = ind_ac[:elite_num]
    elite_population = population[top_perc, :]
    
    # Normalize accuracy to obtain weights for roulette wheel selection
    weight_norm = accuracy / accuracy.sum()
    weight_comu = weight_norm.cumsum()
    # Roulette wheel selection   
    num_parents_wo_elite = population.shape[0] - elite_num
    parents_wo_elite = np.empty([num_parents_wo_elite, population.shape[1]])
    for count in range(num_parents_wo_elite):
        b = weight_comu[-1]
        rand_num = np.random.uniform(0, b)
        indices = np.searchsorted(weight_comu, rand_num)
        parents_wo_elite[count, :] = population[indices, :]
    
    parents = np.concatenate((elite_population, parents_wo_elite), axis=0)
    return parents


def one_point_crossover(parents, elite_percent, mutation_probability, min_features, max_features):
    num_parents, num_features = parents.shape
    elite_num = int(round(((elite_percent * num_parents) // 2) * 2))
    crossover_population = np.zeros((num_parents, num_features), dtype=int)
    crossover_population[0:elite_num, :] = parents[0:elite_num, :]
    
    for i in range(int((num_parents - elite_num) / 2)):
        n = 2 * i + elite_num
        parents_couple = parents[n:n + 2, :]
        crossover_point = np.random.randint(1, num_features - 1)
        crossover_population[n, :] = np.concatenate([parents_couple[0, :crossover_point], parents_couple[1, crossover_point:]])
        crossover_population[n + 1, :] = np.concatenate([parents_couple[1, :crossover_point], parents_couple[0, crossover_point:]])
    
    for j in range(crossover_population.shape[0]):
        feature_sum = np.sum(crossover_population[j, :])
        if feature_sum > max_features:
            excess = feature_sum - max_features
            indices = np.where(crossover_population[j, :] == 1)[0]
            to_turn_off = np.random.choice(indices, size=excess, replace=False)
            crossover_population[j, to_turn_off] = 0
        elif feature_sum < min_features:
            missing = min_features - feature_sum
            indices = np.where(crossover_population[j, :] == 0)[0]
            to_turn_on = np.random.choice(indices, size=missing, replace=False)
            crossover_population[j, to_turn_on] = 1
    
    
    # Mutation
    child_row = crossover_population.shape[0]
    child_col = crossover_population.shape[1]
    num_mutations = round(child_row * child_col * mutation_probability)
    
    for jj in range(num_mutations):
        ind_row = np.random.randint(0, child_row)
        ind_col = np.random.randint(0, child_col)
        if (crossover_population[ind_row, ind_col] == 0 and 
            np.sum(crossover_population[ind_row, :]) < max_features):
            crossover_population[ind_row, ind_col] = 1
        elif (crossover_population[ind_row, ind_col] == 1 and 
              np.sum(crossover_population[ind_row, :]) >= min_features + 1):
            crossover_population[ind_row, ind_col] = 0
    
    return crossover_population

# Hyperparameters
# Updated Hyperparameters
num_features = data_predictors.shape[1]
min_features = 2            # Minimal number of features in a subset
population_size = 25       # Larger population size
max_iterations = 2          # More iterations for optimization
elite_percent = 0.1         # Adjust elite percentage
mutation_probability = 0.1 # Lower mutation rate
max_features = 15            # Targeting 5 features for optimization 

import numpy as np

num_runs = 1
for run in range(1, num_runs + 1):
    print(f"\n====== Run {run} =======\n")
    
  

    # Step 1: Generate initial population
    population = generate_random_individuals(population_size, num_features, min_features, max_features)
    
    # Step 2: Evaluate initial population
    accuracy = np.zeros(population_size)
    predictor_names = data_predictors.columns

    for i in range(population_size):
        predictor_names_i = predictor_names[population[i, :] == 1]
        accuracy[i] = train_model(x_train, x_test, y_train, y_test, predictor_names_i)

    # Store best accuracy from the first generation
    best_acc_i = np.zeros(max_iterations)
    best_acc_i[0] = max(accuracy)

    # Main loop: Generations 1 to max_iterations - 1
    for gen in range(1, max_iterations):
        print(f'Begin iteration num {gen + 1}/{max_iterations}')
        
        # Step 3: Select parents based on accuracy
        parents = choose_parents(population, accuracy, elite_percent)
        
        # Step 4: Create new generation through crossover and mutation
        children = one_point_crossover(parents, elite_percent, mutation_probability, min_features, max_features)
        population = children

        # Step 5: Evaluate new population
        for ind in range(population_size):
            predictor_names_ind = predictor_names[population[ind, :] == 1]
            accuracy[ind] = train_model(x_train, x_test, y_train, y_test, predictor_names_ind)

        # Store best accuracy for this generation
        best_acc_i[gen] = max(accuracy)

    # Step 6: Identify the best individual from the final generation
    ind_max_acc = np.argmax(accuracy)
    best_features = population[ind_max_acc, :]
    best_feature_names = predictor_names[best_features == 1]

    # Train the final model with the best features
    x_train_final = x_train.loc[:, best_feature_names]
    x_test_final = x_test.loc[:, best_feature_names]
    final_model = RandomForestClassifier(
        n_estimators=300, 
        max_depth=15, 
        min_samples_split=5, 
        min_samples_leaf=2, 
        max_features='sqrt', 
        bootstrap=True
    )
    final_model.fit(x_train_final, y_train)
    y_pred_final = final_model.predict(x_test_final)
    final_accuracy = accuracy_score(y_test, y_pred_final)

    # Print the results
    print("Best feature subset:")
    print("===================")
    for i, feature in enumerate(best_feature_names, start=1):
        print(f"{i}. {feature}")
        print("===================")
    print(f"Best Accuracy: {final_accuracy}") 

