import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

def train_model():
    features = np.load("features.npy")
    results = np.load("results.npy")
    model = DecisionTreeRegressor()
    model.fit(features, results)
    
    with open('decision_tree_model.pkl', 'wb') as file:
        pickle.dump(model, file)
    # plt.figure(figsize=(12, 8))
    # plot_tree(model, feature_names=[
    #     "Age", "Gender", "Annual Income", "Marital Status", "Number of Dependents",
    #     "Education", "Employment Status", "Health Score", "Location", "Insurance Plan",
    #     "Previous Claims", "Months Since Last Claim", "Credit Score", "Insurance Duration",
    #     "Policy Start Date", "Customer Feedbacks", "Smoking", "Exercise Frequency", "Property Type"
    # ], filled=True)
    # plt.show()

def test_model():
    with open('decision_tree_model.pkl', 'rb') as file:
        model : DecisionTreeRegressor = pickle.load(file)
    
    test_f = np.load("test_f.npy")
    predictions = model.predict(test_f)
    with open("predictions.csv", "w") as file:
        file.write("id,Premium Amount\n")
        for i, prediction in enumerate(predictions):
            file.write(f"{1200000+i},{prediction}\n")