Task 5: Decision Trees and Random Forests - Heart Disease Prediction

Objective:
The objective of this task was to explore tree-based models, namely Decision Trees and Random Forests, for classification problems, understanding concepts like overfitting, ensemble learning, and feature importance.

Dataset:
The dataset used for this task is the heart.csv dataset, which contains various health parameters used to predict the presence of heart disease.

Tools and Libraries Used:
Python
Pandas: For data loading and manipulation, including one-hot encoding.
Scikit-learn: For machine learning model implementation (DecisionTreeClassifier, RandomForestClassifier, train-test split, cross_val_score) and evaluation metrics.
Matplotlib: For creating static visualizations (tree visualization, accuracy plots, feature importance plots, cross-validation plots).
Seaborn: For enhanced statistical graphics (used for bar plots).

Tree-Based Model Steps Performed:
1. Train a Decision Tree Classifier and Visualize the Tree
Loaded and preprocessed the heart.csv dataset, including one-hot encoding of categorical features.
Split the data into training and testing sets (70/30 split, stratified).
Trained a DecisionTreeClassifier with a max_depth of 3 for initial visualization.
Visualized the trained Decision Tree, showing the decision paths and feature splits.
Outcome: A basic Decision Tree model was implemented and its structure visualized, providing insight into its decision-making process.
2. Analyze Overfitting and Control Tree Depth
Trained multiple Decision Trees with varying max_depth values.
Evaluated training and testing accuracies for each depth.
Plotted the accuracies against max_depth to visually demonstrate how increasing tree complexity leads to overfitting (where training accuracy continues to rise but test accuracy plateaus or drops).
Outcome: Demonstrated the concept of overfitting in Decision Trees and how max_depth is a critical hyperparameter for controlling model complexity and preventing poor generalization.
3. Train a Random Forest and Compare Accuracy
Trained a RandomForestClassifier, an ensemble method that builds multiple decision trees and merges their predictions.
Evaluated its accuracy on the test set and compared it with the accuracy of a single Decision Tree (e.g., at max_depth=4).
Outcome: Showcased the power of ensemble learning, where Random Forests typically outperform single Decision Trees by reducing variance and improving generalization.
4. Interpret Feature Importances
Extracted and ranked feature importances from the trained RandomForestClassifier.
Visualized the top features using a bar plot.
Outcome: Identified the most influential features in predicting heart disease according to the Random Forest model, providing valuable insights into the dataset.
5. Evaluate using Cross-Validation
Performed 5-fold cross-validation on the RandomForestClassifier.
Calculated individual cross-validation scores, the mean accuracy, and the standard deviation of the scores.
Visualized the cross-validation scores across different folds.
Outcome: Demonstrated a robust method for evaluating model performance, providing a more reliable estimate of generalization ability compared to a single train-test split and confirming the model's consistent performance.

Visualizations:
The repository includes the following generated plots:
decision_tree_classifier_max_depth_3.png: Visualization of the initial Decision Tree.
dt_accuracy_vs_depth.png: Plot showing Decision Tree accuracy on train and test sets against tree depth, illustrating overfitting.
rf_feature_importances.png: Bar plot of feature importances from the Random Forest model.
rf_cross_validation_scores.png: Bar plot showing accuracy scores for each fold of cross-validation.

Conclusion:
This task provided a comprehensive hands-on experience with tree-based models, covering the fundamentals of Decision Trees, the benefits of ensemble learning with Random Forests, strategies for mitigating overfitting, and robust model evaluation techniques like cross-validation and feature importance analysis.

