**Project Purpose**

**Project 2: Data Classification Using A**I is a supervised machine learning project built to demonstrate the complete pipeline of training, testing, and validating a classification model — the foundational skill required before moving into more advanced AI techniques like neural networks and deep learning.

**Why this project matters**
**Moving beyond rule-based logic** — Instead of writing manual if/else rules to make decisions, this project shows how a machine can learn patterns directly from historical data and use them to classify new, unseen information. This is the core idea behind supervised learning: "We do not write the rules — we provide history, and the machine derives the logic."
**Building a foundation for advanced AI** — Before tackling neural networks or computer vision, it's essential to master the basic ML workflow: loading data, splitting it into training/testing sets, scaling features, training a model, and evaluating it properly.
**End-to-end ML workflow** — The project covers the full pipeline:
Loading and exploring a dataset
Data cleaning and feature engineering
Splitting data into training (80%) and testing (20%) sets
Feature scaling (StandardScaler)
Training a K-Nearest Neighbors (KNN) classification model
Hyperparameter tuning (finding the optimal K value using the elbow method)
Evaluating performance using Accuracy, Confusion Matrix, and F1 Score — not just raw accuracy, since accuracy alone can be misleading on imbalanced datasets
**Applying it to real-world data**— I first validated the pipeline on the classic Iris dataset (achieving 96.67% accuracy, since flower measurements strongly correlate with species). I then applied the same pipeline to a real-world e-commerce orders dataset to predict order status.
**Key takeaway / what I learned**

Not every dataset contains a learnable pattern. When I applied the model to the e-commerce dataset, accuracy dropped to ~25% (close to random guessing for 5 classes). Rather than treating this as a failure, I investigated further and found that the target variable (OrderStatus) had no meaningful statistical relationship with the available features — the averages for Quantity, UnitPrice, and TotalPrice were nearly identical across all order statuses.

This reinforced an important real-world ML lesson: a low-performing model isn't always a coding or algorithm problem — sometimes it correctly reflects that the data itself lacks predictive signal. Recognizing this distinction, rather than blindly trying to "fix" the model, is a core skill for any AI/ML engineer.

**Tech stack**

Python, pandas, scikit-learn, matplotlib, seaborn — KNN Classifier, StandardScaler, Confusion Matrix, F1 Score evaluation.
