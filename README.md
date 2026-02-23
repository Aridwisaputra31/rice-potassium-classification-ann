# rice-potassium-classification-ann
This project implements an Artificial Neural Network (ANN) to predict the potassium status of rice plants using color features extracted from leaf images.
The dataset used in this repository has already been prepared and contains numerical color feature values extracted from rice leaf images captured using three different smartphone cameras.

## Dataset
The dataset is provided in this repository and includes:
1. Extracted color features 
2. Target label or potassium value
Data ready for training (no image preprocessing required)

### What This Code Does
- Loads prepared dataset
- Splits data into training and testing sets
- Trains an ANN model
- Evaluates model performance
- Outputs prediction results

#### Program Workflow
The script performs:
1️⃣ Load training and validation datasets
2️⃣ Normalize input features using StandardScaler
3️⃣ Convert class labels using OneHotEncoder
4️⃣ Build Deep ANN model (5 hidden layers + dropout)
5️⃣ Train model using Adam optimizer
6️⃣ Evaluate performance using:
Accuracy
Confusion Matrix
7️⃣ Visualize:
Training vs Validation Accuracy
Training vs Validation Loss
8️⃣ Save trained model to disk
