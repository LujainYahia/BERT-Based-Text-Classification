# BERT-Based-Text-Classification
Here's a detailed README file for the provided code. This README outlines the setup, usage, and functionalities of the script for training and evaluating a BERT-based text classifier.

---

# BERT-Based Text Classification

This repository contains a script to train a BERT-based text classifier using the BBC News dataset. The script includes loading the dataset, preprocessing, defining the model architecture, training, and evaluating the model.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [License](#license)

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/bert-text-classification.git
   cd bert-text-classification
   ```

2. **Install the Dependencies**

   ```bash
   pip install transformers pandas torch numpy tqdm matplotlib
   ```

## Dataset Preparation

1. **Download the Dataset**

   Download the BBC News dataset and place it in the project directory. Ensure the dataset file is named `bbc-text.csv`.

2. **Load and Explore the Dataset**

   The dataset is loaded into a pandas DataFrame, and a bar plot is generated to visualize the distribution of categories.

   ```python
   df = pd.read_csv("bbc-text.csv")
   df.groupby(['category']).size().plot.bar()
   ```

## Model Architecture

The model architecture is based on BERT with a linear layer for classification. The `BertClassifier` class defines the model:

```python
class BertClassifier(nn.Module):

    def __init__(self, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 5)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids= input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
```

## Training

The `train` function trains the model on the training data and evaluates it on the validation data for each epoch. It uses CrossEntropyLoss and the Adam optimizer.

```python
def train(model, train_data, val_data, learning_rate, epochs):
    # Implementation details...
```

## Evaluation

The `evaluate` function evaluates the trained model on the test data and prints the test accuracy.

```python
def evaluate(model, test_data):
    # Implementation details...
```

## Usage

1. **Prepare the Data**

   Split the dataset into training, validation, and test sets.

   ```python
   np.random.seed(112)
   df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                        [int(.8*len(df)), int(.9*len(df))])
   ```

2. **Set Hyperparameters**

   Define the number of epochs and learning rate.

   ```python
   EPOCHS = 5
   LR = 1e-6
   ```

3. **Initialize and Train the Model**

   Initialize the `BertClassifier` model and train it.

   ```python
   model = BertClassifier()
   train(model, df_train, df_val, LR, EPOCHS)
   ```

4. **Evaluate the Model**

   Evaluate the trained model on the test data.

   ```python
   evaluate(model, df_test)
   ```

