# Spam Detection Using Naive Bayes on Kaggle Dataset

This project is a basic implementation of Naive Bayes theory for spam detection. It utilizes a dataset from Kaggle to classify messages as "spam" or "ham" based on word frequencies, a statistical approach that does not require complex machine learning models or libraries.

## Dataset

The dataset used is the **SMS Spam Collection Dataset**, which can be downloaded from Kaggle:
[Kaggle Spam Emails Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/abdallahwagih/spam-emails))

## Project Overview

This project demonstrates spam detection using Bayesian inference, focusing on conditional probability. The process includes:
- **Preprocessing**: Removing non-alphabetic characters, stemming words, and excluding common English stopwords.
- **Parallel Processing**: Counting word frequencies in parallel for efficiency, especially useful with larger datasets.
- **Naive Bayes Probability Calculation**: Each word's probability of appearing in spam or ham messages is calculated using Laplace smoothing to handle unseen words.

The statistical approach used in this project is a simple, effective method that can solve text classification issues without complex machine learning models.

## Dependencies

- **Pandas** for data handling
- **NumPy** for numerical operations
- **NLTK** for natural language processing, including stopwords and stemming
- **multiprocessing** for parallel processing

Install the required packages with:
```bash
pip install pandas numpy nltk
```

## How to Use

1. Download the Kaggle dataset and place it in the project directory as `spam.csv`.
2. Run the script. It will load, preprocess, and count word frequencies in the dataset.
3. The script enters a loop where you can input any message and receive a spam/ham prediction based on Naive Bayes theory.

### Running the Code
```bash
python spam_detection.py
```

## Example Usage

After launching, the script will prompt for input messages. Enter any message to check if it is classified as spam or ham.

```plaintext
Enter a message: "Free entry to win a prize!"
Prediction: spam
Do you want to continue? (yes/no): yes
```

![image](https://github.com/user-attachments/assets/edd42c61-cf38-4e88-bc22-164e51383ff3)


## Notes

- This solution leverages basic statistical probability (Bayesian inference) without relying on a specific machine learning model, making it lightweight and effective for simple classification tasks.
- This project is meant to illustrate how statistical methods can be applied to text data for classification without additional ML complexity.
