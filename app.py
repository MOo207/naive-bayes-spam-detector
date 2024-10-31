# Import necessary libraries
import pandas as pd
import re
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from multiprocessing import Pool, cpu_count
import numpy as np  # Import numpy
# Download stopwords
nltk.download('stopwords')

# Load dataset
spam_data = pd.read_csv('spam.csv', encoding='latin-1')

# Select only the relevant columns
spam_data = spam_data[['Category', 'Message']]

# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Preprocessing function to clean and stem text
def preprocess_text(message):
    message = re.sub('[^a-zA-Z]', ' ', message).lower()
    tokens = [stemmer.stem(word) for word in message.split() if word not in stop_words]
    return tokens

# Function to count word frequencies in parallel
def count_words(messages):
    word_counts = defaultdict(int)
    for message in messages:
        for word in message:
            word_counts[word] += 1
    return word_counts

# Function to split data into chunks
def chunkify(data, num_chunks):
    chunk_size = len(data) // num_chunks
    return [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

# Parallel word counting
def parallel_word_count(messages):
    # Determine the number of processes to run in parallel (using available CPU cores)
    num_cores = cpu_count()

    # Chunkify the data into smaller parts for each process
    message_chunks = chunkify(list(messages['Processed_Message']), num_cores)

    # Create a pool of workers and count words in parallel
    with Pool(num_cores) as pool:
        results = pool.map(count_words, message_chunks)

    # Merge the results from all processes
    total_word_counts = Counter()
    for result in results:
        total_word_counts.update(result)

    return total_word_counts

# Function to calculate conditional probabilities for a given message
def calculate_spam_ham_probabilities(message, spam_word_counts, ham_word_counts, total_spam_words, total_ham_words, prior_spam, prior_ham):
    message_tokens = preprocess_text(message)

    # Initialize log probabilities
    spam_prob = np.log(prior_spam)
    ham_prob = np.log(prior_ham)

    # Laplace smoothing parameter
    alpha = 1

    for word in message_tokens:
        # Calculate probability of the word in spam
        spam_word_prob = (spam_word_counts[word] + alpha) / (total_spam_words + alpha * len(spam_word_counts))
        # Calculate probability of the word in ham
        ham_word_prob = (ham_word_counts[word] + alpha) / (total_ham_words + alpha * len(ham_word_counts))

        # Add the log probabilities for each word
        spam_prob += np.log(spam_word_prob)
        ham_prob += np.log(ham_word_prob)

    return spam_prob, ham_prob

# Function to predict whether a message is spam or ham
def predict(message, spam_word_counts, ham_word_counts, total_spam_words, total_ham_words, prior_spam, prior_ham):
    spam_prob, ham_prob = calculate_spam_ham_probabilities(message, spam_word_counts, ham_word_counts, total_spam_words, total_ham_words, prior_spam, prior_ham)

    if spam_prob > ham_prob:
        return 'spam'
    else:
        return 'ham'

# Main block for running parallel processing
if __name__ == '__main__':
    # Apply preprocessing to the Message column and split data into spam and ham
    spam_data['Processed_Message'] = spam_data['Message'].apply(preprocess_text)
    spam_messages = spam_data[spam_data['Category'] == 'spam']
    ham_messages = spam_data[spam_data['Category'] == 'ham']

    # Count word frequencies for spam and ham messages in parallel
    spam_word_counts = parallel_word_count(spam_messages)
    ham_word_counts = parallel_word_count(ham_messages)

    # Calculate total words in spam and ham messages
    total_spam_words = sum(spam_word_counts.values())
    total_ham_words = sum(ham_word_counts.values())

    # Calculate prior probabilities
    num_spam = len(spam_messages)
    num_ham = len(ham_messages)
    total_messages = len(spam_data)
    prior_spam = num_spam / total_messages
    prior_ham = num_ham / total_messages

    # Continuous loop to get user input and make predictions
    continue_loop = True
    while continue_loop:
        user_input = input('Enter a message: ')
        prediction = predict(user_input, spam_word_counts, ham_word_counts, total_spam_words, total_ham_words, prior_spam, prior_ham)
        print('Prediction:', prediction)
        continue_input = input('Do you want to continue? (yes/no): ')
        if continue_input.lower() != 'yes':
            continue_loop = False
