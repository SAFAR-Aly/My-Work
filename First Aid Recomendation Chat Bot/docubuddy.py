import nltk
import json
import numpy as np
import tensorflow as tf
import random
from nltk.stem.lancaster import LancasterStemmer
import os

# Initialize the Lancaster Stemmer
stemmer = LancasterStemmer()

# Function to compile the dataset
def recompile_dataset(data_file):
    with open(data_file) as file:
        data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))

    labels = sorted(labels)

    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = np.array(training)
    output = np.array(output)

    return training, output, words, labels, data

# Recompile the dataset
data_file = "C:\Mini Project\DataSet\intents.json"
training, output, words, labels, data = recompile_dataset(data_file)

unique_words = set(words)
num_unique_words = len(unique_words)

# Check if the model file exists, and load it if it does
model_file = "model_1.keras"
if os.path.exists(model_file):
    model = tf.keras.models.load_model(model_file)
else:
    # Define your model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(231,)),  # Specify the input shape
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(8),
        tf.keras.layers.Dense(len(output[0]), activation="softmax")
    ])

    # Compile and train your model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(training, output, epochs=500, batch_size=8)

    # Save your trained model
    model.save(model_file)

# Function to convert user input into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    words = [stemmer.stem(w.lower()) for w in words]

    for w in s.split():
        if w in words:
            bag[words.index(w)] = 1

    return np.array(bag).reshape(1, len(words))

# Chat function
def chat(data):
    print("Start Talking with the DocuBuddy (type quit to stop!)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict(bag_of_words(inp, words))[0]
        results_index = np.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.3:
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    for response in responses:
                        print("DocuBuddy:", response)
                    print("\n")  # Add two lines of space after each set of responses
        else:
            print("DocuBuddy: I didn't get that, try again")
# Start the chat
chat(data)
