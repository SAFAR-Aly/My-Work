import speech_recognition as sr
import json
import random
import spacy

# Load spaCy's English language model
nlp = spacy.load("en_core_web_sm")

# Load your intents from the JSON data file
with open("C:\Mini Project\DataSet\intents.json") as file:
    intents = json.load(file)

# Create a function for speech recognition
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something:")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio)
        return text.lower()  # Convert recognized text to lowercase for case-insensitive matching
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))
        return None

# A function to get chatbot responses
def chatbot_response(input_text):
    doc = nlp(input_text)
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            if any(token.text in pattern.lower() for token in doc):
                return random.choice(intent["responses"])

    return "I'm sorry, I don't have a response for that."

if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "voice":
            voice_input = recognize_speech()
            if voice_input:
                user_input = voice_input
            else:
                print("Please try again or type your question.")
                continue

        response = chatbot_response(user_input)
        print("Chatbot:", response)

