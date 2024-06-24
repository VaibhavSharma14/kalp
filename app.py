from flask import Flask, Response, jsonify, render_template, request, send_file, url_for
import requests
import openai
import gtts
import os
import random
import wave
import librosa
import numpy as np
import joblib
import tempfile
import soundfile as sf
from keras.models import load_model
import sounddevice as sd


openai.api_key = "sk-proj-baJwoM0j0XerdW27gOEhT3BlbkFJUnc9i1NRryAorRo7vPXM"

app = Flask(__name__)

app.static_folder = 'static'

@app.route('/generate_audio', methods=['POST'])
def text_to_speech():
    data = request.get_json()
    text = data['text']
    tts = gtts.gTTS(text=text, lang='en')
    audio_file = 'speech.mp3'
    tts.save(os.path.join(app.static_folder, audio_file))
    audio_url = url_for('static', filename=audio_file)
    return jsonify({'audio_url': audio_url})

conversation_history = []
max_context_length = 4096
def process_conversation_history(question, max_context_length):
    """
    Processes the conversation history by keeping only the relevant context.
    """
    relevant_context = [{"role": "system", "content": "You are a helpful assistant."}]
    context_length = len("You are a helpful assistant.")
    for message in reversed(conversation_history):
        message_length = len(message["content"].split())
        if context_length + message_length <= max_context_length - len(question.split()):
            relevant_context.insert(1, message)
            context_length += message_length
        else:
            break

    relevant_context.append({"role": "user", "content": question})
    return relevant_context

def ask_question(question, model="gpt-3.5-turbo"):
    """
    Asks a question to the OpenAI language model and returns the response.
    """
    conversation_history.append({"role": "user", "content": question})
    relevant_context = process_conversation_history(question, max_context_length)

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=relevant_context,
            temperature=0.7,
            n=1,
            stop=None,
        )

        conversation_history.append({"role": "assistant", "content": response.choices[0].message['content']})
        return response.choices[0].message['content']

    except openai.error.InvalidRequestError as e:
        # Handle other errors
        raise e
    

model = load_model('emotion.keras')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

behavior_rules = {
    'angry': 'provide_calming_response',
    'sad': 'offer_emotional_support',
    'happy': 'engage_in_positive_conversation',
    'neutral': 'neutral_response',
    'calm': 'neutral_response',
    'fear': 'offer_reassurance',
    'disgust': 'offer_understanding',
    'surprise': 'acknowledge_surprise'
}

# Define the response generation function using OpenAI
def generate_response(predicted_behavior):
    prompts = {
        'provide_calming_response': "Respond empathetically to a user who is feeling angry:",
        'offer_emotional_support': "Respond empathetically to a user who is feeling sad:",
        'engage_in_positive_conversation': "Respond positively to a user who is feeling happy:",
        'neutral_response': "Respond neutrally to a user who is feeling neutral or calm:",
        'offer_reassurance': "Respond reassuringly to a user who is feeling fear:",
        'offer_understanding': "Respond with understanding to a user who is feeling disgust:",
        'acknowledge_surprise': "Acknowledge and respond to a user who is feeling surprised:"
    }

    prompt = prompts.get(predicted_behavior, "Respond neutrally:")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=100,
    )

    return response['choices'][0]['message']['content']

def extract_features(data, sample_rate=16000):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result

def get_features(data, sample_rate=16000):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    # data, _ = librosa.load(data, duration=2.5, offset=0.6)  # Remove this line

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    pitch_factor = 0.7  # you can adjust this value
    data_stretch_pitch = pitch(new_data, sample_rate, pitch_factor)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate=rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

    
@app.route('/process_voice', methods=['POST'])
def process_voice():
    def preprocess_user_voice(chunk=1024, format=None, channels=1, rate=16000, record_seconds=5, input_device_index=None):
        frames = []
        wav_file = None

        try:
            print("Recording...")
            recording = sd.rec(int(record_seconds * rate), samplerate=rate, channels=channels, device=input_device_index)
            sd.wait()
            print("Finished recording.")
        except Exception as e:
            print(f"An error occurred during recording: {e}")
            return None

        # Save the recorded audio data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = temp_wav.name
            try:
                sf.write(temp_wav_path, recording, rate)
            except Exception as e:
                print(f"An error occurred while writing the audio file: {e}")
                return None

        # Load the audio from the temporary file
        try:
            user_audio, _ = librosa.load(temp_wav_path, sr=None)
        except Exception as e:
            print(f"An error occurred while loading audio: {e}")
            return None
        finally:
            # Remove the temporary file
            os.remove(temp_wav_path)

        user_audio_features = get_features(user_audio)
        user_audio_features = scaler.transform(user_audio_features)
        user_audio_features = np.expand_dims(user_audio_features, axis=2)

        return user_audio_features

    try:
        processed_audio = preprocess_user_voice(input_device_index=None)
        if processed_audio is None:
            return "Error occurred during audio processing."
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error occurred during audio processing."

    # Predict the user's emotion
    try:
        pred = model.predict(processed_audio)
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return "Error occurred during emotion prediction."

    predicted_emotion = encoder.inverse_transform(pred)[0][0]

    # Predict the system's behavior based on the emotion
    predicted_behavior = behavior_rules.get(predicted_emotion, 'neutral_response')

    # Generate the system's response
    try:
        response_text = generate_response(predicted_behavior)
    except Exception as e:
        print(f"An error occurred during response generation: {e}")
        return "Error occurred during response generation."

    print("Predicted Emotion:", predicted_emotion)

    return response_text


def search_links_and_images(query, max_links=2, max_images=2):
    api_key = 'AIzaSyDbBWV7YbM6q1WaKRsTronGUwLRggw_skE'
    cx = 'd34ae84b1bb0d4079'
    api_url = f'https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={query}'

    try:
        response = requests.get(api_url)
        data = response.json()

        if 'items' in data and len(data['items']) > 0:
            web_results = [item for item in data['items'] if item['kind'] == 'customsearch#result']
            results_html = '<h3>Related Links and Images:</h3>'

            link_count = 0
            image_count = 0

            for item in web_results:
                if link_count < max_links:
                    filtered_title = ''.join(e for e in item['title'] if e.isalnum() or e.isspace())
                    results_html += f'<a href="{item["link"]}" target="_blank">{filtered_title}</a><br>'
                    link_count += 1

                if image_count < max_images and 'pagemap' in item and 'cse_image' in item['pagemap'] and len(item['pagemap']['cse_image']) > 0:
                    for image in item['pagemap']['cse_image']:
                        if image_count < max_images:
                            results_html += f'<img src="{image["src"]}" alt="{filtered_title}" style="max-width: 100%;"><br>'
                            image_count += 1

            return results_html

        else:
            return 'No links found'
    except Exception as e:
        return f'Error: {str(e)}'
    

@app.route('/')
def index():
    try:
        return render_template('KalpAI.html')
    except Exception as e:
        print("Error in index route:", e)
        return "An error occurred. Please try again later."
    
    
random_messages = [
    "How can I further assist you?",
    "Is there anything else I can help you with?",
    "Do you have any other questions or concerns?",
    "Feel free to ask me anything else!",
    "Would you like more information on this topic?",
    "I'm here to help! What else can I do for you?",
    "If you have any other inquiries, just let me know.",
    "Don't hesitate to ask if you need anything else.",
    "I'm at your service! What's next?",
    "Let me know if there's anything else you need assistance with.",
    "Need help with anything else?",
    "What else can I do to assist you?",
    "Ready to help with anything else you need.",
    "If there's anything else on your mind, feel free to ask.",
    "Want to explore any other topics?",
    "I'm here to provide support for anything else you require.",
    "Any other questions or tasks you need assistance with?",
    "Looking for more information or assistance?",
    "Happy to provide further assistance if needed."
]

custom_messages = {
   "hi hi": [
        "Hi hi [name]! How may I assist you today?"
    ],
    "how's it going": [
        "How's it going [name]? What can I do for you today?"
    ],
    "hey you": [
        "Hey you [name]! How may I assist you today?"
    ],
    "hi": [
        "Hi [name]! How can I assist you today?"
    ],
    "hello": [
        "Hello [name]! How can I assist you today?"
    ],
    "hello AI": [
        "Hello [name]! How can I assist you today?"
    ],
    "hello Technikalp": [
        "Hello [name]! How can I assist you today?"
    ],
    "hello KalpAI": [
        "Hello [name]! How can I assist you today?"
    ],
    "hello Buddy": [
        "Hello [name]! How can I assist you today?"
    ],
    "hey": [
        "Hey [name]! What can I do for you today?"
    ],
    "hey there": [
        "Hey [name]! What can I do for you today?"
    ],
    "hey buddy": [
        "Hey [name]! What can I do for you today?"
    ],
    "good morning": [
        "Good morning [name]! How may I help you today?",
        "Rise and shine [name]! How can I assist you?",
        "Top of the morning [name]! How may I be of service?",
        "Morning [name]! What brings you here today?",
        "Good morning [name]! How's your day looking?"
    ],
    "good afternoon": [
        "Good afternoon [name]! How may I assist you today?",
        "Afternoon [name]! How's your day going so far?",
        "Hey there [name]! What can I do for you this afternoon?",
        "How's the day treating you [name]? What can I assist you with?",
        "Hello [name]! Ready to tackle the afternoon? How can I help?"
    ],
    "good evening": [
        "Good evening [name]! How may I be of service to you today?",
        "Evening [name]! What can I assist you with as the day winds down?",
        "Hey [name], good evening! Anything I can do for you?",
        "Hello [name]! How can I assist you on this lovely evening?",
        "Good evening [name]! How's your day been so far?"
    ],
    "what's up": [
        "Not much [name]! How can I assist you today?",
        "Hey [name]! What's up? How can I help?",
        "Hey [name]! What's going on? How can I assist you today?",
        "What's up [name]? How can I be of assistance?",
        "Hey [name]! How's it going? What can I assist you with?"
    ],
    "how are you": [
        "I'm doing great, hopefully you are too! How may I assist you?"
    ],
    "greetings": [
        "Greetings [name]! What can I assist you with today?"
    ],
    "salutations": [
        "Salutations [name]! How can I assist you today?"
    ],
    "hey friend": [
        "Hey [name]! How may I be of assistance to you today?"
    ],
    "top of the morning": [
        "Top of the morning [name]! How may I help you today?"
    ],
    "good day": [
        "Good day [name]! What can I do for you today?"
    ],
    "hello there": [
        "Hello [name]! How can I assist you today?"
    ],
    "hi there": [
        "Hi [name]! How may I assist you today?"
    ],
    "howdy": [
        "Howdy [name]! What can I do for you today?"
    ],
    "hey pal": [
        "Hey [name]! What's on your mind today?"
    ],
    "hiya": [
        "Hiya [name]! How can I assist you today?"
    ],
    "yo": [
        "Yo [name]! What's going on?"
    ],
    "good to see you": [
        "Good to see you [name]! How can I assist you today?"
    ],
    "well met": [
        "Well met [name]! What brings you here?"
    ],
    "bye buddy": ["Goodbye [name]! Have a great day!"],
    "see you later buddy": ["See you later [name]! Take care!"],
    "goodbye buddy" : ["Bye for now [name]! Reach out anytime!"],
    "goodbye" : ["Goodbye [name]! Have a great day!"],
    "bye": ["Goodbye [name]! Have a great day!"],
    "bye technikalp": ["Goodbye [name]! Have a great day!"],
    "bye kalpai": ["Goodbye [name]! Have a great day!"],
    "bye friend" : ["Goodbye [name]! Have a great day!"],
    "see you later": ["See you later [name]! Take care!"],
    "see you soon": ["See you soon [name]! Have a great day!"],
    "take care": ["Take care [name]! Have a great day!"],
    "good night": ["Good night [name]! Sweet dreams!"],
    "goodnight": ["Good night [name]! Sweet dreams!"],
    "good night buddy": ["Good night [name]! Sweet dreams!"],
    "who is your developer": ["I was created by a team of developers and engineers at TechnikalpAI."],
    "who created you": ["I was created by a team of developers and engineers at TechnikalpAI."],
    


}

@app.route('/get_response', methods=['POST'])
def get_response():
    query = request.form['query']
    name = "friend"
    if query.lower() in custom_messages:
        response = random.choice(custom_messages[query.lower()]).replace("[name]", name)
        return jsonify({'response': response})

    # If the query does not match any custom message, use the ask_question function
    response = ask_question(query)
    links_and_images = search_links_and_images(query)
    random_message = random.choice(random_messages)
    return jsonify({'response': response + " " + random_message, 'links_and_images': links_and_images})


if __name__ == '__main__':
    app.run(debug=True)