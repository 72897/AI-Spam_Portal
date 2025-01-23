from flask import Flask, request, render_template, jsonify
import face_recognition
import speech_recognition as sr
import cv2
import os

app = Flask(__name__)

def verify_faces(image1_path, image2_path):
    img1 = face_recognition.load_image_file(image1_path)
    img2 = face_recognition.load_image_file(image2_path)

    encodings1 = face_recognition.face_encodings(img1)
    encodings2 = face_recognition.face_encodings(img2)

    if len(encodings1) > 0 and len(encodings2) > 0:
        result = face_recognition.compare_faces([encodings1[0]], encodings2[0])
        return "Match" if result[0] else "No Match"
    return "No Face Detected"

def recognize_speech(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Speech not recognized"
    except sr.RequestError:
        return "API error"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify_face', methods=['POST'])
def verify_face():
    if 'id_image' not in request.files or 'live_image' not in request.files:
        return jsonify({'error': 'Please upload both images'})

    id_image = request.files['id_image']
    live_image = request.files['live_image']

    id_image.save('static/id_image.jpg')
    live_image.save('static/live_image.jpg')

    result = verify_faces('static/id_image.jpg', 'static/live_image.jpg')
    return jsonify({'verification_result': result})

@app.route('/detect_spam', methods=['POST'])
def detect_spam():
    if 'audio' not in request.files:
        return jsonify({'error': 'Please upload an audio file'})

    audio = request.files['audio']
    audio.save('static/audio.wav')

    text = recognize_speech('static/audio.wav')
    return jsonify({'transcription': text})

if __name__ == '__main__':

    app.run(debug=True)