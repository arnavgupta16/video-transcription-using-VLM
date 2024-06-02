import os
import cv2
import json
import whisper
import moviepy.editor as mp
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-34b-hf")
model = AutoModelForSequenceClassification.from_pretrained("llava-hf/llava-v1.6-34b-hf")

def transcribe_audio(audio_file):
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    return result['text']

def extract_audio(video_path, audio_path):
    video = mp.VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)

def analyze_frames(video_path):
    video = cv2.VideoCapture(video_path)
    frame_data = []
    frame_rate = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(0, frame_count, int(frame_rate)):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = video.read()
        if ret:
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize frame to 224x224 for the model
            resized_frame = cv2.resize(rgb_frame, (224, 224))
            # Convert frame to a format suitable for the model
            inputs = tokenizer(resized_frame, return_tensors="pt")
            # Use model to analyze the frame
            outputs = model(**inputs)
            predicted_emotion = torch.argmax(outputs.logits).item()
            frame_data.append(predicted_emotion)
    return frame_data

def enhance_transcription_with_metadata(transcription, metadata):
    enhanced_transcription = []
    words = transcription.split()
    for idx, word in enumerate(words):
        emotion_data = metadata[min(idx, len(metadata) - 1)]
        enhanced_transcription.append(f"{word} [{emotion_data}]")
    return ' '.join(enhanced_transcription)

def main(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    transcription = transcribe_audio(audio_path)
    metadata = analyze_frames(video_path)
    enhanced_transcription = enhance_transcription_with_metadata(transcription, metadata)
    print(enhanced_transcription)
    os.remove(audio_path)  # Clean up temporary audio file

if __name__ == "__main__":
    video_path = r""
    main(video_path)
