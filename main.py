import os
import io
import json
import cv2
import moviepy.editor as mp
import speech_recognition as sr
from transformers import pipeline
import whisper

emotion_classifier = pipeline('sentiment-analysis', model='cointegrated/rubert-tiny2-cedr-emotion-detection')

def transcribe_audio(audio_files):
    r = sr.Recognizer()
    with sr.AudioFile(audio_files) as source:
        audio_data = r.record(source)
        text = r.recognize_whisper(audio_data)
        
    return text

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
        
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
           
            results = emotion_classifier(rgb_frame)
            frame_data.append(results)
    return frame_data

def enhance_transcription_with_metadata(transcription, metadata):
    enhanced_transcription = []
    for idx, text in enumerate(transcription.split()):
        emotion_data = metadata[min(idx, len(metadata) - 1)]
        enhanced_transcription.append(f"{text} [{emotion_data}]")
    return ' '.join(enhanced_transcription)

def main(video_path):
    audio_path = "temp_audio.wav"
    extract_audio(video_path, audio_path)
    transcription = transcribe_audio(audio_path)
    metadata = analyze_frames(video_path)
    enhanced_transcription = enhance_transcription_with_metadata(transcription, metadata)
    print(enhanced_transcription)

if __name__ == "__main__":
    video_path = r"your_file_path"
    main(video_path)
