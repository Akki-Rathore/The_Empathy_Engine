import os
import base64
import webbrowser
import socket
import time
import threading
from transformers import pipeline
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

app = FastAPI(title="The Empathy Engine")

ELEVENLABS_API_KEY = "YOUR_API_KEY_HERE"
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

print("Loading NLP Pipeline")
classifier = pipeline(
    "text-classification", 
    model="SamLowe/roberta-base-go_emotions"
)

emotion_mapping = {
    'joy': ['joy', 'amusement', 'excitement', 'optimism', 'love', 'relief', 'pride', 'admiration', 'approval', 'gratitude'],
    'sadness': ['sadness', 'disappointment', 'grief', 'remorse', 'embarrassment'],
    'anger': ['anger', 'annoyance', 'disapproval', 'disgust'],
    'fear': ['fear', 'nervousness', 'desire'],
    'surprise': ['surprise', 'realization', 'confusion', 'curiosity']
}

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return """
    <html>
        <head>
            <title>The Empathy Engine</title>
            <style>
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px 20px; background-color: #f8fafc; color: #1e293b; }
                .container { background: white; padding: 50px; border-radius: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.04); border: 1px solid #e2e8f0; text-align: center; }
                h1 { font-size: 2.5rem; font-weight: 800; color: #0f172a; margin-bottom: 10px; letter-spacing: -1px; }
                p { color: #64748b; font-size: 1.1rem; margin-bottom: 40px; }
                textarea { width: 100%; height: 160px; padding: 20px; border-radius: 12px; border: 2px solid #e2e8f0; font-size: 18px; line-height: 1.6; margin-bottom: 25px; resize: none; transition: all 0.2s; outline: none; }
                textarea:focus { border-color: #6366f1; box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.1); }
                button { width: 100%; padding: 16px; background: #6366f1; color: white; border: none; border-radius: 12px; font-size: 1.2rem; font-weight: 600; cursor: pointer; transition: all 0.2s; }
                button:hover { background: #4f46e5; transform: translateY(-2px); }
                .footer { margin-top: 30px; font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>The Empathy Engine 🎙️</h1>
                <p>Advanced Neural Text-to-Speech Analysis</p>
                <form action="/generate" method="post">
                    <textarea name="text" required placeholder="Type a sentence to see it analyzed and converted to expressive speech..."></textarea>
                    <button type="submit">Analyze & Synthesize Voice</button>
                </form>
                <div class="footer">Powered by ElevenLabs & Roberta NLP</div>
            </div>
        </body>
    </html>
    """

@app.post("/generate", response_class=HTMLResponse)
async def generate_audio(text: str = Form(...)):
    # 1.EMOTION DETECTION
    results = classifier(text, top_k=5)
    
    detected_granular = "neutral"
    intensity = 0.0
    broad_cat = "neutral"

    for pred in results:
        label = pred['label']
        if any(label in specifics for specifics in emotion_mapping.values()):
            for cat, specifics in emotion_mapping.items():
                if label in specifics:
                    broad_cat = cat
                    detected_granular = label
                    intensity = pred['score']
                    break
            break
    
    if broad_cat == "neutral":
        detected_granular = results[0]['label']
        intensity = results[0]['score']

    # 2.VOCAL MODULATION
    api_text = text
    if broad_cat == 'joy':
        target_stability, target_style = 0.45, 0.75
    elif broad_cat == 'surprise':
        target_stability, target_style = 0.40, 0.85
    elif broad_cat == 'sadness' or broad_cat == 'fear':
        target_stability, target_style = 0.65, 0.10
    elif broad_cat == 'anger':
        target_stability, target_style = 0.40, 0.80
        api_text = f"{text.upper()}!" 
    else: 
        target_stability, target_style = 0.75, 0.0

    # 3.ELEVENLABS API CALL
    output_filename = "empathy_output.wav"
    try:
        audio_stream = client.text_to_speech.convert(
            text=api_text,
            voice_id="EXAVITQu4vr4xnSDxMaL", 
            model_id="eleven_multilingual_v2",
            voice_settings=VoiceSettings(
                stability=target_stability,
                similarity_boost=0.75,
                style=target_style,
                use_speaker_boost=True
            )
        )
        with open(output_filename, "wb") as f:
            for chunk in audio_stream:
                if chunk: f.write(chunk)
    except Exception as e:
        return f"<h2>API Error</h2><p>{str(e)}</p><a href='/'>Back</a>"

    # 4.ENCODE AND UI
    with open(output_filename, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode('utf-8')

    return f"""
    <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', sans-serif; text-align: center; padding: 60px 20px; background: #f8fafc; color: #1e293b; }}
                .result-card {{ background: white; padding: 50px; border-radius: 24px; display: inline-block; box-shadow: 0 20px 40px rgba(0,0,0,0.06); border: 1px solid #e2e8f0; min-width: 450px; }}
                .label {{ font-size: 14px; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; }}
                .emotion-value {{ font-size: 38px; font-weight: 800; color: #6366f1; margin-bottom: 25px; text-transform: capitalize; }}
                .meter-bg {{ width: 100%; background: #f1f5f9; height: 12px; border-radius: 6px; margin: 20px 0 40px 0; overflow: hidden; }}
                .meter-fill {{ height: 100%; background: #6366f1; width: {int(intensity*100)}%; border-radius: 6px; }}
                audio {{ width: 100%; border-radius: 12px; }}
                .btn-new {{ display: inline-block; margin-top: 40px; padding: 12px 30px; background: #f1f5f9; color: #475569; text-decoration: none; border-radius: 10px; font-weight: 600; transition: 0.2s; }}
                .btn-new:hover {{ background: #e2e8f0; color: #1e293b; }}
            </style>
        </head>
        <body>
            <div class="result-card">
                <div class="label">Primary Detected Emotion</div>
                <div class="emotion-value">{detected_granular}</div>
                
                <div class="label">Sentiment Intensity</div>
                <div class="meter-bg"><div class="meter-fill"></div></div>
                
                <audio controls autoplay><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>
                <br>
                <a href="/" class="btn-new">New Analysis</a>
            </div>
        </body>
    </html>
    """

def start_browser():
    while True:
        try:
            with socket.create_connection(("127.0.0.1", 8000), timeout=1): break
        except OSError: time.sleep(0.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    threading.Thread(target=start_browser, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=8000)