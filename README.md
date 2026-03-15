# The Empathy Engine 

## Project Description
The Empathy Engine is an advanced neural Text-to-Speech (TTS) pipeline that dynamically adapts its vocal delivery based on the emotional context of the input text. Instead of relying on static, robotic voices, this application uses a state-of-the-art Natural Language Processing (NLP) model to detect the underlying sentiment of a sentence and programmatically modulates the ElevenLabs TTS API to reflect that emotion in real-time.

Built with Python and FastAPI, the project features a sleek web interface that allows users to input text, view the detected granular emotion alongside a confidence meter, and immediately hear the dynamically generated, emotionally expressive audio.


## Environment Setup & Installation Instructions

Follow these steps to run the application on your local machine.

### 1.prerequisites
* Python 3.8+ installed.
* An active ElevenLabs API Key.


### 2. Environment Setup
Open your terminal and create a virtual environment in your project directory:
```bash
python -m venv venv

# Activate the virtual environment:

Windows: .\venv\Scripts\activate

Mac/Linux: source venv/bin/activate

# Install Dependencies
using: pip install -r requirements.txt

# Configure API key:
In main.py find API key variable at top:
ELEVENLABS_API_KEY = "your_api_key_here"


#Run the Application:
python main.py


## Design Choices and Logical Mapping
To satisfy the core requirement of mapping emotions to voice parameters, I implemented a multi-layered architectural approach focusing on granular emotion detection, dynamic API modulation, and text pre-processing.


# Granular Emotion Detection (Top-K Sensitivity)
Instead of basic positive/negative sentiment analysis, I integrated the SamLowe/roberta-base-go_emotions model via Hugging Face. This model detects 28 highly specific emotional states. To prevent the model from defaulting to "neutral" too easily, I implemented a Top-K (K=5) search heuristic. The system scans the top 5 predicted labels to find any match within our predefined emotional categories before falling back to neutral.

These 28 states are mapped into 5 broad vocal categories:

Joy: (joy, amusement, excitement, optimism, love, relief, pride, admiration, approval, gratitude)

Sadness: (sadness, disappointment, grief, remorse, embarrassment)

Anger: (anger, annoyance, disapproval, disgust)

Fear: (fear, nervousness, desire)

Surprise: (surprise, realization, confusion, curiosity)



# Vocal Parameter Modulation Logic (ElevenLabs API)
To fulfill the requirement of altering at least two distinct vocal parameters, I utilized the ElevenLabs VoiceSettings object to dynamically modulate Stability (emotional variance) and Style (emotional exaggeration).

Joy/Surprise: Moderate Stability (0.40 - 0.45) and High Style (0.75 - 0.85) to create a bright, expressive, and highly animated delivery.

Sadness/Fear: High Stability (0.65) and Low Style (0.10) to force a subdued, flatter, and more constrained delivery.

Anger: Lower Stability (0.40) and High Style (0.80) to produce an intense, aggressive tone. (Note: Stability was intentionally kept at 0.40 rather than 0.15 to prevent the AI from generating unwanted audio artifacts like whispering or murmuring).

Neutral: High Stability (0.75) and Zero Style (0.0) for standard, informational delivery.



# Emotional Text Pre-Processing
Because advanced LLM-based TTS engines react strongly to punctuation and capitalization, I implemented a text pre-processor that acts as a "director" for the AI. Before sending the string to ElevenLabs, the engine injects contextual prosody cues:

Anger: Applies .upper() and !!! to forcefully trigger the model's high-energy shouting parameters.

Sadness: Wraps the text in ellipses (... text ...) to force natural pauses, sighs, and a slower cadence.

Joy/Surprise: Injects leading conversational markers (Oh!, What?) to guarantee an upward inflection.