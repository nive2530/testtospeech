from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoProcessor, VitsModel
import torch
import soundfile as sf
from fastapi.responses import FileResponse

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; change it to a specific domain if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load pre-trained VITS model
model_name = "facebook/mms-tts-eng"
processor = AutoProcessor.from_pretrained(model_name)
model = VitsModel.from_pretrained(model_name)

# Define request model
class TextToSpeechRequest(BaseModel):
    text: str

# Define API endpoint
@app.post("/synthesize")
async def synthesize_speech(request: TextToSpeechRequest):
    try:
        # Convert text to waveform
        inputs = processor(request.text, return_tensors="pt")

        with torch.no_grad():
            speech = model(**inputs).waveform

        # Save audio to a file
        audio_data = speech.squeeze().cpu().numpy()
        output_file = "output.wav"
        sf.write(output_file, audio_data, samplerate=22050)

        # Return generated speech as a downloadable file
        return FileResponse(output_file, media_type="audio/wav", filename="speech.wav")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
