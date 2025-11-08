from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
import numpy as np
import io
import wave
import logging
from pathlib import Path
import os
from typing import Optional

# Import Moshi components
from moshi.models import loaders, LMGen

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Moshi TTS API", version="1.0.0")

# Global variables for model
model = None
device = None
sample_rate = 24000  # Moshi uses 24kHz

class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[int] = 0
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.95

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Load the model on startup"""
    global model, device
    
    try:
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Load Moshi model
        logger.info("Loading Moshi model...")
        model = loaders.load_moshi_model(
            "kyutai/moshika-pytorch-bf16", 
            device=device
        )
        model.eval()
        
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=(model is not None)
    )

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Generate speech from text
    
    Example:
    curl -X POST "http://localhost:8000/tts" \
         -H "Content-Type: application/json" \
         -d '{"text": "Hello world"}' \
         --output audio.wav
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        logger.info(f"Generating audio for text: {request.text[:100]}...")
        
        # Tokenize text
        text_tokens = model.text_tokenizer.encode(request.text)
        text_tokens = torch.tensor(text_tokens).unsqueeze(0).to(device)
        
        # Generate audio codes
        with torch.no_grad():
            audio_codes = model.generate(
                text_tokens,
                max_new_tokens=int(len(request.text) * 10),  # Approximate
                temperature=request.temperature,
                top_p=request.top_p,
                use_sampling=True
            )
        
        # Decode to audio
        audio_data = model.audio_tokenizer.decode(audio_codes)
        
        # Convert to numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_np = audio_data.cpu().numpy()
        else:
            audio_np = audio_data
        
        # Ensure correct shape
        if len(audio_np.shape) > 1:
            audio_np = audio_np.squeeze()
        
        # Normalize audio to int16 range
        audio_np = np.clip(audio_np, -1, 1)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=generated_audio.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """
    Stream audio generation (for longer texts)
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    async def audio_generator():
        try:
            # Similar to above but with streaming
            text_tokens = model.text_tokenizer.encode(request.text)
            text_tokens = torch.tensor(text_tokens).unsqueeze(0).to(device)
            
            # Generate audio in chunks
            with torch.no_grad():
                for audio_chunk in model.generate_stream(
                    text_tokens,
                    max_new_tokens=int(len(request.text) * 10),
                    temperature=request.temperature,
                    top_p=request.top_p
                ):
                    # Decode chunk
                    audio_data = model.audio_tokenizer.decode(audio_chunk)
                    
                    # Convert to bytes
                    if isinstance(audio_data, torch.Tensor):
                        audio_np = audio_data.cpu().numpy()
                    else:
                        audio_np = audio_data
                    
                    audio_np = np.clip(audio_np, -1, 1)
                    audio_int16 = (audio_np * 32767).astype(np.int16)
                    
                    yield audio_int16.tobytes()
                    
        except Exception as e:
            logger.error(f"Error in stream generation: {str(e)}")
            raise
    
    return StreamingResponse(
        audio_generator(),
        media_type="audio/raw",
        headers={
            "Content-Type": "audio/raw",
            "X-Audio-Sample-Rate": str(sample_rate),
            "X-Audio-Channels": "1",
            "X-Audio-Bits": "16"
        }
    )

@app.get("/")
async def root():
    """Root endpoint with API documentation"""
    return {
        "name": "Moshi TTS API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "GET - Health check",
            "/tts": "POST - Generate audio from text",
            "/tts/stream": "POST - Stream audio generation",
            "/docs": "GET - Interactive API documentation"
        },
        "example": {
            "curl": "curl -X POST http://localhost:8000/tts -H 'Content-Type: application/json' -d '{\"text\": \"Hello world\"}' --output audio.wav"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
