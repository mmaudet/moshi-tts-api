from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel, Field
import torch
import numpy as np
import io
import wave
import logging
import tempfile
import os
from typing import Optional, List
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Moshi TTS API", 
    version="1.0.0",
    description="Text-to-Speech API using Kyutai's Moshi model"
)

# Global variables
model = None
device = None
executor = ThreadPoolExecutor(max_workers=2)
sample_rate = 24000  # Moshi sample rate

class TTSRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize", max_length=5000)
    speaker_id: Optional[int] = Field(0, description="Speaker ID (if multiple speakers)", ge=0, le=10)
    temperature: Optional[float] = Field(0.7, description="Sampling temperature", ge=0.1, le=1.0)
    top_p: Optional[float] = Field(0.9, description="Nucleus sampling threshold", ge=0.1, le=1.0)
    max_length: Optional[int] = Field(None, description="Maximum audio length in samples")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

class TTSBatchRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to synthesize")
    speaker_id: Optional[int] = Field(0, description="Speaker ID")
    
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    available_memory_gb: Optional[float] = None

def load_moshi_model():
    """Load the Moshi model with proper error handling"""
    global model, device
    
    try:
        import sys
        sys.path.append('/tmp/moshi')
        
        # Check available device
        if torch.cuda.is_available():
            device = "cuda"
            logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            logger.info("CUDA not available. Using CPU")
        
        # Import Moshi components after adding to path
        from moshi import MoshiServer, load_model
        
        logger.info("Loading Moshi model...")
        
        # Load the model with the correct method
        model_path = "kyutai/moshika-pytorch-bf16"
        
        # Try to load model from HuggingFace
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Load tokenizer and model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if device == "cpu":
                model = model.to(device)
                
            model.eval()
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.warning(f"Could not load from HuggingFace, trying alternative: {e}")
            
            # Alternative loading method
            model = MoshiServer(
                model_path=model_path,
                device=device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32
            )
            logger.info("Model loaded with MoshiServer")
            
    except ImportError as e:
        logger.error(f"Failed to import Moshi components: {e}")
        logger.info("Attempting to install moshi...")
        os.system("cd /tmp && git clone https://github.com/kyutai-labs/moshi.git && cd moshi && pip install -e .")
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    try:
        load_moshi_model()
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Continue running but model will be None

@app.get("/", response_class=dict)
async def root():
    """API information and usage examples"""
    return {
        "name": "Moshi TTS API",
        "version": "1.0.0",
        "description": "Text-to-Speech using Kyutai's Moshi model",
        "endpoints": {
            "/": "This page",
            "/health": "GET - Health check and system info",
            "/tts": "POST - Generate audio from text",
            "/tts/batch": "POST - Generate multiple audio files",
            "/docs": "GET - Interactive API documentation",
            "/redoc": "GET - Alternative API documentation"
        },
        "usage_examples": {
            "simple": 'curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" -d \'{"text": "Hello world"}\' --output audio.wav',
            "with_options": 'curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" -d \'{"text": "Hello", "temperature": 0.5, "top_p": 0.9}\' --output audio.wav'
        },
        "model_info": {
            "loaded": model is not None,
            "device": device if device else "not initialized"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system information"""
    health_info = {
        "status": "healthy" if model is not None else "model_not_loaded",
        "model_loaded": model is not None,
        "device": device if device else "none"
    }
    
    # Add GPU memory info if available
    if device == "cuda" and torch.cuda.is_available():
        try:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            health_info["available_memory_gb"] = memory_reserved - memory_allocated
        except:
            pass
    
    return HealthResponse(**health_info)

def generate_audio_sync(text: str, temperature: float, top_p: float, seed: Optional[int] = None):
    """Synchronous audio generation function"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    try:
        # This is a simplified version - actual implementation depends on Moshi's API
        # You'll need to adapt this based on the actual Moshi model interface
        
        with torch.no_grad():
            # Generate audio from text
            # This is pseudo-code - replace with actual Moshi API calls
            if hasattr(model, 'generate_audio'):
                audio_data = model.generate_audio(
                    text=text,
                    temperature=temperature,
                    top_p=top_p
                )
            elif hasattr(model, 'generate'):
                # Alternative API
                inputs = model.tokenizer(text, return_tensors="pt").to(device)
                audio_tokens = model.generate(
                    **inputs,
                    temperature=temperature,
                    top_p=top_p,
                    max_length=512
                )
                audio_data = model.decode_audio(audio_tokens)
            else:
                # Fallback - generate dummy audio for testing
                logger.warning("Using dummy audio generation - model API not recognized")
                duration = len(text) * 0.1  # Rough estimate
                t = np.linspace(0, duration, int(sample_rate * duration))
                frequency = 440  # A4 note
                audio_data = np.sin(2 * np.pi * frequency * t) * 0.3
        
        # Ensure audio is numpy array
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Ensure correct shape (1D array)
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()
        
        # Normalize to [-1, 1] range
        audio_data = np.clip(audio_data, -1, 1)
        
        return audio_data
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise

@app.post("/tts")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """
    Generate speech from text and return WAV file
    
    Example:
    ```bash
    curl -X POST "http://localhost:8000/tts" \
         -H "Content-Type: application/json" \
         -d '{"text": "Hello, this is a test"}' \
         --output audio.wav
    ```
    """
    if model is None:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Please wait for initialization or check logs."
        )
    
    try:
        logger.info(f"Generating audio for text: '{request.text[:50]}...'")
        
        # Run generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            executor,
            generate_audio_sync,
            request.text,
            request.temperature,
            request.top_p,
            request.seed
        )
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            
            # Apply max_length if specified
            if request.max_length and len(audio_int16) > request.max_length:
                audio_int16 = audio_int16[:request.max_length]
            
            wav_file.writeframes(audio_int16.tobytes())
        
        wav_buffer.seek(0)
        
        # Log generation info
        duration = len(audio_int16) / sample_rate
        logger.info(f"Generated {duration:.2f} seconds of audio")
        
        return StreamingResponse(
            wav_buffer,
            media_type="audio/wav",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.wav",
                "X-Audio-Duration": str(duration),
                "X-Sample-Rate": str(sample_rate)
            }
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.post("/tts/batch")
async def text_to_speech_batch(request: TTSBatchRequest, background_tasks: BackgroundTasks):
    """
    Generate multiple audio files from a list of texts
    Returns a ZIP file containing all generated WAV files
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.texts) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 texts per batch")
    
    try:
        import zipfile
        
        # Create temporary directory for files
        with tempfile.TemporaryDirectory() as temp_dir:
            zip_path = os.path.join(temp_dir, "audio_batch.zip")
            
            with zipfile.ZipFile(zip_path, 'w') as zip_file:
                for i, text in enumerate(request.texts):
                    logger.info(f"Processing text {i+1}/{len(request.texts)}")
                    
                    # Generate audio
                    loop = asyncio.get_event_loop()
                    audio_data = await loop.run_in_executor(
                        executor,
                        generate_audio_sync,
                        text,
                        0.7,  # Default temperature
                        0.9,  # Default top_p
                        None
                    )
                    
                    # Convert to WAV
                    audio_int16 = (audio_data * 32767).astype(np.int16)
                    
                    # Create WAV file
                    wav_buffer = io.BytesIO()
                    with wave.open(wav_buffer, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                        wav_file.writeframes(audio_int16.tobytes())
                    
                    # Add to ZIP
                    wav_buffer.seek(0)
                    zip_file.writestr(f"audio_{i+1:03d}.wav", wav_buffer.read())
            
            # Return ZIP file
            return FileResponse(
                zip_path,
                media_type="application/zip",
                filename="audio_batch.zip"
            )
            
    except Exception as e:
        logger.error(f"Batch generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model
    if model is not None:
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    executor.shutdown(wait=True)
    logger.info("Shutdown complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )
