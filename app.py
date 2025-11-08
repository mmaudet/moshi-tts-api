"""
Moshi TTS API Server
====================
FastAPI server for Moshi text-to-speech synthesis
Supporting French and English languages

Backend Support:
- MLX (macOS with Metal GPU) - preferred on Apple Silicon
- PyTorch (CUDA/CPU) - fallback for other platforms
"""

from fastapi import FastAPI, HTTPException, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from enum import Enum
import numpy as np
import io
import wave
import logging
import os
import json
from typing import Optional, Dict, Any
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# Import configuration
from config import get_settings

# Backend detection - try MLX first (for macOS Metal GPU), then PyTorch
BACKEND = None
try:
    import mlx.core as mx
    import mlx.nn as nn
    import sentencepiece
    from moshi_mlx import models as mlx_models
    from moshi_mlx.models.tts import TTSModel as MLXTTSModel
    from moshi_mlx.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO
    from moshi_mlx.utils.loaders import hf_get
    BACKEND = "mlx"
    logger_init = logging.getLogger(__name__)
    logger_init.info("✅ MLX backend detected (Metal GPU acceleration)")
except ImportError:
    try:
        import torch
        BACKEND = "pytorch"
        logger_init = logging.getLogger(__name__)
        logger_init.info("✅ PyTorch backend detected")
    except ImportError:
        BACKEND = "dummy"
        torch = None
        logger_init = logging.getLogger(__name__)
        logger_init.warning("⚠️ No ML backend available, using dummy mode")

# Load settings
settings = get_settings()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
API_DESCRIPTION = """
## 🎯 Moshi Text-to-Speech API

This API provides text-to-speech synthesis using Kyutai Labs' Moshi model.

### Features:
- 🌐 Bilingual support (French & English)
- 🎵 High-quality 24kHz audio output
- 🚀 GPU acceleration support
- 📦 WAV format output
- 🔄 Batch processing support

### Usage Example:
```bash
curl -X POST "http://localhost:8000/api/v1/tts" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Hello world", "language": "en"}' \\
     --output output.wav
```
"""

tags_metadata = [
    {
        "name": "TTS",
        "description": "Text-to-Speech synthesis endpoints",
    },
    {
        "name": "System",
        "description": "System information and health checks",
    },
]

# Create FastAPI app with rich documentation
app = FastAPI(
    title=settings.api_title,
    description=API_DESCRIPTION,
    version=settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    contact={
        "name": "Moshi TTS Support",
        "url": "https://github.com/kyutai-labs/moshi",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_credentials,
    allow_methods=["*"] if settings.cors_methods == "*" else settings.cors_methods.split(","),
    allow_headers=["*"] if settings.cors_headers == "*" else settings.cors_headers.split(","),
)

# Global variables
model = None
tts_model = None  # Real Moshi TTS model
device = None
cfg_coef_conditioning = None  # For CFG distillation
executor = ThreadPoolExecutor(max_workers=settings.max_workers)
SAMPLE_RATE = settings.sample_rate

# Enums for API
class LanguageCode(str, Enum):
    """Supported languages for TTS"""
    french = "fr"
    english = "en"
    
    @classmethod
    def get_description(cls):
        return {
            "fr": "French (Français)",
            "en": "English"
        }

class AudioFormat(str, Enum):
    """Supported audio output formats"""
    wav = "wav"
    raw = "raw"

class VoicePreset(str, Enum):
    """Available voice presets from kyutai/tts-voices"""
    # Default voice
    default = "default"

    # VCTK voices (English - Voice Cloning Toolkit)
    vctk_p225 = "vctk/p225_023.wav"
    vctk_p226 = "vctk/p226_023.wav"
    vctk_p227 = "vctk/p227_023.wav"
    vctk_p228 = "vctk/p228_023.wav"
    vctk_p229 = "vctk/p229_023.wav"
    vctk_p230 = "vctk/p230_023.wav"
    vctk_p231 = "vctk/p231_023.wav"
    vctk_p232 = "vctk/p232_023.wav"
    vctk_p233 = "vctk/p233_023.wav"
    vctk_p234 = "vctk/p234_023.wav"

    # CML-TTS French voices (Français)
    cml_fr_1406 = "cml-tts/fr/1406_1028_000009-0003.wav"
    cml_fr_1591 = "cml-tts/fr/1591_1028_000108-0004.wav"
    cml_fr_1770 = "cml-tts/fr/1770_1028_000036-0002.wav"
    cml_fr_2114 = "cml-tts/fr/2114_1656_000053-0001.wav"
    cml_fr_2154 = "cml-tts/fr/2154_2576_000020-0003.wav"
    cml_fr_2216 = "cml-tts/fr/2216_1745_000007-0001.wav"
    cml_fr_2223 = "cml-tts/fr/2223_1745_000009-0002.wav"
    cml_fr_2465 = "cml-tts/fr/2465_1943_000152-0002.wav"
    cml_fr_296 = "cml-tts/fr/296_1028_000022-0001.wav"
    cml_fr_3267 = "cml-tts/fr/3267_1902_000075-0001.wav"

    # Expresso voices (English with emotions/styles)
    expresso_default_ch1 = "expresso/ex01-ex02_default_001_channel1_168s.wav"
    expresso_enunciated_ch1 = "expresso/ex01-ex02_enunciated_001_channel1_432s.wav"
    expresso_fast_ch1 = "expresso/ex01-ex02_fast_001_channel1_104s.wav"
    expresso_projected_ch1 = "expresso/ex01-ex02_projected_001_channel1_46s.wav"
    expresso_whisper_ch1 = "expresso/ex01-ex02_whisper_001_channel1_579s.wav"
    expresso_angry_ch1 = "expresso/ex03-ex01_angry_001_channel1_201s.wav"
    expresso_happy_ch1 = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
    expresso_calm_ch1 = "expresso/ex03-ex01_calm_001_channel1_1143s.wav"
    expresso_confused_ch1 = "expresso/ex03-ex01_confused_001_channel1_909s.wav"

    # EARS voices (English - 50 diverse voices)
    ears_p001 = "ears/p001/freeform_speech_01.wav"
    ears_p002 = "ears/p002/freeform_speech_01.wav"
    ears_p003 = "ears/p003/freeform_speech_01.wav"
    ears_p004 = "ears/p004/freeform_speech_01.wav"
    ears_p005 = "ears/p005/freeform_speech_01.wav"
    ears_p010 = "ears/p010/freeform_speech_01.wav"
    ears_p015 = "ears/p015/freeform_speech_01.wav"
    ears_p020 = "ears/p020/freeform_speech_01.wav"
    ears_p025 = "ears/p025/freeform_speech_01.wav"
    ears_p030 = "ears/p030/freeform_speech_01.wav"
    ears_p035 = "ears/p035/freeform_speech_01.wav"
    ears_p040 = "ears/p040/freeform_speech_01.wav"
    ears_p045 = "ears/p045/freeform_speech_01.wav"
    ears_p050 = "ears/p050/freeform_speech_01.wav"

# Pydantic models for request/response validation
class TTSRequest(BaseModel):
    """Text-to-Speech synthesis request"""
    text: str = Field(
        ...,
        description=f"Text to synthesize (max {settings.max_text_length} characters)",
        min_length=1,
        max_length=settings.max_text_length,
        example="Bonjour, ceci est un test de synthèse vocale."
    )
    language: LanguageCode = Field(
        LanguageCode.french,
        description="Language for synthesis",
        example="fr"
    )
    format: AudioFormat = Field(
        AudioFormat.wav,
        description="Output audio format",
        example="wav"
    )
    voice: VoicePreset = Field(
        VoicePreset.default,
        description="Voice preset to use for synthesis",
        example="default"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate and clean input text"""
        # Remove excessive whitespace
        v = ' '.join(v.split())
        if not v:
            raise ValueError("Text cannot be empty after cleaning")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "text": "Bonjour, je suis Moshi, votre assistant vocal.",
                "language": "fr",
                "format": "wav"
            }
        }

class TTSResponse(BaseModel):
    """Response model for successful synthesis"""
    message: str
    duration_seconds: float
    file_size_bytes: int
    language: str
    sample_rate: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Computing device (cuda/cpu)")
    available_languages: list = Field(..., description="List of available languages")
    api_version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Current server time")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "cuda",
                "available_languages": ["fr", "en"],
                "api_version": "1.0.0",
                "timestamp": "2024-01-01T12:00:00Z"
            }
        }

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")

# Model loading functions
def load_moshi_model():
    """Initialize the Moshi TTS model with auto-detected backend"""
    global model, tts_model, device, cfg_coef_conditioning

    try:
        if BACKEND == "dummy":
            logger.warning("⚠️ No ML backend available, using dummy model for testing")
            model = "dummy"
            device = "cpu"
            return

        # MLX Backend (macOS with Metal GPU)
        if BACKEND == "mlx":
            logger.info("🍎 Loading Moshi TTS with MLX backend (Metal GPU)")
            device = "mlx"

            try:
                # Load configuration
                logger.info(f"Loading config from {settings.default_tts_repo}...")
                raw_config_path = hf_get("config.json", settings.default_tts_repo)
                with open(raw_config_path, "r") as f:
                    raw_config = json.load(f)

                # Load model weights
                mimi_weights = hf_get(raw_config["mimi_name"], settings.default_tts_repo)
                moshi_name = raw_config.get("moshi_name", "model.safetensors")
                moshi_weights = hf_get(moshi_name, settings.default_tts_repo)
                tokenizer_path = hf_get(raw_config["tokenizer_name"], settings.default_tts_repo)

                # Create language model
                logger.info("Loading language model...")
                lm_config = mlx_models.LmConfig.from_config_dict(raw_config)
                lm_model = mlx_models.Lm(lm_config)
                lm_model.set_dtype(mx.bfloat16)
                lm_model.load_pytorch_weights(str(moshi_weights), lm_config, strict=True)

                # Create text tokenizer
                logger.info("Loading text tokenizer...")
                text_tokenizer = sentencepiece.SentencePieceProcessor(str(tokenizer_path))

                # Create audio tokenizer (MIMI)
                logger.info("Loading audio tokenizer (MIMI)...")
                generated_codebooks = lm_config.generated_codebooks
                audio_tokenizer = mlx_models.mimi.Mimi(mlx_models.mimi_202407(generated_codebooks))
                audio_tokenizer.load_pytorch_weights(str(mimi_weights), strict=True)

                # Create TTS model
                tts_model = MLXTTSModel(
                    lm_model,
                    audio_tokenizer,
                    text_tokenizer,
                    voice_repo=settings.default_voice_repo,
                    n_q=settings.model_n_q,
                    temp=settings.model_temp,
                    cfg_coef=settings.model_cfg_coef,
                    raw_config=raw_config,
                )

                # Handle CFG distillation
                cfg_coef_conditioning = None
                if tts_model.valid_cfg_conditionings:
                    cfg_coef_conditioning = tts_model.cfg_coef
                    tts_model.cfg_coef = 1.0

                model = "moshi_mlx"
                logger.info("✅ Moshi TTS model loaded successfully with MLX!")

            except Exception as e:
                logger.error(f"❌ Failed to load MLX model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                model = "dummy"
                device = "cpu"

        # PyTorch Backend (CUDA/CPU)
        elif BACKEND == "pytorch":
            logger.info("🔥 Loading Moshi TTS with PyTorch backend")

            # Determine device
            if settings.model_device:
                device = settings.model_device
                logger.info(f"Using forced device from config: {device}")
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Auto-detected device: {device}")

            try:
                from moshi.models.tts import TTSModel
                from moshi.models.loaders import CheckpointInfo

                logger.info(f"Loading from {settings.default_tts_repo}...")

                # Load model checkpoint
                checkpoint_info = CheckpointInfo.from_hf_repo(
                    settings.default_tts_repo,
                    None,  # moshi_weight
                    None,  # mimi_weight
                    None,  # tokenizer
                    None   # config
                )

                # Determine dtype
                if settings.model_dtype == "auto":
                    model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
                elif settings.model_dtype == "bfloat16":
                    model_dtype = torch.bfloat16
                else:
                    model_dtype = torch.float32

                # Create TTS model
                tts_model = TTSModel.from_checkpoint_info(
                    checkpoint_info,
                    voice_repo=settings.default_voice_repo,
                    n_q=settings.model_n_q,
                    temp=settings.model_temp,
                    cfg_coef=settings.model_cfg_coef,
                    device=device,
                    dtype=model_dtype
                )

                # Handle CFG distillation
                cfg_coef_conditioning = None
                if tts_model.valid_cfg_conditionings:
                    cfg_coef_conditioning = tts_model.cfg_coef
                    tts_model.cfg_coef = 1.0

                model = "moshi_pytorch"
                logger.info("✅ Moshi TTS model loaded successfully with PyTorch!")

            except ImportError as e:
                logger.warning(f"⚠️ Moshi library not available: {e}")
                logger.warning("⚠️ Using dummy model for testing")
                model = "dummy"
                device = "cpu"
            except Exception as e:
                logger.error(f"❌ Failed to load PyTorch model: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                model = "dummy"
                device = "cpu"

    except Exception as e:
        logger.error(f"❌ Unexpected error during model loading: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        model = "dummy"
        device = "cpu"

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting Moshi TTS API Server...")
    try:
        load_moshi_model()
    except Exception as e:
        logger.error(f"Startup error: {e}")
        # Continue running for testing
    logger.info("✅ API Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("🛑 Shutting down Moshi TTS API Server...")
    global model, tts_model
    if model is not None:
        if tts_model is not None:
            del tts_model
        del model
        # Only empty CUDA cache for PyTorch backend
        if BACKEND == "pytorch" and device == "cuda":
            torch.cuda.empty_cache()
    executor.shutdown(wait=True)
    logger.info("👋 Shutdown complete")

# Utility functions
def synthesize_text(text: str, language: str, voice: str = "default") -> np.ndarray:
    """
    Synthesize audio from text using available backend (MLX or PyTorch)
    Returns numpy array of audio samples
    """
    global tts_model

    if model == "dummy":
        # Generate dummy audio for testing
        logger.warning("Using dummy audio generation for testing")
        duration = min(len(text) * 0.05, 10)  # Rough estimate, max 10 seconds
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration))
        # Different frequency for different languages
        frequency = 440 if language == "en" else 494
        audio = np.sin(2 * np.pi * frequency * t) * 0.3
        # Add some variation
        audio += np.sin(2 * np.pi * frequency * 2 * t) * 0.1
        return audio

    # MLX Backend
    if model == "moshi_mlx":
        try:
            # Prepare text entries for TTS
            entries = tts_model.prepare_script([text])

            # Handle voice selection
            voices = []
            prefixes = None

            if tts_model.multi_speaker:
                # Multi-speaker model: pass voices to attributes
                if voice == "default":
                    voice = "vctk/p225_023.wav"
                voice_path = tts_model.get_voice_path(voice)
                voices = [voice_path]
            else:
                # Single-speaker model: use voice as prefix
                if voice == "default":
                    voice = "vctk/p225_023.wav"
                voice_path = tts_model.get_voice_path(voice)
                prefixes = [tts_model.get_prefix(voice_path)]

            # Create condition attributes
            attributes = tts_model.make_condition_attributes(voices, cfg_coef_conditioning)

            # Generate audio
            result = tts_model.generate(
                [entries],
                [attributes],
                prefixes=prefixes,
                cfg_is_no_prefix=(prefixes is None),
                cfg_is_no_text=True
            )

            # Decode frames to audio using MLX
            wav_frames = []
            for frame in result.frames:
                # Decode each frame
                _pcm = tts_model.mimi.decode_step(frame)
                wav_frames.append(_pcm)

            # Concatenate all frames
            wav = mx.concat(wav_frames, axis=-1)

            # Get the actual generated audio length
            end_step = result.end_steps[0]
            if end_step is not None:
                wav_length = int((tts_model.mimi.sample_rate * (end_step + tts_model.final_padding) / tts_model.mimi.frame_rate))
                wav = wav[0, :, :wav_length]
            else:
                wav = wav[0, :, :]

            # Convert MLX array to numpy
            audio_data = np.array(wav.squeeze())

            return audio_data

        except Exception as e:
            logger.error(f"MLX synthesis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    # PyTorch Backend
    elif model == "moshi_pytorch":
        try:
            with torch.no_grad():
                # Prepare text entries for TTS
                entries = tts_model.prepare_script([text])

                # Handle voice selection based on model type
                voices = []
                prefixes = None

                if tts_model.multi_speaker:
                    # Multi-speaker model: pass voices to attributes
                    if voice == "default":
                        voice = "vctk/p225_023.wav"
                    voice_path = tts_model.get_voice_path(voice)
                    voices = [voice_path]
                else:
                    # Single-speaker model: use voice as prefix
                    if voice == "default":
                        voice = "vctk/p225_023.wav"
                    voice_path = tts_model.get_voice_path(voice)
                    prefixes = [tts_model.get_prefix(voice_path)]

                # Create condition attributes with CFG distillation support
                attributes = tts_model.make_condition_attributes(voices, cfg_coef_conditioning)

                # Generate audio
                result = tts_model.generate(
                    [entries],
                    [attributes],
                    prefixes=prefixes,
                    cfg_is_no_prefix=(prefixes is None),
                    cfg_is_no_text=True
                )

                # Decode frames to audio
                wav_frames = []
                with tts_model.mimi.streaming(1):
                    for frame in result.frames[tts_model.delay_steps:]:
                        wav_frames.append(tts_model.mimi.decode(frame[:, 1:]))

                wav = torch.cat(wav_frames, dim=-1)

                # Get the actual generated audio length
                end_step = result.end_steps[0]
                if end_step is not None:
                    wav_length = int((tts_model.mimi.sample_rate * (end_step + tts_model.final_padding) / tts_model.mimi.frame_rate))
                    wav = wav[0, :, :wav_length]
                else:
                    wav = wav[0, :, :]

                # Convert to numpy and ensure mono
                audio_data = wav.squeeze().cpu().numpy()

                return audio_data

        except Exception as e:
            logger.error(f"PyTorch synthesis error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    else:
        raise RuntimeError(f"Unknown model type: {model}")

# API Endpoints

@app.get("/", tags=["System"])
async def root():
    """
    Root endpoint - API information
    
    Returns basic API information and usage examples
    """
    return {
        "api": API_TITLE,
        "version": API_VERSION,
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json"
        },
        "endpoints": {
            "tts": "/api/v1/tts",
            "health": "/api/v1/health",
            "languages": "/api/v1/languages",
            "voices": "/api/v1/voices"
        }
    }

@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health Check",
    description="Check the health status of the API and model"
)
async def health_check():
    """
    Health check endpoint

    Returns the current status of the service including:
    - Model loading status
    - Available languages
    - Computing device information
    - Backend type (MLX/PyTorch/dummy)
    """
    # Determine device display string
    device_str = device if device else "not initialized"
    if BACKEND == "mlx":
        device_str = f"mlx (Metal GPU)"
    elif BACKEND == "pytorch" and device == "cuda":
        device_str = f"cuda (GPU)"
    elif BACKEND == "pytorch" and device == "cpu":
        device_str = "cpu"

    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        device=device_str,
        available_languages=[lang.value for lang in LanguageCode],
        api_version=settings.api_version,
        timestamp=datetime.utcnow()
    )

@app.get(
    "/api/v1/languages",
    tags=["System"],
    summary="List Available Languages",
    description="Get list of supported languages for synthesis"
)
async def get_languages():
    """
    Get available languages

    Returns a list of supported languages with their codes and descriptions
    """
    return {
        "languages": [
            {
                "code": lang.value,
                "name": LanguageCode.get_description()[lang.value]
            }
            for lang in LanguageCode
        ]
    }

@app.get(
    "/api/v1/voices",
    tags=["System"],
    summary="List Available Voices",
    description="Get list of available voice presets"
)
async def get_voices():
    """
    Get available voice presets

    Returns a list of voice presets that can be used for synthesis
    """
    return {
        "voices": [
            {
                "id": voice.value,
                "name": voice.name,
                "description": _get_voice_description(voice.value)
            }
            for voice in VoicePreset
        ]
    }

def _get_voice_description(voice_id: str) -> str:
    """Get description for a voice preset"""
    descriptions = {
        # Default
        "default": "Default voice (VCTK p225)",

        # VCTK voices (English - Voice Cloning Toolkit)
        "vctk/p225_023.wav": "VCTK English - Female speaker p225",
        "vctk/p226_023.wav": "VCTK English - Male speaker p226",
        "vctk/p227_023.wav": "VCTK English - Male speaker p227",
        "vctk/p228_023.wav": "VCTK English - Female speaker p228",
        "vctk/p229_023.wav": "VCTK English - Female speaker p229",
        "vctk/p230_023.wav": "VCTK English - Female speaker p230",
        "vctk/p231_023.wav": "VCTK English - Female speaker p231",
        "vctk/p232_023.wav": "VCTK English - Male speaker p232",
        "vctk/p233_023.wav": "VCTK English - Female speaker p233",
        "vctk/p234_023.wav": "VCTK English - Female speaker p234",

        # CML-TTS French voices
        "cml-tts/fr/1406_1028_000009-0003.wav": "French - Speaker 1406",
        "cml-tts/fr/1591_1028_000108-0004.wav": "French - Speaker 1591",
        "cml-tts/fr/1770_1028_000036-0002.wav": "French - Speaker 1770",
        "cml-tts/fr/2114_1656_000053-0001.wav": "French - Speaker 2114",
        "cml-tts/fr/2154_2576_000020-0003.wav": "French - Speaker 2154",
        "cml-tts/fr/2216_1745_000007-0001.wav": "French - Speaker 2216",
        "cml-tts/fr/2223_1745_000009-0002.wav": "French - Speaker 2223",
        "cml-tts/fr/2465_1943_000152-0002.wav": "French - Speaker 2465",
        "cml-tts/fr/296_1028_000022-0001.wav": "French - Speaker 296",
        "cml-tts/fr/3267_1902_000075-0001.wav": "French - Speaker 3267",

        # Expresso voices (English with emotions/styles)
        "expresso/ex01-ex02_default_001_channel1_168s.wav": "English - Default speaking style",
        "expresso/ex01-ex02_enunciated_001_channel1_432s.wav": "English - Enunciated speaking style",
        "expresso/ex01-ex02_fast_001_channel1_104s.wav": "English - Fast speaking style",
        "expresso/ex01-ex02_projected_001_channel1_46s.wav": "English - Projected speaking style",
        "expresso/ex01-ex02_whisper_001_channel1_579s.wav": "English - Whisper speaking style",
        "expresso/ex03-ex01_angry_001_channel1_201s.wav": "English - Angry emotion",
        "expresso/ex03-ex01_happy_001_channel1_334s.wav": "English - Happy emotion",
        "expresso/ex03-ex01_calm_001_channel1_1143s.wav": "English - Calm emotion",
        "expresso/ex03-ex01_confused_001_channel1_909s.wav": "English - Confused emotion",

        # EARS voices (English - diverse speakers)
        "ears/p001/freeform_speech_01.wav": "EARS English - Speaker p001",
        "ears/p002/freeform_speech_01.wav": "EARS English - Speaker p002",
        "ears/p003/freeform_speech_01.wav": "EARS English - Speaker p003",
        "ears/p004/freeform_speech_01.wav": "EARS English - Speaker p004",
        "ears/p005/freeform_speech_01.wav": "EARS English - Speaker p005",
        "ears/p010/freeform_speech_01.wav": "EARS English - Speaker p010",
        "ears/p015/freeform_speech_01.wav": "EARS English - Speaker p015",
        "ears/p020/freeform_speech_01.wav": "EARS English - Speaker p020",
        "ears/p025/freeform_speech_01.wav": "EARS English - Speaker p025",
        "ears/p030/freeform_speech_01.wav": "EARS English - Speaker p030",
        "ears/p035/freeform_speech_01.wav": "EARS English - Speaker p035",
        "ears/p040/freeform_speech_01.wav": "EARS English - Speaker p040",
        "ears/p045/freeform_speech_01.wav": "EARS English - Speaker p045",
        "ears/p050/freeform_speech_01.wav": "EARS English - Speaker p050",
    }
    return descriptions.get(voice_id, "Unknown voice")

@app.post(
    "/api/v1/tts",
    tags=["TTS"],
    summary="Text-to-Speech Synthesis",
    description="Convert text to speech audio",
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "Audio file (WAV format)",
            "content": {"audio/wav": {}},
        },
        400: {"model": ErrorResponse, "description": "Invalid request"},
        503: {"model": ErrorResponse, "description": "Service unavailable"},
    }
)
async def synthesize(
    request: TTSRequest,
):
    """
    Synthesize speech from text
    
    Converts the provided text to speech audio using the Moshi model.
    
    ### Parameters:
    - **text**: The text to synthesize (1-5000 characters)
    - **language**: Language code (fr for French, en for English)
    - **format**: Output format (wav or raw)
    
    ### Returns:
    - Audio file in the requested format
    
    ### Example:
    ```bash
    curl -X POST "http://localhost:8000/api/v1/tts" \\
         -H "Content-Type: application/json" \\
         -d '{"text": "Bonjour monde", "language": "fr"}' \\
         --output audio.wav
    ```
    """
    
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service is temporarily unavailable."
        )
    
    try:
        logger.info(f"Synthesizing text in {request.language} with voice {request.voice}: '{request.text[:50]}...'")

        # Run synthesis in thread pool
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            executor,
            synthesize_text,
            request.text,
            request.language.value,
            request.voice.value
        )
        
        # Ensure audio is 1D
        if len(audio_data.shape) > 1:
            audio_data = audio_data.squeeze()
        
        # Normalize and convert to int16
        audio_data = np.clip(audio_data, -1, 1)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Calculate duration
        duration = len(audio_int16) / SAMPLE_RATE
        
        if request.format == AudioFormat.wav:
            # Create WAV file
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(SAMPLE_RATE)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_buffer.seek(0)
            file_size = wav_buffer.getbuffer().nbytes
            
            logger.info(f"✅ Generated {duration:.2f}s of audio ({file_size} bytes)")
            
            return StreamingResponse(
                wav_buffer,
                media_type="audio/wav",
                headers={
                    "Content-Disposition": f"attachment; filename=moshi_output_{request.language}.wav",
                    "X-Audio-Duration": str(duration),
                    "X-Audio-Language": request.language.value,
                    "X-Sample-Rate": str(SAMPLE_RATE)
                }
            )
        else:
            # Return raw PCM data
            raw_buffer = io.BytesIO(audio_int16.tobytes())
            
            return StreamingResponse(
                raw_buffer,
                media_type="audio/pcm",
                headers={
                    "Content-Disposition": f"attachment; filename=moshi_output_{request.language}.raw",
                    "X-Audio-Duration": str(duration),
                    "X-Audio-Language": request.language.value,
                    "X-Sample-Rate": str(SAMPLE_RATE),
                    "X-Sample-Width": "2",
                    "X-Channels": "1"
                }
            )
            
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Audio synthesis failed: {str(e)}"
        )

@app.post(
    "/api/v1/tts/file",
    tags=["TTS"],
    summary="Text-to-Speech from File",
    description="Convert text file to speech audio"
)
async def synthesize_from_file(
    file: UploadFile = File(..., description="Text file to synthesize"),
    language: LanguageCode = Query(LanguageCode.french, description="Language for synthesis")
):
    """
    Synthesize speech from uploaded text file
    
    Accepts a text file and converts its content to speech.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read file content
        content = await file.read()
        text = content.decode('utf-8')
        
        # Create request and process
        request = TTSRequest(text=text, language=language)
        return await synthesize(request)
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be valid UTF-8 text")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Custom exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid input",
            "detail": str(exc),
            "status_code": 400
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run with custom configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        use_colors=True
    )
