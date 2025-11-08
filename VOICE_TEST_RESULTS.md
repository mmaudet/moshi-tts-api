# Voice Testing Results

## Test Summary

- **Total voices**: 40
- **Working**: 40 (100%)
- **Failed**: 0 (0%)

## Working Voices by Category

### Default (1 voice)
- ✅ default

### VCTK English (10 voices)
- ✅ vctk_p225
- ✅ vctk_p226
- ✅ vctk_p227
- ✅ vctk_p228
- ✅ vctk_p229
- ✅ vctk_p230
- ✅ vctk_p231
- ✅ vctk_p232
- ✅ vctk_p233
- ✅ vctk_p234

### CML-TTS French (10 voices)
- ✅ cml_fr_1406
- ✅ cml_fr_1591
- ✅ cml_fr_1770
- ✅ cml_fr_2114
- ✅ cml_fr_2154
- ✅ cml_fr_2216
- ✅ cml_fr_2223
- ✅ cml_fr_2465
- ✅ cml_fr_296
- ✅ cml_fr_3267

### Expresso Emotions (9 voices)
- ✅ expresso_default
- ✅ expresso_enunciated
- ✅ expresso_fast
- ✅ expresso_projected
- ✅ expresso_whisper
- ✅ expresso_angry
- ✅ expresso_happy
- ✅ expresso_calm
- ✅ expresso_confused

### EARS English (10 voices)
- ✅ ears_p001
- ✅ ears_p002
- ✅ ears_p003
- ✅ ears_p004
- ✅ ears_p005
- ✅ ears_p010
- ✅ ears_p015
- ✅ ears_p020
- ✅ ears_p025
- ✅ ears_p030

## Known Issues & Solutions

### Permission Denied Errors (FIXED)

**Problem**: When using Docker with a non-root user (appuser, UID 1001), Hugging Face Hub downloads models with the host user's permissions (typically ubuntu, UID 1000). This caused permission denied errors when trying to access voice files.

**Solution**: Fix permissions on the models directory:

```bash
# On the host machine
sudo chown -R 1001:1001 ./models
```

Or add this to your setup:

```bash
# After first run with models downloaded
docker compose down
sudo chown -R 1001:1001 ./models
docker compose up -d
```

**Prevention**: The issue occurs because:
1. Container runs as appuser (UID 1001) for security
2. HuggingFace Hub downloads files as the volume owner (UID 1000)
3. appuser cannot access files owned by UID 1000

## Testing

To test all voices:

```bash
python3 test_all_voices.py
```

This will:
- Test all 40 voices
- Report success/failure for each
- Group results by category
- Show detailed error messages for failures

## Example API Calls

### French voice
```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bonjour, ceci est un test.",
    "language": "fr",
    "voice": "cml-tts/fr/1406_1028_000009-0003.wav"
  }' \
  --output test_fr.wav
```

### English voice with emotion
```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, this is a happy test!",
    "language": "en",
    "voice": "expresso/ex03-ex01_happy_001_channel1_334s.wav"
  }' \
  --output test_happy.wav
```

### VCTK English voice
```bash
curl -X POST http://localhost:8000/api/v1/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Testing voice cloning.",
    "language": "en",
    "voice": "vctk/p225_023.wav"
  }' \
  --output test_vctk.wav
```

## Last Tested

- Date: 2025-11-08
- Docker Image: `moshi-tts-api:latest` (CUDA 12.6.3, Ubuntu 24.04, Python 3.12)
- All voices: ✅ PASSING
