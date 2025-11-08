# 🎙️ Moshi TTS API

[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-red.svg)](LICENSE)

API REST pour la synthèse vocale utilisant le modèle [Moshi de Kyutai Labs](https://github.com/kyutai-labs/moshi), avec documentation Swagger interactive et déploiement Docker.

## ✨ Fonctionnalités

- 🌐 **Support bilingue** : Français et Anglais
- 📖 **Documentation Swagger** : Interface interactive pour tester l'API
- 🎵 **Audio haute qualité** : 24kHz en format WAV ou RAW
- 🚀 **Support GPU** : Accélération CUDA automatique
- 🔒 **Sécurisé** : Utilisateur non-root, validation des entrées
- 📦 **Docker** : Déploiement simple et reproductible
- 🔄 **API RESTful** : Endpoints bien structurés avec OpenAPI
- 📊 **Health checks** : Monitoring de l'état du service

## 🚀 Installation rapide

### Prérequis
- Docker installé
- NVIDIA Docker Runtime (optionnel, pour support GPU)
- Au moins 8GB de RAM
- ~10GB d'espace disque pour le modèle

### Installation

1. **Cloner ou créer le projet**
```bash
mkdir moshi-tts-api
cd moshi-tts-api
# Copier tous les fichiers fournis
```

2. **Build et lancement rapide**
```bash
chmod +x build-and-run.sh
./build-and-run.sh
```

Ou manuellement :

```bash
# Build
docker build -t moshi-tts-api:latest .

# Run avec GPU
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    --gpus all \
    moshi-tts-api:latest

# Run sans GPU (CPU uniquement)
docker run -d --name moshi-tts-api \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    moshi-tts-api:latest
```

### Avec Docker Compose

```bash
# Avec GPU
docker-compose up -d

# Sans GPU (éditer docker-compose.yml pour retirer la section deploy)
docker-compose up -d
```

## 📖 Utilisation

### Documentation Interactive (Swagger)

Une fois l'API démarrée, accédez à la documentation interactive :

- **Swagger UI** : http://localhost:8000/docs
- **ReDoc** : http://localhost:8000/redoc
- **OpenAPI JSON** : http://localhost:8000/openapi.json

### Test rapide avec le script
```bash
chmod +x test_api.sh
./test_api.sh
```

### Exemples d'utilisation avec cURL

#### Synthèse en français
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Bonjour, je suis Moshi, votre assistant vocal.",
       "language": "fr"
     }' \
     --output bonjour.wav
```

#### Synthèse en anglais
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Hello, I am Moshi, your voice assistant.",
       "language": "en"
     }' \
     --output hello.wav
```

#### Format RAW (PCM)
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Test audio",
       "language": "en",
       "format": "raw"
     }' \
     --output test.raw

# Convertir RAW en WAV
ffmpeg -f s16le -ar 24000 -ac 1 -i test.raw output.wav
```

### Endpoints disponibles

#### 1. **GET /** - Information sur l'API
```bash
curl http://localhost:8000/
```

#### 2. **GET /api/v1/health** - État de santé
```bash
curl http://localhost:8000/api/v1/health
```
Réponse :
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "available_languages": ["fr", "en"],
  "api_version": "1.0.0",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### 3. **GET /api/v1/languages** - Langues disponibles
```bash
curl http://localhost:8000/api/v1/languages
```
Réponse :
```json
{
  "languages": [
    {"code": "fr", "name": "French (Français)"},
    {"code": "en", "name": "English"}
  ]
}
```

#### 4. **POST /api/v1/synthesize** - Génération de voix
```bash
curl -X POST http://localhost:8000/api/v1/synthesize \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Votre texte ici",
       "language": "fr",
       "format": "wav"
     }' \
     --output audio.wav
```

Paramètres :
- `text` (requis) : Le texte à synthétiser (1-5000 caractères)
- `language` (optionnel, défaut: "fr") : Code langue ("fr" ou "en")  
- `format` (optionnel, défaut: "wav") : Format de sortie ("wav" ou "raw")

#### 5. **POST /api/v1/synthesize/file** - Synthèse depuis fichier
```bash
curl -X POST http://localhost:8000/api/v1/synthesize/file \
     -F "file=@mon_texte.txt" \
     -F "language=fr" \
     --output audio.wav
```

## 🔧 Configuration avancée

### Variables d'environnement

```bash
# Spécifier le GPU à utiliser
docker run -e CUDA_VISIBLE_DEVICES=0 ...

# Changer le cache des modèles
docker run -e HF_HOME=/custom/path ...

# Désactiver le cache de transformers
docker run -e TRANSFORMERS_OFFLINE=1 ...
```

### Personnalisation du modèle

Modifier `app.py` pour changer le modèle :
```python
model = loaders.load_moshi_model(
    "kyutai/moshika-pytorch-bf16",  # ou un autre modèle
    device=device
)
```

### Performance

- **GPU** : Génération en temps réel ou plus rapide
- **CPU** : Génération plus lente (2-10x temps réel selon CPU)
- **Mémoire** : ~6GB pour le modèle en bf16
- **Première requête** : Plus lente (chargement du modèle)

## 🐳 Commandes Docker utiles

```bash
# Voir les logs
docker logs -f moshi-tts-api

# Arrêter le container
docker stop moshi-tts-api

# Redémarrer
docker restart moshi-tts-api

# Supprimer le container
docker rm -f moshi-tts-api

# Nettoyer l'image
docker rmi moshi-tts-api:latest

# Entrer dans le container
docker exec -it moshi-tts-api bash
```

## 🔍 Débogage

### L'API ne démarre pas
```bash
# Vérifier les logs
docker logs moshi-tts-api

# Vérifier que le port 8000 est libre
lsof -i :8000
```

### Erreur GPU
```bash
# Vérifier NVIDIA Docker
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Mémoire insuffisante
- Utiliser un modèle plus petit
- Augmenter la mémoire Docker
- Utiliser le mode CPU

## 📦 Build multi-architecture

Pour créer une image compatible ARM64 et AMD64 :
```bash
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
    -t moshi-tts-api:latest --push .
```

## 🤝 Intégration

### Python
```python
import requests
import base64

response = requests.post(
    "http://localhost:8000/tts",
    json={"text": "Hello world"}
)

with open("output.wav", "wb") as f:
    f.write(response.content)
```

### Node.js
```javascript
const axios = require('axios');
const fs = require('fs');

axios.post('http://localhost:8000/tts', {
    text: 'Hello world'
}, {
    responseType: 'arraybuffer'
}).then(response => {
    fs.writeFileSync('output.wav', response.data);
});
```

### n8n Integration
Utilisez le node HTTP Request avec :
- Method: POST
- URL: http://localhost:8000/tts
- Body: JSON avec `{"text": "votre texte"}`
- Response Format: File

## 📄 Licence

Ce projet utilise Moshi de Kyutai Labs. Consultez leur [licence](https://github.com/kyutai-labs/moshi/blob/main/LICENSE).

Ce wrapper API est sous licence MIT - voir [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contributing

Les contributions sont les bienvenues ! N'hésitez pas à :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push sur la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📸 Screenshots

### Swagger UI
La documentation interactive permet de tester tous les endpoints directement depuis le navigateur :

- `/docs` - Interface Swagger UI
- `/redoc` - Documentation alternative ReDoc
- `/openapi.json` - Spécification OpenAPI

## 🙏 Remerciements

- [Kyutai Labs](https://github.com/kyutai-labs) pour le modèle Moshi
- [FastAPI](https://fastapi.tiangolo.com/) pour le framework web
- [Docker](https://www.docker.com/) pour la containerisation

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

⭐ Si ce projet vous est utile, n'oubliez pas de lui donner une étoile sur GitHub !
