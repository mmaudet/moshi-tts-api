# Configuration Docker Hub pour GitHub Actions

Ce guide vous explique comment configurer l'automatisation de la publication des images Docker sur Docker Hub.

## 📋 Prérequis

1. **Compte Docker Hub** (gratuit)
   - Créez un compte sur https://hub.docker.com si vous n'en avez pas

2. **Repository Docker Hub**
   - Créez un repository public nommé `moshi-tts-api` sur Docker Hub
   - URL: https://hub.docker.com/repository/create

## 🔐 Étape 1: Créer un Access Token Docker Hub

1. Connectez-vous sur https://hub.docker.com
2. Allez dans **Account Settings** → **Security** → **Access Tokens**
3. Cliquez sur **New Access Token**
4. Donnez un nom: `github-actions-moshi-tts-api`
5. Permissions: **Read & Write**
6. Cliquez sur **Generate**
7. **IMPORTANT**: Copiez le token immédiatement (il ne sera plus visible après)

## 🔧 Étape 2: Configurer les GitHub Secrets

1. Allez sur votre repository GitHub: https://github.com/mmaudet/moshi-tts-api
2. Cliquez sur **Settings** → **Secrets and variables** → **Actions**
3. Cliquez sur **New repository secret**

Créez ces deux secrets:

### Secret 1: DOCKERHUB_USERNAME
- **Name**: `DOCKERHUB_USERNAME`
- **Value**: Votre nom d'utilisateur Docker Hub (ex: `mmaudet`)

### Secret 2: DOCKERHUB_TOKEN
- **Name**: `DOCKERHUB_TOKEN`
- **Value**: Le token que vous avez copié à l'étape 1

## ✅ Étape 3: Vérifier la configuration

Une fois les secrets configurés:

1. Faites un push vers la branche `main`:
   ```bash
   git add .github/
   git commit -m "ci: Add Docker Hub automated publishing"
   git push origin main
   ```

2. Allez dans l'onglet **Actions** de votre repository GitHub
3. Vous devriez voir le workflow "Build and Push Docker Image" en cours d'exécution
4. Après ~5-10 minutes, vérifiez sur Docker Hub que l'image est publiée

## 🏷️ Tags automatiques

Le workflow crée automatiquement ces tags:

### Sur push vers `main`:
- `latest` - Toujours la dernière version de main
- `main-abc1234` - SHA du commit pour traçabilité

### Sur release (ex: `v1.0.0`):
- `1.0.0` - Version complète
- `1.0` - Version majeure.mineure
- `1` - Version majeure
- `latest` - Dernière release stable

## 📦 Utilisation des images

Une fois publiées, les utilisateurs pourront faire:

```bash
# Dernière version
docker pull mmaudet/moshi-tts-api:latest

# Version spécifique
docker pull mmaudet/moshi-tts-api:1.0.0

# Via docker-compose (mettez à jour docker-compose.yml)
services:
  moshi-tts-api:
    image: mmaudet/moshi-tts-api:latest
    # au lieu de: build: .
```

## 🔍 Vérification

Pour vérifier que tout fonctionne:

```bash
# Téléchargez l'image publiée
docker pull mmaudet/moshi-tts-api:latest

# Testez-la
docker run --rm --gpus all -p 8000:8000 mmaudet/moshi-tts-api:latest
```

## 🛠️ Mise à jour du README.md

N'oubliez pas de mettre à jour le README.md pour indiquer aux utilisateurs qu'ils peuvent utiliser l'image pré-buildée:

```markdown
## 🚀 Quick Start

### Option 1: Using pre-built image (recommended)
```bash
docker pull mmaudet/moshi-tts-api:latest
docker run --rm --gpus all -p 8000:8000 mmaudet/moshi-tts-api:latest
```

### Option 2: Build from source
```bash
git clone https://github.com/mmaudet/moshi-tts-api.git
cd moshi-tts-api
docker compose up -d
```
```

## 🎯 Avantages

✅ Build automatique à chaque push
✅ Versioning automatique
✅ Cache GitHub Actions (builds plus rapides)
✅ Description Docker Hub synchronisée avec README
✅ Les utilisateurs n'ont plus besoin de builder localement
✅ Distribution facilitée de votre API

## 🐛 Dépannage

### Le workflow échoue avec "unauthorized"
- Vérifiez que `DOCKERHUB_USERNAME` et `DOCKERHUB_TOKEN` sont correctement configurés
- Vérifiez que le token Docker Hub a les permissions "Read & Write"

### L'image ne se met pas à jour
- Vérifiez que le workflow s'est bien exécuté dans l'onglet Actions
- Le push vers Docker Hub ne se fait que sur la branche `main`, pas sur les PRs

### Le README ne se synchronise pas
- La synchronisation du README nécessite que le repository Docker Hub existe
- Vérifiez les logs du workflow dans GitHub Actions
