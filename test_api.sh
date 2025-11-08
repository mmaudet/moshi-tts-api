#!/bin/bash

# Test script for Moshi TTS API
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_URL="http://localhost:8000"

echo "🧪 Testing Moshi TTS API"
echo "========================"
echo ""

# Function to check if API is running
check_api() {
    echo -n "Checking if API is running... "
    if curl -s -f "$API_URL/api/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ API is running${NC}"
        return 0
    else
        echo -e "${RED}✗ API is not running${NC}"
        echo "Please start the API first with: docker-compose up -d"
        exit 1
    fi
}

# Test 1: Root endpoint
test_root() {
    echo ""
    echo "📍 Test 1: Root Endpoint"
    echo "------------------------"
    curl -s "$API_URL/" | python3 -m json.tool
}

# Test 2: Health check
test_health() {
    echo ""
    echo "🏥 Test 2: Health Check"
    echo "-----------------------"
    curl -s "$API_URL/api/v1/health" | python3 -m json.tool
}

# Test 3: List languages
test_languages() {
    echo ""
    echo "🌐 Test 3: Available Languages"
    echo "------------------------------"
    curl -s "$API_URL/api/v1/languages" | python3 -m json.tool
}

# Test 4: Synthesize French text
test_synthesis_fr() {
    echo ""
    echo "🇫🇷 Test 4: French Synthesis"
    echo "---------------------------"
    echo "Generating French audio..."
    
    curl -X POST "$API_URL/api/v1/synthesize" \
         -H "Content-Type: application/json" \
         -d '{
           "text": "Bonjour, je suis Moshi. Je peux parler en français avec un accent naturel.",
           "language": "fr"
         }' \
         --output test_fr.wav \
         -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\nSize: %{size_download} bytes\n"
    
    if [ -f "test_fr.wav" ]; then
        echo -e "${GREEN}✓ French audio saved to test_fr.wav${NC}"
        file test_fr.wav
    else
        echo -e "${RED}✗ Failed to generate French audio${NC}"
    fi
}

# Test 5: Synthesize English text
test_synthesis_en() {
    echo ""
    echo "🇬🇧 Test 5: English Synthesis"
    echo "----------------------------"
    echo "Generating English audio..."
    
    curl -X POST "$API_URL/api/v1/synthesize" \
         -H "Content-Type: application/json" \
         -d '{
           "text": "Hello, I am Moshi. I can speak English with a natural voice.",
           "language": "en"
         }' \
         --output test_en.wav \
         -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\nSize: %{size_download} bytes\n"
    
    if [ -f "test_en.wav" ]; then
        echo -e "${GREEN}✓ English audio saved to test_en.wav${NC}"
        file test_en.wav
    else
        echo -e "${RED}✗ Failed to generate English audio${NC}"
    fi
}

# Test 6: Raw format output
test_raw_format() {
    echo ""
    echo "🎵 Test 6: Raw Audio Format"
    echo "---------------------------"
    echo "Generating raw PCM audio..."
    
    curl -X POST "$API_URL/api/v1/synthesize" \
         -H "Content-Type: application/json" \
         -d '{
           "text": "Test audio raw format",
           "language": "en",
           "format": "raw"
         }' \
         --output test.raw \
         -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\nSize: %{size_download} bytes\n"
    
    if [ -f "test.raw" ]; then
        echo -e "${GREEN}✓ Raw audio saved to test.raw${NC}"
        echo "Convert to WAV with: ffmpeg -f s16le -ar 24000 -ac 1 -i test.raw test_converted.wav"
    else
        echo -e "${RED}✗ Failed to generate raw audio${NC}"
    fi
}

# Test 7: Long text
test_long_text() {
    echo ""
    echo "📚 Test 7: Long Text Synthesis"
    echo "------------------------------"
    
    LONG_TEXT="Ceci est un test avec un texte plus long. Moshi est capable de synthétiser des textes de plusieurs phrases avec une qualité constante. La synthèse vocale moderne utilise des réseaux de neurones profonds pour générer une voix naturelle et expressive. Cette technologie permet de créer des assistants vocaux, des livres audio, et bien d'autres applications."
    
    curl -X POST "$API_URL/api/v1/synthesize" \
         -H "Content-Type: application/json" \
         -d "{
           \"text\": \"$LONG_TEXT\",
           \"language\": \"fr\"
         }" \
         --output test_long.wav \
         -w "\nHTTP Status: %{http_code}\nTime: %{time_total}s\nSize: %{size_download} bytes\n"
    
    if [ -f "test_long.wav" ]; then
        echo -e "${GREEN}✓ Long text audio saved to test_long.wav${NC}"
    else
        echo -e "${RED}✗ Failed to generate long text audio${NC}"
    fi
}

# Test 8: Error handling - empty text
test_error_empty() {
    echo ""
    echo "⚠️ Test 8: Error Handling - Empty Text"
    echo "-------------------------------------"
    
    RESPONSE=$(curl -s -X POST "$API_URL/api/v1/synthesize" \
         -H "Content-Type: application/json" \
         -d '{
           "text": "",
           "language": "fr"
         }')
    
    echo "$RESPONSE" | python3 -m json.tool
}

# Test 9: Check Swagger documentation
test_swagger() {
    echo ""
    echo "📖 Test 9: API Documentation"
    echo "---------------------------"
    
    if curl -s -f "$API_URL/docs" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Swagger UI available at: $API_URL/docs${NC}"
    else
        echo -e "${RED}✗ Swagger UI not accessible${NC}"
    fi
    
    if curl -s -f "$API_URL/redoc" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ ReDoc available at: $API_URL/redoc${NC}"
    else
        echo -e "${RED}✗ ReDoc not accessible${NC}"
    fi
    
    if curl -s -f "$API_URL/openapi.json" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ OpenAPI spec available at: $API_URL/openapi.json${NC}"
    else
        echo -e "${RED}✗ OpenAPI spec not accessible${NC}"
    fi
}

# Main execution
main() {
    check_api
    
    echo ""
    echo -e "${YELLOW}Starting tests...${NC}"
    
    test_root
    test_health
    test_languages
    test_synthesis_fr
    test_synthesis_en
    test_raw_format
    test_long_text
    test_error_empty
    test_swagger
    
    echo ""
    echo "🎉 Testing complete!"
    echo ""
    echo "Generated files:"
    ls -la test*.wav test*.raw 2>/dev/null || echo "No audio files generated"
    
    echo ""
    echo "To play audio files:"
    echo "  ffplay test_fr.wav"
    echo "  ffplay test_en.wav"
    echo ""
    echo "To convert raw to WAV:"
    echo "  ffmpeg -f s16le -ar 24000 -ac 1 -i test.raw test_converted.wav"
}

# Run main function
main
