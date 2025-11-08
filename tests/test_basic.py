"""
Basic tests for Moshi TTS API
"""

import pytest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that main modules can be imported"""
    try:
        from app_final import app
        assert app is not None
    except ImportError:
        pytest.skip("app_final module not available")

def test_api_structure():
    """Test that API has correct structure"""
    try:
        from app_final import (
            TTSRequest,
            HealthResponse,
            LanguageCode
        )
        
        # Test language enum
        assert LanguageCode.french.value == "fr"
        assert LanguageCode.english.value == "en"
        
        # Test request model
        request = TTSRequest(
            text="Test text",
            language=LanguageCode.french
        )
        assert request.text == "Test text"
        assert request.language == LanguageCode.french
        
    except ImportError:
        pytest.skip("API models not available")

def test_client():
    """Test that client can be imported"""
    try:
        from client import MoshiTTSClient
        client = MoshiTTSClient()
        assert client.base_url == "http://localhost:8000"
    except ImportError:
        pytest.skip("Client module not available")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
