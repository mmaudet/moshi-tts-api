#!/usr/bin/env python3
"""
Moshi TTS API Client
====================
Python client for easy integration with the Moshi TTS API
"""

import requests
import json
from pathlib import Path
from typing import Optional, Literal
import argparse
import sys

class MoshiTTSClient:
    """Client for Moshi TTS API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the Moshi TTS client
        
        Args:
            base_url: Base URL of the API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json"
        })
    
    def health_check(self) -> dict:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/api/v1/health")
        response.raise_for_status()
        return response.json()
    
    def get_languages(self) -> dict:
        """Get available languages"""
        response = self.session.get(f"{self.base_url}/api/v1/languages")
        response.raise_for_status()
        return response.json()
    
    def synthesize(
        self,
        text: str,
        language: Literal["fr", "en"] = "fr",
        format: Literal["wav", "raw"] = "wav",
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            language: Language code ("fr" or "en")
            format: Output format ("wav" or "raw")
            output_file: Optional path to save the audio file
            
        Returns:
            Audio data as bytes
        """
        payload = {
            "text": text,
            "language": language,
            "format": format
        }
        
        response = self.session.post(
            f"{self.base_url}/api/v1/synthesize",
            json=payload
        )
        response.raise_for_status()
        
        audio_data = response.content
        
        if output_file:
            Path(output_file).write_bytes(audio_data)
            print(f"✅ Audio saved to: {output_file}")
        
        return audio_data
    
    def synthesize_file(
        self,
        file_path: str,
        language: Literal["fr", "en"] = "fr",
        output_file: Optional[str] = None
    ) -> bytes:
        """
        Synthesize text from file
        
        Args:
            file_path: Path to text file
            language: Language code
            output_file: Optional output file path
            
        Returns:
            Audio data as bytes
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        return self.synthesize(text, language, output_file=output_file)
    
    def batch_synthesize(
        self,
        texts: list,
        language: Literal["fr", "en"] = "fr",
        output_dir: str = "."
    ):
        """
        Synthesize multiple texts
        
        Args:
            texts: List of texts to synthesize
            language: Language code
            output_dir: Directory to save audio files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        for i, text in enumerate(texts, 1):
            output_file = output_dir / f"audio_{i:03d}.wav"
            try:
                self.synthesize(text, language, output_file=str(output_file))
                print(f"[{i}/{len(texts)}] Generated: {output_file}")
            except Exception as e:
                print(f"[{i}/{len(texts)}] Failed: {e}")


def main():
    """Command-line interface"""
    parser = argparse.ArgumentParser(
        description="Moshi TTS API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Synthesize text in French
  %(prog)s -t "Bonjour monde" -l fr -o bonjour.wav
  
  # Synthesize text in English
  %(prog)s -t "Hello world" -l en -o hello.wav
  
  # Synthesize from file
  %(prog)s -f input.txt -l fr -o output.wav
  
  # Check API health
  %(prog)s --health
        """
    )
    
    parser.add_argument(
        "-t", "--text",
        help="Text to synthesize"
    )
    parser.add_argument(
        "-f", "--file",
        help="Text file to synthesize"
    )
    parser.add_argument(
        "-l", "--language",
        choices=["fr", "en"],
        default="fr",
        help="Language (default: fr)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output audio file"
    )
    parser.add_argument(
        "--format",
        choices=["wav", "raw"],
        default="wav",
        help="Output format (default: wav)"
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Check API health"
    )
    parser.add_argument(
        "--languages",
        action="store_true",
        help="List available languages"
    )
    
    args = parser.parse_args()
    
    # Create client
    client = MoshiTTSClient(args.url)
    
    try:
        # Health check
        if args.health:
            health = client.health_check()
            print("🏥 API Health Status:")
            print(json.dumps(health, indent=2))
            return
        
        # List languages
        if args.languages:
            languages = client.get_languages()
            print("🌐 Available Languages:")
            for lang in languages["languages"]:
                print(f"  - {lang['code']}: {lang['name']}")
            return
        
        # Synthesize text
        if args.text:
            if not args.output:
                args.output = f"output_{args.language}.{args.format}"
            
            client.synthesize(
                args.text,
                args.language,
                args.format,
                args.output
            )
            print(f"🎵 Audio duration: Check {args.output}")
            
        # Synthesize from file
        elif args.file:
            if not args.output:
                base_name = Path(args.file).stem
                args.output = f"{base_name}_{args.language}.wav"
            
            client.synthesize_file(
                args.file,
                args.language,
                args.output
            )
            print(f"🎵 Audio generated from {args.file}")
            
        else:
            parser.print_help()
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Is the server running?")
        print(f"   URL: {args.url}")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        if e.response.text:
            print(f"   Details: {e.response.text}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
