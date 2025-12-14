#!/usr/bin/env python
"""
Entry point untuk production deployment dengan Railway/Heroku.
"""
import os
import sys
from waitress import serve

# Import Flask app
from app import app, use_model, model_info

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print("=" * 80)
    print("🚀 CARDIOVASCULAR RISK PREDICTION API v6.0")
    print("=" * 80)
    print(f"Server: Waitress")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Model: {'✅ LOADED' if use_model else '❌ NOT LOADED'}")
    
    if use_model:
        print(f"Version: {model_info.get('version', 'unknown')}")
    
    print("=" * 80)
    
    # Serve dengan waitress
    serve(app, host=host, port=port, threads=4)