#!/usr/bin/env python
"""
Entry point script untuk production deployment dengan Railway/Heroku.
Handles environment variable expansion dan server startup.
"""

import os
import sys
from waitress import serve

# Import Flask app dari app.py
from app import app

if __name__ == '__main__':
    # Dapatkan PORT dari environment variable, default ke 8080
    port = int(os.environ.get('PORT', 8080))
    host = os.environ.get('HOST', '0.0.0.0')
    
    print(f"🚀 Starting Waitress server on {host}:{port}")
    
    # Serve dengan waitress
    serve(app, host=host, port=port)
