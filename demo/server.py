#!/usr/bin/env python3
"""
Simple HTTP server for CAD Validator Pro demo
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

class DemoHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

def main():
    # Change to demo directory
    demo_dir = Path(__file__).parent
    os.chdir(demo_dir)
    
    PORT = 8000
    
    with socketserver.TCPServer(("", PORT), DemoHTTPRequestHandler) as httpd:
        print(f"""
🚀 CAD Validator Pro Demo Server
================================
Server running at: http://localhost:{PORT}
Open your browser and navigate to the URL above

Features:
✅ Interactive file upload
✅ Real-time validation simulation
✅ Sample files to test
✅ Export functionality
✅ Responsive design

Press Ctrl+C to stop the server
        """)
        
        # Auto-open browser
        webbrowser.open(f'http://localhost:{PORT}')
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Demo server stopped")

if __name__ == "__main__":
    main()
