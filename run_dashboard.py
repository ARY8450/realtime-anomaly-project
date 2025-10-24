
import os
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import threading
import time

def start_server():
    '''Start a simple HTTP server to serve the dashboard'''
    port = 8080
    os.chdir('frontend_dashboard')
    
    try:
        server = HTTPServer(('localhost', port), SimpleHTTPRequestHandler)
        print(f"Dashboard server starting on http://localhost:{port}")
        print("Open your browser and navigate to the URL above")
        print("Press Ctrl+C to stop the server")
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    start_server()
        