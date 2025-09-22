#!/usr/bin/env python3
"""
Simple script to run the RAG Chatbot application.
"""
import subprocess
import sys
import os

def main():
    """Run the Streamlit chatbot application."""
    # Change to the src directory
    src_path = os.path.join(os.path.dirname(__file__), 'src')
    app_path = os.path.join(src_path, 'chatbot_final.py')
    
    if not os.path.exists(app_path):
        print("❌ Error: chatbot_app.py not found in src directory")
        sys.exit(1)
    
    print("🚀 Starting RAG Chatbot...")
    print("📝 Open your browser to http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()