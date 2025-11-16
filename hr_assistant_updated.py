#!/usr/bin/env python3
"""
Nestlé HR Policy Assistant - Standalone Python Script
AI-Powered HR Assistant for Nestlé's HR Policy Documents
"""

import os
import sys
import getpass
import subprocess
from pathlib import Path

def install_packages():
    """Install required packages"""
    print("Installing required packages...")
    
    # Uninstall conflicting packages
    subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pydantic', 'chromadb', 'gradio', 'fastapi'], 
                  capture_output=True)
    
    # Install working versions in correct order
    packages = [
        'pydantic==1.10.12',
        'chromadb==0.4.15', 
        'gradio==3.50.2',
        'langchain==0.1.20',
        'langchain-community==0.0.38',
        'langchain-openai==0.1.7',
        'langchain-core==0.1.52',
        'pypdf',
        'openai'
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        result = subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Failed to install {package}")
    
    # Try to install pysqlite3-binary with fallback
    print("Installing pysqlite3-binary...")
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', 'pysqlite3-binary'], 
                          capture_output=True, text=True)
    if result.returncode != 0:
        print("Warning: pysqlite3-binary failed. Trying alternative...")
        # Try without binary
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'pysqlite3'], 
                      capture_output=True, text=True)

def setup_sqlite():
    """Fix SQLite compatibility"""
    try:
        import pysqlite3
        sys.modules['sqlite3'] = pysqlite3
        print("✅ Using pysqlite3 for ChromaDB compatibility")
    except ImportError:
        print("⚠️ pysqlite3 not available, using system SQLite (may cause issues with ChromaDB)")
        # Try to use system sqlite3 with compatibility check
        import sqlite3
        if hasattr(sqlite3, 'enable_load_extension'):
            print("✅ System SQLite supports extensions")
        else:
            print("⚠️ System SQLite may not be fully compatible with ChromaDB")

def test_network_connectivity():
    """Test network connectivity to OpenAI"""
    import urllib.request
    import socket
    
    print("🔍 Testing network connectivity...")
    
    # Test basic internet
    try:
        urllib.request.urlopen('https://www.google.com', timeout=5)
        print("✅ Basic internet connection: OK")
    except:
        print("❌ No internet connection")
        return False
    
    # Test OpenAI API endpoint
    try:
        urllib.request.urlopen('https://api.openai.com', timeout=10)
        print("✅ OpenAI API endpoint: Reachable")
        return True
    except Exception as e:
        print(f"❌ OpenAI API endpoint: Blocked ({e})")
        print("🔍 This suggests corporate firewall/proxy is blocking OpenAI")
        return False

def get_api_key():
    """Prompt user for OpenAI API key"""
    api_key = getpass.getpass("Enter your OpenAI API Key: ")
    if not api_key.strip():
        raise ValueError("OpenAI API key is required")
    os.environ['OPENAI_API_KEY'] = api_key
    return api_key

def get_pdf_path():
    """Get PDF file path from user"""
    while True:
        pdf_path = input("Enter the path to your HR policy PDF file: ").strip()
        if not pdf_path:
            print("PDF path is required")
            continue
        
        # Remove quotes if present
        pdf_path = pdf_path.strip('"').strip("'")
        
        pdf_file = Path(pdf_path)
        
        # If it's a directory, look for PDF files
        if pdf_file.is_dir():
            pdf_files = list(pdf_file.glob('*.pdf'))
            if pdf_files:
                print(f"Found PDF files in directory:")
                for i, file in enumerate(pdf_files, 1):
                    print(f"{i}. {file.name}")
                
                while True:
                    try:
                        choice = int(input("Select PDF file number: ")) - 1
                        if 0 <= choice < len(pdf_files):
                            return str(pdf_files[choice])
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a valid number")
            else:
                print("No PDF files found in the directory")
                continue
        
        # Check if it's a valid PDF file
        elif pdf_file.exists() and pdf_file.suffix.lower() == '.pdf':
            return str(pdf_file)
        else:
            print(f"File not found or not a PDF: {pdf_path}")
            print("Please provide either:")
            print("1. Full path to PDF file (e.g., C:\\folder\\file.pdf)")
            print("2. Directory path containing PDF files")

def main():
    """Main application function"""
    print("🤖 Nestlé HR Policy Assistant Setup")
    print("=" * 50)
    
    # Install packages
    try:
        install_packages()
        print("✅ Packages installed successfully")
    except Exception as e:
        print(f"❌ Package installation failed: {e}")
        return
    
    # Setup SQLite
    setup_sqlite()
    
    # Test network connectivity first
    if not test_network_connectivity():
        print("\n🚨 NETWORK ISSUE DETECTED")
        print("Your corporate network/firewall is likely blocking OpenAI API.")
        print("\n🔧 Possible solutions:")
        print("1. Use personal internet/hotspot")
        print("2. Contact IT to whitelist api.openai.com")
        print("3. Use VPN if allowed")
        print("4. Run this from home network")
        
        continue_anyway = input("\nTry anyway? (y/n): ")
        if continue_anyway.lower() != 'y':
            return
    
    # Get API key
    try:
        get_api_key()
        print("✅ OpenAI API key configured")
    except Exception as e:
        print(f"❌ API key setup failed: {e}")
        return
    
    # Get PDF path
    try:
        pdf_path = get_pdf_path()
        print(f"✅ PDF file found: {pdf_path}")
    except Exception as e:
        print(f"❌ PDF setup failed: {e}")
        return
    
    # Import libraries after installation
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_openai import OpenAIEmbeddings, ChatOpenAI
        from langchain_community.vectorstores import Chroma
        from langchain.prompts import PromptTemplate
        from langchain.chains import RetrievalQA
        import gradio as gr
        print("✅ Libraries imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Please restart the script to ensure packages are properly installed.")
        return
    
    print("\n🔄 Processing PDF document...")
    
    # Load and process PDF
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        
        print(f"✅ Loaded PDF with {len(documents)} pages")
        print(f"✅ Split into {len(texts)} text chunks")
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return
    
    # Create embeddings and vector store
    try:
        print("🔄 Creating vector embeddings...")
        print("⏳ This may take a few minutes for the first time...")
        
        # Test OpenAI connection first
        print("🔍 Testing OpenAI API connection...")
        embeddings = OpenAIEmbeddings()
        
        print("🔍 Attempting to create test embedding...")
        test_embedding = embeddings.embed_query("test")
        print("✅ OpenAI connection successful")
        
        print("🔍 Creating ChromaDB vector store...")
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings
        )
        print("✅ Vector representations created")
    except Exception as e:
        print(f"❌ Vector store creation failed")
        print(f"📋 Full error details: {type(e).__name__}: {str(e)}")
        
        # Import traceback for detailed error info
        import traceback
        print(f"📋 Full traceback:")
        traceback.print_exc()
        
        print("\n🔍 Troubleshooting:")
        print("1. Check your OpenAI API key is valid")
        print("2. Ensure you have internet connection")
        print("3. Check if you have OpenAI API credits")
        print("4. Try running the script again")
        
        # Check specific error types
        error_str = str(e).lower()
        if "api" in error_str or "auth" in error_str or "key" in error_str:
            print("\n❗ This looks like an API key issue.")
        elif "connection" in error_str or "network" in error_str:
            print("\n❗ This looks like a network connection issue.")
        elif "rate" in error_str or "limit" in error_str:
            print("\n❗ This looks like an API rate limit issue.")
        elif "credit" in error_str or "quota" in error_str:
            print("\n❗ This looks like an API credit/quota issue.")
        
        retry = input("\nWould you like to re-enter your API key? (y/n): ")
        if retry.lower() == 'y':
            try:
                get_api_key()
                print("✅ API key updated. Please restart the script.")
            except:
                pass
        return
    
    # Setup AI model
    try:
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        print("✅ AI model configured")
    except Exception as e:
        print(f"❌ AI model setup failed: {e}")
        return
    
    # Create prompt template and QA chain
    template = """You are a Nestlé HR assistant. Use the following context from Nestlé's HR policy documents to answer the question.

Context: {context}

Question: {question}

Instructions:
- Provide accurate information based on the context
- If information is not available, suggest contacting HR
- Maintain a professional, helpful tone

Answer:"""
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    print("✅ QA system ready")
    
    # Define chat function
    def chat_function(message):
        """Process user message and return response"""
        try:
            response = qa_chain.invoke({"query": message})
            return response["result"]
        except Exception as e:
            return f"I apologize, but I'm having trouble accessing the information. Please contact HR directly. Error: {str(e)}"
    
    # Create Gradio interface
    interface = gr.Interface(
        fn=chat_function,
        inputs=gr.Textbox(placeholder="Ask about HR policies...", label="Your Question", lines=2),
        outputs=gr.Textbox(label="HR Assistant Response", lines=6),
        title="🤖 Nestlé HR Policy Assistant",
        description="Ask questions about Nestlé HR policies and procedures.",
        examples=[
            ["What is Nestlé's approach to performance management?"],
            ["Tell me about career development opportunities"],
            ["What are Nestlé's diversity and inclusion policies?"]
        ],
        allow_flagging="never"
    )
    
    print("✅ Chatbot interface created")
    print("\n🚀 Launching HR Assistant...")
    print("The web interface will open in your browser.")
    print("Press Ctrl+C to stop the application.")
    
    # Launch the interface
    try:
        interface.launch(share=True)
    except KeyboardInterrupt:
        print("\n👋 HR Assistant stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Launch failed: {e}")

if __name__ == "__main__":
    main()