{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "# Nestlé HR Policy Assistant - Course End Project\n",
        "\n",
        "**Project**: Crafting an AI-Powered HR Assistant for Nestlé's HR Policy Documents\n",
        "\n",
        "This notebook demonstrates the complete workflow for creating a conversational chatbot that responds to user inquiries using PDF document information."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 1: Import Essential Tools and Set Up OpenAI's API Environment"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# RESTART KERNEL FIRST, then run this cell\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "# Uninstall conflicting packages\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'pydantic', 'chromadb', 'gradio', 'fastapi'])\n",
        "\n",
        "# Install working versions in correct order\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'pydantic==1.10.12'])\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'chromadb==0.4.15'])\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'gradio==3.50.2'])\n",
        "subprocess.run([sys.executable, '-m', 'pip', 'install', 'langchain', 'langchain-community', 'langchain-openai', 'pypdf', 'openai', 'pysqlite3-binary'])\n",
        "\n",
        "print(\"✅ Packages installed. RESTART KERNEL NOW before proceeding.\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Fix SQLite BEFORE any other imports\n",
        "import sys\n",
        "try:\n",
        "    import pysqlite3\n",
        "    sys.modules['sqlite3'] = pysqlite3\n",
        "except ImportError:\n",
        "    pass\n",
        "\n",
        "# Import essential libraries\n",
        "import os\n",
        "from langchain_community.document_loaders import PyPDFLoader\n",
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "from langchain_openai import OpenAIEmbeddings, ChatOpenAI\n",
        "from langchain_community.vectorstores import Chroma\n",
        "from langchain.prompts import PromptTemplate\n",
        "from langchain.chains import RetrievalQA\n",
        "\n",
        "# Set up OpenAI API key from environment variable\n",
        "if not os.getenv('OPENAI_API_KEY'):\n",
        "    raise ValueError('Please set OPENAI_API_KEY environment variable')\n",
        "\n",
        "print(\"✅ Essential tools imported and OpenAI API environment set up\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 2: Load Nestlé's HR Policy Using PyPDFLoader and Split for Processing"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Load PDF document\n",
        "pdf_path = \"1728286846_the_nestle_hr_policy_pdf_2012.pdf\"\n",
        "loader = PyPDFLoader(pdf_path)\n",
        "documents = loader.load()\n",
        "\n",
        "# Split documents into chunks\n",
        "text_splitter = RecursiveCharacterTextSplitter(\n",
        "    chunk_size=1000,\n",
        "    chunk_overlap=200\n",
        ")\n",
        "texts = text_splitter.split_documents(documents)\n",
        "\n",
        "print(f\"✅ Loaded PDF with {len(documents)} pages\")\n",
        "print(f\"✅ Split into {len(texts)} text chunks\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 3: Create Vector Representations Using ChromaDB and OpenAI's Embeddings"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Create embeddings\n",
        "embeddings = OpenAIEmbeddings()\n",
        "\n",
        "# Create vector store using ChromaDB\n",
        "vectorstore = Chroma.from_documents(\n",
        "    documents=texts,\n",
        "    embedding=embeddings\n",
        ")\n",
        "\n",
        "print(\"✅ Vector representations created using ChromaDB and OpenAI embeddings\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 4: Build Question-Answering System Using GPT-3.5 Turbo"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Initialize GPT-3.5 Turbo model\n",
        "llm = ChatOpenAI(\n",
        "    model_name=\"gpt-3.5-turbo\",\n",
        "    temperature=0.3\n",
        ")\n",
        "\n",
        "# Create retriever\n",
        "retriever = vectorstore.as_retriever(search_kwargs={\"k\": 3})\n",
        "\n",
        "print(\"✅ Question-answering system built with GPT-3.5 Turbo\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 5: Create Prompt Template for Chatbot Understanding"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Create prompt template\n",
        "template = \"\"\"You are a Nestlé HR assistant. Use the following context from Nestlé's HR policy documents to answer the question.\n",
        "\n",
        "Context: {context}\n",
        "\n",
        "Question: {question}\n",
        "\n",
        "Instructions:\n",
        "- Provide accurate information based on the context\n",
        "- If information is not available, suggest contacting HR\n",
        "- Maintain a professional, helpful tone\n",
        "\n",
        "Answer:\"\"\"\n",
        "\n",
        "prompt = PromptTemplate(\n",
        "    template=template,\n",
        "    input_variables=[\"context\", \"question\"]\n",
        ")\n",
        "\n",
        "# Create QA chain\n",
        "qa_chain = RetrievalQA.from_chain_type(\n",
        "    llm=llm,\n",
        "    chain_type=\"stuff\",\n",
        "    retriever=retriever,\n",
        "    chain_type_kwargs={\"prompt\": prompt}\n",
        ")\n",
        "\n",
        "print(\"✅ Prompt template created for chatbot understanding\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 6: Build User-Friendly Chatbot Interface with Gradio"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "def chat_function(message):\n",
        "    \"\"\"Process user message and return response\"\"\"\n",
        "    try:\n",
        "        response = qa_chain.invoke({\"query\": message})\n",
        "        return response[\"result\"]\n",
        "    except Exception as e:\n",
        "        return f\"I apologize, but I'm having trouble accessing the information. Please contact HR directly. Error: {str(e)}\"\n",
        "\n",
        "# Create Gradio interface\n",
        "import gradio as gr\n",
        "\n",
        "interface = gr.Interface(\n",
        "    fn=chat_function,\n",
        "    inputs=gr.Textbox(placeholder=\"Ask about HR policies...\", label=\"Your Question\", lines=2),\n",
        "    outputs=gr.Textbox(label=\"HR Assistant Response\", lines=6),\n",
        "    title=\"🤖 Nestlé HR Policy Assistant\",\n",
        "    description=\"Ask questions about Nestlé HR policies and procedures.\",\n",
        "    examples=[\n",
        "        [\"What is Nestlé's approach to performance management?\"],\n",
        "        [\"Tell me about career development opportunities\"],\n",
        "        [\"What are Nestlé's diversity and inclusion policies?\"]\n",
        "    ],\n",
        "    allow_flagging=\"never\"\n",
        ")\n",
        "\n",
        "print(\"✅ User-friendly chatbot interface created with Gradio\")"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Step 7: Launch the Chatbot"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Launch the Gradio interface\n",
        "interface.launch(share=True)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": [
        "## Testing the System"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "# Test the QA system with example questions\n",
        "test_questions = [\n",
        "    \"What is Nestlé's approach to performance management?\",\n",
        "    \"Tell me about career development opportunities\",\n",
        "    \"What are Nestlé's diversity and inclusion policies?\"\n",
        "]\n",
        "\n",
        "print(\"Testing the HR Assistant:\")\n",
        "print(\"=\" * 80)\n",
        "\n",
        "for i, question in enumerate(test_questions, 1):\n",
        "    print(f\"\\nTest {i}: {question}\")\n",
        "    answer = chat_function(question)\n",
        "    print(f\"Answer: {answer}\")\n",
        "    print(\"-\" * 80)"
      ]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}