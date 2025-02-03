# PUMA: A Project Understanding & Modification Accelerator
A Retrieval-Augmented Generation (RAG) system designed to process and query GitHub repositories or project folders.
The system is built to support user's modifications without a clear understanding of the codebase/project structure. This system accelerates your coding process when ...
- You want to find out how a novel operation is defined in a deep learning method codebase. But you don't know where to look.
- You want clone a web template and modify it to your needs. But you are tired of reading through the structure.
- You want to understand the function of different modules in a project. But you don't have the time read every file.

This system will build a RAG model on your local machine to help you query your codebase and get the information you need. It basically will return the position of the code you are looking for, and provide a brief explanation of how you can modify it, or how it is used in the project.

## Features
- **Data Processing**: Splits files into meaningful **chunks** and records project structure.
- **Embedding**: Converts chunks into vector representations and stores them in a **vector database**.
- **Query Engine**: Uses LangChain to process and respond to queries.
- **Interface**: Supports **terminal-based queries** with a plan for future *FastAPI web support*.
- **Testing & Validation**: Automated pipeline to verify correctness.

## Project Structure
```
📦 local-rag-system
│── 📂 data_processing
│   │── chunker.py            # Splits files into chunks
│   │── structure_recorder.py  # Records file structure
│   │── utils.py               # Common utilities for preprocessing
│
│── 📂 embedding
│   │── embedder.py            # Calls embedding API
│   │── vector_store.py        # Handles vector database interactions
│
│── 📂 query_engine
│   │── retriever.py           # Retrieves relevant chunks from the vector DB
│   │── generator.py           # Uses LangChain to generate responses
│   │── query_handler.py       # Main interface for query execution
│
│── 📂 interface
│   │── terminal_ui.py         # Handles terminal-based user interaction
│   │── web_app.py             # FastAPI web server (for future expansion)
│
│── 📂 tests
│   │── test_pipeline.py       # Runs automated tests for RAG performance
│   │── evaluation.py          # Evaluates system correctness using ground truth
│
│── 📂 configs
│   │── settings.yaml          # Config file for paths, API keys, etc.
│
│── 📂 storage
│   │── vectors/               # Directory to store vector database files
│   │── cache/                 # Temporary cache storage (e.g., processed files)
│
│── 📂 assets
│   │── images/                # Images for documentation and README
│   │── diagrams/              # Flowcharts or system architecture illustrations
│
│── 📂 docs
│   │── API.md                 # API documentation for embedding and querying
│   │── INSTALL.md             # Installation and setup guide
│   │── USAGE.md               # Instructions for using the system
│
│── 📂 scripts
│   │── run_terminal.py        # Entry point for terminal-based queries
│   │── run_web.py             # Entry point for FastAPI server (future)
│   │── data_cleanup.py        # Script for cleaning up old vector or cache data
│
│── requirements.txt           # Python dependencies
│── README.md                  # Project documentation with usage instructions
│── .gitignore                 # Ignore unnecessary files
```

## References GitHub Repositories
- [🏯 Eunomia 🏯: 🔐 Query & analyze your code locally using a GPT model🔐
](https://github.com/Ngz91/Eunomia)

## Installation (Not yet available)
Refer to [INSTALL.md](docs/INSTALL.md).

## Usage (Not yet available)
Refer to [USAGE.md](docs/USAGE.md).

