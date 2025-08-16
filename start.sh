#!/bin/bash
set -e


python nltk_downloader.py

streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
