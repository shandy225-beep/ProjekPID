#!/bin/bash

# Script untuk restart Streamlit dashboard dengan clear cache

echo "🔄 Stopping existing Streamlit processes..."
pkill -f streamlit

echo "⏳ Waiting 2 seconds..."
sleep 2

echo "🗑️ Clearing Streamlit cache directory..."
rm -rf ~/.streamlit/cache/

echo "🚀 Starting Streamlit dashboard..."
streamlit run dashboard/app.py --server.port 8502

