#!/bin/bash
cd /Users/adikol/Desktop/aria-agent
export PYTHONPATH="/Users/adikol/Desktop/aria-agent"
source venv/bin/activate
exec python run_mcp_server.py
