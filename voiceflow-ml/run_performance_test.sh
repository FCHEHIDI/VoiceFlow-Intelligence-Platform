#!/bin/bash
# Quick deployment and performance test script

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   VoiceFlow Inference Server - Deployment & Performance Test  ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Activate virtual environment
echo "🔧 Activating Python environment..."
source /c/Users/Fares/VoiceFlow-Intelligence-Platform/.venv/Scripts/activate

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q fastapi uvicorn aiohttp

# Start inference server in background
echo "🚀 Starting inference server on port 3000..."
cd /c/Users/Fares/VoiceFlow-Intelligence-Platform/VoiceFlow-Intelligence-Platform/voiceflow-ml
python inference_server.py > server.log 2>&1 &
SERVER_PID=$!

echo "   Server PID: $SERVER_PID"
echo "   Waiting for server to start..."
sleep 5

# Check if server is running
if curl -s http://localhost:3000/health > /dev/null 2>&1; then
    echo "✅ Server is running!"
    echo ""
    
    # Run load test
    echo "🧪 Running load test..."
    python load_test.py --url http://localhost:3000 --requests 200 --concurrency 20
    
    echo ""
    echo "📊 Server metrics available at: http://localhost:3000/metrics"
    echo ""
    echo "Press Enter to stop server and exit..."
    read
    
    # Stop server
    echo "🛑 Stopping server..."
    kill $SERVER_PID 2>/dev/null
    echo "✅ Server stopped"
else
    echo "❌ Server failed to start. Check server.log for details"
    cat server.log
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
