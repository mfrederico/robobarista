#!/bin/bash
# launch_voice_assistant.sh
# Script to launch all components of the voice assistant drive-through system

# Configuration
SPEECH_RECOGNITION="stt/enhanced_voice_redis.py"
LLM_SERVICE="llm/simplified_ollama_service.py"
TTS_SERVICE="tts/simplified_tts_service.py"
MENU_FILE="restaurant_menu.json"
OLLAMA_MODEL="llama3.2:3b"  # great!
VOSK_MODEL="vosk-model-en-us-0.22"
VOICE_EMBEDDING="tts/speaker_embeddings/cmu_us_rms_arctic-wav-arctic_b0353.npy"

# Create necessary directories
mkdir -p stt llm tts
mkdir -p tts/speaker_embeddings

# Check if Redis is running
echo "Checking Redis server..."
if ! redis-cli ping > /dev/null 2>&1; then
    echo "Redis server not running. Starting Redis..."
    sudo systemctl start redis-server
    sleep 2
    
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "Failed to start Redis server. Please check your Redis installation."
        exit 1
    fi
fi
echo "Redis server is running."

# Check if Ollama is running
echo "Checking Ollama server..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "Ollama server not running. Please start Ollama with 'ollama serve' in another terminal."
    exit 1
fi
echo "Ollama server is running."

# Check if the specified model is available
echo "Checking if model '$OLLAMA_MODEL' is available..."
MODEL_LIST=$(curl -s http://localhost:11434/api/tags)
if ! echo $MODEL_LIST | grep -q "\"$OLLAMA_MODEL\""; then
    echo "Model '$OLLAMA_MODEL' not found. Do you want to pull it? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Pulling model '$OLLAMA_MODEL'..."
        ollama pull $OLLAMA_MODEL
    else
        echo "Please modify this script to use an available model."
        echo "Available models:"
        echo $MODEL_LIST | grep -o '"name":"[^"]*"' | cut -d'"' -f4
        exit 1
    fi
fi
echo "Model '$OLLAMA_MODEL' is available."

# Check for required Python modules
echo "Checking required Python modules..."
MISSING_MODULES=0
for module in redis sounddevice numpy vosk requests torch transformers; do
    if ! python -c "import $module" &> /dev/null; then
        echo "Missing Python module: $module"
        MISSING_MODULES=1
    fi
done

if [ $MISSING_MODULES -eq 1 ]; then
    echo "Some required Python modules are missing. Do you want to install them? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Installing missing modules..."
        pip install redis sounddevice numpy vosk requests torch transformers
    else
        echo "Please install the missing modules before continuing."
        exit 1
    fi
fi

# Check for VOSK model
if [ ! -d "$VOSK_MODEL" ]; then
    echo "VOSK model not found. Please download it from:"
    echo "https://alphacephei.com/vosk/models"
    echo "and extract to $VOSK_MODEL"
    echo "Continue without speech recognition? (y/n)"
    read -r response
    if ! [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        exit 1
    fi
fi

# Check if script files exist
for script in "$SPEECH_RECOGNITION" "$LLM_SERVICE" "$TTS_SERVICE"; do
    if [ ! -f "$script" ]; then
        # Try to find the script in the current directory
        base_script=$(basename "$script")
        if [ -f "$base_script" ]; then
            # Create symbolic link
            ln -sf "$(pwd)/$base_script" "$script"
            echo "Created symbolic link for $script"
        else
            echo "Error: Script $script not found."
            echo "Please place the script files in the correct locations or adjust the paths in this script."
            exit 1
        fi
    fi
done

# Check if menu file exists
if [ ! -f "$MENU_FILE" ]; then
    echo "Error: Menu file $MENU_FILE not found."
    echo "Continue without menu? (y/n)"
    read -r response
    if ! [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        exit 1
    fi
fi

# Function to start a component in a new terminal
start_component() {
    local script=$1
    local title=$2
    local args=$3
    
    # Check if script exists
    if [ ! -f "$script" ]; then
        echo "Error: Script $script not found."
        return 1
    fi
    
    # Start in a new terminal
    if command -v gnome-terminal &> /dev/null; then
        # GNOME Terminal (Ubuntu)
        gnome-terminal --title="$title" -- bash -c "python $script $args; read -p 'Press Enter to close...'"
    elif command -v konsole &> /dev/null; then
        # KDE Konsole
        konsole --new-tab --title "$title" -e bash -c "python $script $args; read -p 'Press Enter to close...'"
    elif command -v xterm &> /dev/null; then
        # xterm
        xterm -title "$title" -e "python $script $args; read -p 'Press Enter to close...'"
    else
        # Fallback - start in background
        echo "No supported terminal found. Starting $title in background."
        python "$script" $args &
    fi
    
    # Wait a moment for the component to start
    sleep 2
}

# Ask for launch mode
echo "How would you like to launch the system?"
echo "1. Full system with speech recognition and TTS"
echo "2. Text-only mode (no speech, terminal interface)"
echo "3. Select components manually"
read -r launch_mode

case "$launch_mode" in
    1)
        # Start all components
        echo "Starting Voice Assistant components..."
        
        # Start LLM service first (this needs to be ready to receive input)
        echo "Starting LLM Processing Service..."
        start_component "$LLM_SERVICE" "LLM Service" "--ollama-model $OLLAMA_MODEL --menu-file $MENU_FILE"
        
        # Start TTS service
        echo "Starting Text-to-Speech Service..."
        start_component "$TTS_SERVICE" "TTS Service" "--voice-embedding $VOICE_EMBEDDING"
        
        # Start speech recognition last
        echo "Starting Speech Recognition Service..."
        start_component "$SPEECH_RECOGNITION" "Speech Recognition" ""
        ;;
    2)
        # Start text-only mode
        echo "Starting Voice Assistant in text-only mode..."
        
        # Start LLM service first
        echo "Starting LLM Processing Service..."
        start_component "$LLM_SERVICE" "LLM Service" "--ollama-model $OLLAMA_MODEL --menu-file $MENU_FILE"
        
        # Start test client
        echo "Starting Test Client..."
        start_component "$TEST_CLIENT" "Test Client" ""
        ;;
    3)
        # Select components manually
        echo "Select components to start:"
        
        echo "Start LLM Service? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            start_component "$LLM_SERVICE" "LLM Service" "--ollama-model $OLLAMA_MODEL --menu-file $MENU_FILE"
        fi
        
        echo "Start TTS Service? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            start_component "$TTS_SERVICE" "TTS Service" "--voice-embedding $VOICE_EMBEDDING"
        fi
        
        echo "Start Speech Recognition Service? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            start_component "$SPEECH_RECOGNITION" "Speech Recognition" ""
        fi
        
        echo "Start Test Client? (y/n)"
        read -r response
        if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
            start_component "$TEST_CLIENT" "Test Client" ""
        fi
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac

echo "All components started!"
echo "The Voice Assistant is now running."
echo "Speak into your microphone to interact with it (or use the test client if in text mode)."
echo "To stop all components, close their terminal windows or press Ctrl+C in each."

# Keep the script running
echo "Press Ctrl+C to exit this script (components will continue running)"
while true; do
    sleep 1
done

