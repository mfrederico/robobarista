# simplified_tts_service.py
# Simplified TTS service focused on reliability

import json
import argparse
import threading
import time
import os
import queue
import torch
import numpy as np
import sounddevice as sd
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import redis
import re

# === Configuration Parameters ===
class Config:
    def __init__(self):
        # Parse command line arguments
        parser = argparse.ArgumentParser(description="Simplified Text-to-Speech Service")
        parser.add_argument("--redis-host", type=str, default="localhost",
                          help="Redis server hostname")
        parser.add_argument("--redis-port", type=int, default=6379,
                          help="Redis server port")
        parser.add_argument("--redis-channel-in", type=str, default="llm_response_output",
                          help="Redis channel to receive LLM responses")
        parser.add_argument("--redis-channel-out", type=str, default="speech_recognition_control",
                          help="Redis channel to send control messages")
        parser.add_argument("--voice-embedding", type=str, default="tts/speaker_embeddings/cmu_us_rms_arctic-wav-arctic_b0353.npy",
                          help="Path to voice embedding file")
        parser.add_argument("--output-device", type=int, default=None,
                          help="Audio output device ID")
        parser.add_argument("--debug", action="store_true",
                          help="Enable debug output")
        
        args = parser.parse_args()
        
        # Apply arguments
        self.redis_host = args.redis_host
        self.redis_port = args.redis_port
        self.redis_channel_in = args.redis_channel_in
        self.redis_channel_out = args.redis_channel_out
        self.voice_embedding_path = args.voice_embedding
        self.output_device = args.output_device
        self.debug = args.debug
        
        # Additional TTS parameters
        self.sample_rate = 16000
        self.max_text_length = 150  # Maximum length of text to process at once
        self.pause_between_phrases = 0.3  # Seconds to pause between phrases
        self.pause_duration = 1.0  # Duration to pause speech recognition while speaking
        self.resume_delay = 0.5  # Delay after speaking before resuming listening

        # Add a flag to track if we're in "thinking" mode
        self.in_thinking_mode = False

# === List and Choose Audio Device ===
def list_output_devices():
    """List all available audio output devices"""
    print("\nAvailable Output Devices:")
    print("-" * 50)
    devices = sd.query_devices()
    
    for i, dev in enumerate(devices):
        if dev.get('max_output_channels', 0) > 0:
            print(f"[{i}] {dev['name']} (outputs: {dev['max_output_channels']})")
    
    print("-" * 50)

def choose_output_device():
    """Let user select an output device"""
    list_output_devices()
    
    while True:
        try:
            choice = input("Select output device number: ")
            device_id = int(choice)
            devices = sd.query_devices()
            
            if 0 <= device_id < len(devices) and devices[device_id].get('max_output_channels', 0) > 0:
                return device_id
            else:
                print("Invalid device selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

# === Simplified TTS Service ===
class SimplifiedTTSService:
    def __init__(self, config):
        self.config = config
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host=config.redis_host,
            port=config.redis_port
        )
        self.pubsub = self.redis_client.pubsub()
        self.pubsub.subscribe(config.redis_channel_in)
        
        # Check Redis connection
        try:
            self.redis_client.ping()
            print(f"Connected to Redis at {config.redis_host}:{config.redis_port}")
        except redis.ConnectionError:
            raise RuntimeError(f"Failed to connect to Redis at {config.redis_host}:{config.redis_port}")
        
        # Load TTS model
        self._load_tts_model()
        
        # Select output device if not specified
        if self.config.output_device is None:
            self.config.output_device = choose_output_device()
        
        # Status flags
        self.stop_event = threading.Event()
        self.is_speaking = threading.Event()
        
        # Stream tracking
        self.current_stream_id = None
        self.phrase_queue = queue.Queue()
        
        # Initialize the speech playback thread
        self.speech_thread = threading.Thread(target=self._speech_playback_thread)
        self.speech_thread.daemon = True
        
        # Safety timer for resuming speech recognition
        self.resume_timer = None
    
    def _load_tts_model(self):
        """Load the SpeechT5 TTS model and voice embedding"""
        print("Loading text-to-speech model...")
        
        try:
            # Load model components
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
            
            # Check and load voice embedding
            self._load_voice_embedding()
            
            print("TTS model loaded successfully!")
        except Exception as e:
            raise RuntimeError(f"Failed to load TTS model: {e}")
    
    def _load_voice_embedding(self):
        """Load or create a voice embedding file"""
        if not os.path.exists(self.config.voice_embedding_path):
            print(f"Voice embedding file not found: {self.config.voice_embedding_path}")
            print("Creating a default voice embedding...")
            
            # Create a default embedding
            default_embedding = np.ones((1, 512), dtype=np.float32) * 0.01
            
            # Ensure directory exists
            embedding_dir = os.path.dirname(self.config.voice_embedding_path)
            if embedding_dir and not os.path.exists(embedding_dir):
                os.makedirs(embedding_dir)
                
            np.save(self.config.voice_embedding_path, default_embedding)
        
        # Load the embedding
        embedding = np.load(self.config.voice_embedding_path)
        self.speaker_embedding = torch.tensor(embedding).float()
        
        # Process embedding dimensions
        if self.speaker_embedding.ndim > 2:
            self.speaker_embedding = self.speaker_embedding.squeeze(0)
        if self.speaker_embedding.ndim == 1:
            self.speaker_embedding = self.speaker_embedding.unsqueeze(0)
    
    def _text_to_speech(self, text):
        """Convert text to speech audio"""
        if not text.strip():
            return None
        
        # If text is too long, split it
        if len(text) > self.config.max_text_length:
            # Find a good place to split (at punctuation or space)
            split_point = self.config.max_text_length
            while split_point > 0 and text[split_point] not in ".!?, ":
                split_point -= 1
            
            if split_point == 0:
                split_point = self.config.max_text_length
            
            # Process the first chunk
            first_chunk = text[:split_point].strip()
            audio = self._text_to_speech(first_chunk)
            
            # Don't process more if we're stopping
            if self.stop_event.is_set():
                return audio
            
            # Process the remaining text recursively
            remaining = text[split_point:].strip()
            if remaining:
                time.sleep(0.1)  # Small pause between chunks
                remaining_audio = self._text_to_speech(remaining)
                
                # Concatenate the audio
                if audio is not None and remaining_audio is not None:
                    audio = np.concatenate([audio, remaining_audio])
            
            return audio
        
        try:
            # Process text with the model
            inputs = self.processor(text=text, return_tensors="pt")
            
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embeddings=self.speaker_embedding, 
                    vocoder=self.vocoder
                )
            
            # Convert to numpy array
            audio = speech.numpy()
            
            # Normalize audio
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio * 0.9 / max_val
            
            return audio
            
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
    
    def _play_audio(self, audio):
        """Play audio through the selected output device"""
        if audio is None:
            return
        
        try:
            # Set the speaking flag
            self.is_speaking.set()
            
            # Send pause command to speech recognition
            self._send_pause_command()
            if self.config.debug:
                print("[DEBUG] Sent pause command")
            
            # Add a small delay to ensure the pause command is received
            time.sleep(0.2)
            
            # Speed up the audio by reducing its length
            # This is a simple time-domain approach that increases the pitch
            # A better approach would use a vocoder with time stretching
            speed_factor = 1.3  # Speak 30% faster
            audio_len = len(audio)
            new_len = int(audio_len / speed_factor)
            
            # Resample the audio (simple method - using linear interpolation)
            indices = np.linspace(0, audio_len - 1, new_len)
            indices = indices.astype(np.int32)
            faster_audio = audio[indices]
            
            # Play the faster audio
            sd.play(faster_audio, self.config.sample_rate, device=self.config.output_device)
            sd.wait()
            
            # Add a delay after audio finishes
            time.sleep(self.config.resume_delay)
            
            # Clear the speaking flag
            self.is_speaking.clear()
            
            # Send resume command with a timer for safety
            self._schedule_resume_command()
            
        except Exception as e:
            print(f"Error playing audio: {e}")
            self.is_speaking.clear()
            
            # Try to send resume command even if there was an error
            self._schedule_resume_command()
    
    def _schedule_resume_command(self):
        """Schedule the resume command with a timer for safety"""
        # Cancel any existing timer
        if self.resume_timer is not None:
            self.resume_timer.cancel()
        
        # Send resume command immediately
        self._send_resume_command()
        if self.config.debug:
            print("[DEBUG] Sent resume command")
        
        # Also schedule a backup resume command in case the first one is missed
        self.resume_timer = threading.Timer(1.0, self._send_resume_command)
        self.resume_timer.daemon = True
        self.resume_timer.start()
    
    def _send_pause_command(self):
        """Send pause command to speech recognition service"""
        try:
            message = {
                "type": "control",
                "action": "pause",
                "timestamp": time.time()
            }
            
            self.redis_client.publish(
                self.config.redis_channel_out,
                json.dumps(message)
            )
            
            if self.config.debug:
                print("[Sent pause command to speech recognition]")
        except Exception as e:
            print(f"Error sending pause command: {e}")
    
    def _send_resume_command(self):
        """Send resume command to speech recognition service"""
        try:
            message = {
                "type": "control",
                "action": "resume",
                "timestamp": time.time()
            }
            
            self.redis_client.publish(
                self.config.redis_channel_out,
                json.dumps(message)
            )
            
            if self.config.debug:
                print("[Sent resume command to speech recognition]")
        except Exception as e:
            print(f"Error sending resume command: {e}")
    
    def _speech_playback_thread(self):
        """Background thread to process and play speech from the queue"""
        while not self.stop_event.is_set():
            try:
                # Get the next phrase from the queue
                try:
                    phrase = self.phrase_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if phrase == "__STOP__":
                    break
                
                # Convert to speech and play
                print(f"[Speaking] {phrase}")
                audio = self._text_to_speech(phrase)
                self._play_audio(audio)
                
                # Add a small pause between phrases
                time.sleep(self.config.pause_between_phrases)
                
            except Exception as e:
                print(f"Error in speech playback thread: {e}")
    
    def _handle_message(self, message):
        """Process messages from Redis"""
        try:
            # Parse the message
            data = json.loads(message["data"])
            
            # Check message type
            msg_type = data.get("type", "unknown")
            
            if msg_type == "stream_start":
                # Start of a new stream
                stream_id = data.get("stream_id", 0)
                self.current_stream_id = stream_id
                
                # Clear any pending phrases on new stream start
                with self.phrase_queue.mutex:
                    self.phrase_queue.queue.clear()
                
                # Reset thinking mode flag at the start of a new stream
                self.in_thinking_mode = False
                
                if self.config.debug:
                    print(f"\n[Stream {stream_id}] Started")
            
            elif msg_type == "stream_phrase":
                # Process a phrase
                stream_id = data.get("stream_id", 0)
                text = data.get("text", "").strip()
                
                if not text:
                    return
                    
                # Print the text to stdout
                print(f"[Full Text] {text}")
                
                # Check for think tags in this line
                if "<think>" in text:
                    self.in_thinking_mode = True
                    print("[DEBUG] Entering thinking mode")
                
                # Only send text to speech if we're not in thinking mode
                if not self.in_thinking_mode:
                    # Process text for speech (without the tags if they exist)
                    speech_text = text.replace("<think>", "").replace("</think>", "").strip()
                    
                    if speech_text:
                        self.phrase_queue.put(speech_text)
                        if self.config.debug:
                            print(f"[Queued for speech] {speech_text}")
                
                # Check if thinking mode ends in this line
                if "</think>" in text:
                    self.in_thinking_mode = False
                    print("[DEBUG] Exiting thinking mode")
            
            elif msg_type == "stream_end":
                # End of stream
                stream_id = data.get("stream_id", 0)
                
                if stream_id == self.current_stream_id:
                    self.current_stream_id = None
                    # Reset thinking mode at end of stream
                    self.in_thinking_mode = False
                    
                    if self.config.debug:
                        print(f"\n[Stream {stream_id}] Ended")
            
            elif msg_type == "stop_speech":
                # Request to stop current speech immediately
                with self.phrase_queue.mutex:
                    self.phrase_queue.queue.clear()
                
                if self.config.debug:
                    print("\n[Received stop_speech command]")
            
            elif msg_type == "error":
                # Handle error message
                error = data.get("error", "Unknown error")
                print(f"\n[Error] {error}")
                
        except json.JSONDecodeError:
            print(f"Invalid message format: {message['data']}")
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def start(self):
        """Start the TTS service"""
        print(f"Starting Simplified TTS service (output device: {self.config.output_device})")
        print(f"Listening on Redis channel: {self.config.redis_channel_in}")
        print("Press Ctrl+C to stop")
        
        # Reset state
        self.stop_event.clear()
        self.is_speaking.clear()
        self.current_stream_id = None
        
        # Start the speech playback thread
        self.speech_thread.start()
        
        try:
            while not self.stop_event.is_set():
                # Get and process messages from Redis
                message = self.pubsub.get_message(timeout=0.1)
                if message and message["type"] == "message":
                    self._handle_message(message)
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
        except KeyboardInterrupt:
            print("\nStopping service (keyboard interrupt)")
        except Exception as e:
            print(f"\nError in service main loop: {e}")
        finally:
            # Stop any pending timers
            if self.resume_timer is not None:
                self.resume_timer.cancel()
            
            # Signal the threads to stop
            self.stop_event.set()
            self.phrase_queue.put("__STOP__")
            
            # Wait for thread to finish
            if self.speech_thread.is_alive():
                self.speech_thread.join(timeout=2)
            
            # Make sure speech recognition is resumed
            self._send_resume_command()
            
            print("Service stopped")
    
    def stop(self):
        """Stop the TTS service"""
        # Stop any pending timers
        if self.resume_timer is not None:
            self.resume_timer.cancel()
        
        self.stop_event.set()
        self.phrase_queue.put("__STOP__")
        print("Stopping TTS service...")

# === Main Function ===
def main():
    try:
        # Load configuration
        config = Config()
        
        # Create and start TTS service
        service = SimplifiedTTSService(config)
        service.start()
        
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

