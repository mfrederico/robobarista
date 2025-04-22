#!/usr/bin/env python3
# test_client.py
# Test client for the voice assistant system

import json
import redis
import time
import threading
import argparse
from datetime import datetime

class VoiceAssistantClient:
    def __init__(self, redis_host="localhost", redis_port=6379):
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=redis_host,
            port=redis_port
        )
        
        # Test connection
        try:
            self.redis_client.ping()
            print(f"Connected to Redis at {redis_host}:{redis_port}")
        except redis.ConnectionError:
            raise RuntimeError(f"Failed to connect to Redis at {redis_host}:{redis_port}")
        
        # Subscribe to LLM responses
        self.pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
        self.pubsub.subscribe("llm_response_output")
        
        # Set up response handling
        self.response_thread = None
        self.stop_event = threading.Event()
        self.current_stream = None
        self.stream_responses = {}
    
    def start_response_listener(self):
        """Start listening for responses in a background thread"""
        self.stop_event.clear()
        
        self.response_thread = threading.Thread(target=self._response_listener)
        self.response_thread.daemon = True
        self.response_thread.start()
    
    def _response_listener(self):
        """Background thread to listen for responses"""
        print("Response listener started")
        
        while not self.stop_event.is_set():
            try:
                # Get message from Redis
                message = self.pubsub.get_message(timeout=0.1)
                if message and message["type"] == "message":
                    self._handle_message(message)
                
                time.sleep(0.01)  # Small sleep to prevent CPU spinning
                
            except Exception as e:
                print(f"Error in response listener: {e}")
        
        print("Response listener stopped")
    
    def _handle_message(self, message):
        """Process messages from the LLM service"""
        try:
            # Parse message data
            data = json.loads(message["data"])
            
            # Process based on message type
            msg_type = data.get("type", "unknown")
            
            if msg_type == "stream_start":
                # New stream starting
                stream_id = data.get("stream_id", 0)
                self.current_stream = stream_id
                self.stream_responses[stream_id] = {
                    "phrases": [],
                    "full_text": "",
                    "start_time": time.time(),
                    "user_input": data.get("user_input", "")
                }
                
                print(f"\n[Stream {stream_id}] Started - Input: '{data.get('user_input', '')}'")
                print("-" * 50)
            
            elif msg_type == "stream_phrase":
                # Process a phrase from the stream
                stream_id = data.get("stream_id", 0)
                text = data.get("text", "")
                
                if stream_id in self.stream_responses:
                    self.stream_responses[stream_id]["phrases"].append(text)
                    print(f"AI: {text}")
            
            elif msg_type == "stream_end":
                # Stream has ended
                stream_id = data.get("stream_id", 0)
                full_text = data.get("full_text", "")
                
                if stream_id in self.stream_responses:
                    self.stream_responses[stream_id]["full_text"] = full_text
                    self.stream_responses[stream_id]["end_time"] = time.time()
                    
                    # Calculate response time
                    start_time = self.stream_responses[stream_id]["start_time"]
                    duration = self.stream_responses[stream_id]["end_time"] - start_time
                    
                    print("-" * 50)
                    print(f"[Stream {stream_id}] Ended - Response time: {duration:.2f}s")
                    print("Full response:")
                    print(full_text)
                    print("-" * 50)
            
            elif msg_type == "stream_interrupted":
                # Stream was interrupted
                stream_id = data.get("stream_id", 0)
                new_stream_id = data.get("new_stream_id", 0)
                
                print(f"\n[Stream {stream_id}] Interrupted by stream {new_stream_id}")
            
            elif msg_type == "error":
                # Error message
                error = data.get("error", "Unknown error")
                user_input = data.get("user_input", "")
                
                print(f"\n[Error] {error}")
                if user_input:
                    print(f"Input: '{user_input}'")
            
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON message: {message['data']}")
        except Exception as e:
            print(f"Error handling message: {e}")
    
    def send_text(self, text):
        """Send text input to the LLM service"""
        # Create message
        message = {
            "type": "speech_recognition",
            "text": text,
            "timestamp": time.time()
        }
        
        # Publish to Redis
        self.redis_client.publish(
            "speech_recognition_output",
            json.dumps(message)
        )
        
        print(f"You: {text}")
    
    def send_control(self, action):
        """Send a control command"""
        # Create message
        message = {
            "type": "control",
            "action": action,
            "timestamp": time.time()
        }
        
        # Publish to Redis
        self.redis_client.publish(
            "speech_recognition_output",
            json.dumps(message)
        )
        
        print(f"[Control] Sent '{action}' command")
    
    def stop(self):
        """Stop the client"""
        self.stop_event.set()
        
        if self.response_thread and self.response_thread.is_alive():
            self.response_thread.join(timeout=2)

def main():
    parser = argparse.ArgumentParser(description="Voice Assistant Test Client")
    parser.add_argument("--redis-host", type=str, default="localhost",
                      help="Redis server hostname")
    parser.add_argument("--redis-port", type=int, default=6379,
                      help="Redis server port")
    args = parser.parse_args()
    
    print("=" * 50)
    print("Voice Assistant Test Client")
    print("=" * 50)
    
    try:
        # Create client
        client = VoiceAssistantClient(
            redis_host=args.redis_host,
            redis_port=args.redis_port
        )
        
        # Start response listener
        client.start_response_listener()
        
        print("\nReady for input. Type your message and press Enter.")
        print("Special commands:")
        print("  :quit      - Exit the client")
        print("  :clear     - Clear conversation history")
        print("  :interrupt - Interrupt current response")
        print("  :reset     - Reset the current order")
        print("=" * 50)
        
        # Main input loop
        while True:
            try:
                # Get input from user
                user_input = input("\nYou: ")
                
                if not user_input:
                    continue
                
                # Check for special commands
                if user_input.lower() == ":quit":
                    break
                
                elif user_input.lower() == ":clear":
                    client.send_control("clear_history")
                
                elif user_input.lower() == ":interrupt":
                    client.send_control("interrupt")
                
                elif user_input.lower() == ":reset":
                    client.send_control("reset_order")
                
                else:
                    # Send regular text input
                    client.send_text(user_input)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean up
        if 'client' in locals():
            client.stop()
        
        print("Client stopped")

if __name__ == "__main__":
    main()


