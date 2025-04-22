"""
Enhanced voice recognition with real-time VU meter display.
This script combines speech recognition with the Vosk library and
a fixed VU meter display at the top of the terminal.
"""
import json
import os
import queue
import threading
import time
import sys
import subprocess
from datetime import datetime

try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False

try:
    import sounddevice as sd
    SD_AVAILABLE = True
except ImportError:
    SD_AVAILABLE = False
    print("Error: sounddevice not installed. Run: pip install sounddevice")
    sys.exit(1)

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Error: numpy not installed. Run: pip install numpy")
    sys.exit(1)

try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False
    print("Error: vosk not installed. Run: pip install vosk")
    sys.exit(1)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis not installed. Redis publishing will be disabled.")
    print("To enable Redis: pip install redis")

# === Configuration ===
class Config:
    def __init__(self):
        self.vosk_model_path = "vosk-model-small-en-us-0.15"  # Path to Vosk model
        self.redis_host = "localhost"
        self.redis_port = 6379
        self.redis_channel_out = "speech_recognition_output"
        self.redis_channel_in = "speech_recognition_control"
        self.sample_rate = 16000
        self.block_size = 4000  # Audio block size
        self.voice_threshold = 0.002  # Threshold for detecting voice activity
        self.min_voice_frames = 5  # Minimum number of voice frames to consider speech
        self.debug = False  # Debug mode
        
        # VU meter settings
        self.vu_meter_width = 60  # Width of the VU meter
        self.vu_update_rate = 0.05  # Update rate for VU meter in seconds
        self.peak_decay = 0.05  # How quickly the peak indicator falls
        self.smoothing_factor = 0.3  # Smoothing factor for the VU meter (0-1)

# === Message Queue for Thread Communication ===
log_queue = queue.Queue()
recognition_queue = queue.Queue()
speech_status_queue = queue.Queue()

# === ANSI Terminal Codes ===
CLEAR_SCREEN = "\033[2J"
MOVE_TO_TOP = "\033[H"
CLEAR_LINE = "\033[K"
MOVE_UP = "\033[1A"
SAVE_CURSOR = "\033[s"
RESTORE_CURSOR = "\033[u"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

# === Terminal UI Class ===
class TerminalUI:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.log_buffer = []
        self.max_log_lines = 20
        self.peak_hold = 0
        self.rms_smoothed = 0
        self.last_partial = ""
        self.recognition_active = True
        self.currently_speaking = False
        
        # Get terminal size
        try:
            terminal_size = os.get_terminal_size()
            self.cols = terminal_size.columns
            self.rows = terminal_size.lines
        except:
            self.rows, self.cols = 24, 80  # Fallback
        
        # Number of lines reserved for the header (VU meter + status)
        self.header_lines = 5
        
    def start(self):
        """Start the terminal UI thread"""
        self.ui_thread = threading.Thread(target=self.ui_loop)
        self.ui_thread.daemon = True
        self.ui_thread.start()
        
    def stop(self):
        """Stop the terminal UI thread"""
        self.running = False
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=1)
        print(SHOW_CURSOR)  # Ensure cursor is shown when exiting
        
    def ui_loop(self):
        """Main UI loop"""
        # Initial clear screen
        sys.stdout.write(CLEAR_SCREEN + MOVE_TO_TOP + HIDE_CURSOR)
        sys.stdout.flush()
        
        # Draw initial interface
        self.draw_header()
        self.draw_status_line()
        self.draw_log_area()
        
        # Main UI loop
        last_update = 0
        while self.running:
            current_time = time.time()
            
            # Process queued messages
            self.process_queues()
            
            # Update VU meter at specified rate
            if current_time - last_update > self.config.vu_update_rate:
                self.draw_vu_meter()
                self.draw_status_line()
                last_update = current_time
                
            time.sleep(0.01)  # Small delay to prevent CPU overload
            
        # Clean up terminal on exit
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.flush()
        
    def process_queues(self):
        """Process messages from the queues"""
        # Process log messages
        while not log_queue.empty():
            message = log_queue.get_nowait()
            self.add_log_message(message)
            
        # Process recognition results
        while not recognition_queue.empty():
            text, is_partial = recognition_queue.get_nowait()
            if is_partial:
                self.last_partial = text
            else:
                self.log_message(f"[Recognized] {text}")
                self.last_partial = ""
                
        # Process speech status
        while not speech_status_queue.empty():
            speaking = speech_status_queue.get_nowait()
            self.currently_speaking = speaking
            
    def add_log_message(self, message):
        """Add a message to the log buffer and redraw log area"""
        # Split multi-line messages
        lines = message.split('\n')
        for line in lines:
            if line:  # Skip empty lines
                self.log_buffer.append(line)
                
        # Trim buffer if needed
        while len(self.log_buffer) > self.max_log_lines:
            self.log_buffer.pop(0)
            
        # Redraw log area
        self.draw_log_area()
        
    def log_message(self, message):
        """Add a timestamped message to the log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_log_message(f"[{timestamp}] {message}")
        
    def draw_header(self):
        """Draw the header area including title"""
        # Move to top of screen
        sys.stdout.write(MOVE_TO_TOP)
        
        # Title bar
        title = " Enhanced Voice Recognition with VU Meter "
        padding = (self.cols - len(title)) // 2
        title_line = "=" * padding + title + "=" * (self.cols - len(title) - padding)
        sys.stdout.write(title_line + "\n")
        sys.stdout.flush()
        
    def draw_vu_meter(self):
        """Draw the VU meter"""
        # Move to position (line 2)
        sys.stdout.write(MOVE_TO_TOP + "\n\n")
        
        # VU meter title with levels
        db_rms = 20 * np.log10(max(self.rms_smoothed, 1e-9))
        db_peak = 20 * np.log10(max(self.peak_hold, 1e-9))
        sys.stdout.write(f"Microphone Level - RMS: {db_rms:.1f} dB  Peak: {db_peak:.1f} dB\n")
        
        # Create VU meter
        meter_width = min(self.config.vu_meter_width, self.cols - 4)
        rms_len = int(self.rms_smoothed * meter_width * 2)  # Scale for better visibility
        peak_pos = int(self.peak_hold * meter_width * 2)
        
        # Clamp values
        if rms_len > meter_width:
            rms_len = meter_width
        if peak_pos > meter_width:
            peak_pos = meter_width
            
        # Build meter string with color zones
        meter = "["
        for i in range(meter_width):
            # Determine if this position has the peak indicator
            is_peak = (i == peak_pos)
            
            # Determine the character based on whether this position is filled by the RMS level
            if i < rms_len:
                if is_peak:
                    meter += "|"  # Peak position within the filled area
                else:
                    # Color coding within the filled area
                    pos_ratio = i / meter_width
                    if pos_ratio > 0.9:  # Red zone
                        meter += "#"
                    elif pos_ratio > 0.75:  # Yellow zone
                        meter += "="
                    else:  # Green zone
                        meter += "■"
            else:
                if is_peak:
                    meter += "|"  # Peak position outside the filled area
                else:
                    meter += " "  # Unfilled area
        meter += "]"
        
        # Output the VU meter
        sys.stdout.write(meter + "\n")
        
        # Display scale
        scale = "-60dB" + " " * (meter_width - 14) + "-10dB" + " " * 5 + "0dB"
        sys.stdout.write(scale + "\n")
        
        sys.stdout.flush()
        
    def draw_status_line(self):
        """Draw the status line below the VU meter"""
        # Move to position (line 5)
        sys.stdout.write(MOVE_TO_TOP + "\n\n\n\n\n")
        
        # Create status line
        rec_status = "ACTIVE" if self.recognition_active else "PAUSED"
        speaking_status = "SPEAKING" if self.currently_speaking else "SILENT"
        
        # Partial recognition display
        partial_display = self.last_partial[:self.cols-20] if self.last_partial else ""
        if partial_display:
            partial_display = f"[Partial] {partial_display}"
            
        # Status line
        status = f"Recognition: {rec_status} | Mic: {speaking_status}"
        
        # Output status line, clearing to end of line
        sys.stdout.write(status + CLEAR_LINE + "\n")
        
        # Output partial recognition if present
        if partial_display:
            sys.stdout.write(partial_display + CLEAR_LINE + "\n")
        else:
            sys.stdout.write(CLEAR_LINE + "\n")
            
        # Draw separator line
        sys.stdout.write("-" * self.cols + "\n")
        
        sys.stdout.flush()
        
    def draw_log_area(self):
        """Draw the scrolling log area"""
        # Move to start of log area (after header)
        sys.stdout.write(MOVE_TO_TOP)
        for _ in range(self.header_lines):
            sys.stdout.write("\n")
            
        # Output each log line, clearing to end of line
        for line in self.log_buffer:
            # Truncate long lines
            if len(line) > self.cols:
                line = line[:self.cols-3] + "..."
                
            sys.stdout.write(line + CLEAR_LINE + "\n")
            
        # Clear any remaining lines in the log area
        for _ in range(self.max_log_lines - len(self.log_buffer)):
            sys.stdout.write(CLEAR_LINE + "\n")
            
        sys.stdout.flush()
        
    def update_audio_levels(self, rms, peak):
        """Update audio levels for the VU meter"""
        # Smooth RMS for more stable display
        self.rms_smoothed = self.config.smoothing_factor * rms + (1 - self.config.smoothing_factor) * self.rms_smoothed
        
        # Update peak hold with decay
        if peak > self.peak_hold:
            self.peak_hold = peak
        else:
            self.peak_hold = max(0, self.peak_hold - self.config.peak_decay)

# === Voice Recognition Service ===
class EnhancedVoiceService:
    def __init__(self, config, ui):
        self.config = config
        self.ui = ui
        
        # Redis client (optional)
        self.redis_client = None
        self.pubsub = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=config.redis_host,
                    port=config.redis_port
                )
                self.redis_client.ping()
                log_message(f"Connected to Redis at {config.redis_host}:{config.redis_port}")
                
                # Subscribe to control channel
                self.pubsub = self.redis_client.pubsub(ignore_subscribe_messages=True)
                self.pubsub.subscribe(config.redis_channel_in)
                
            except Exception as e:
                log_message(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        # Check for Vosk model
        if not os.path.exists(config.vosk_model_path):
            raise RuntimeError(f"Vosk model not found at {config.vosk_model_path}")
        
        # Load Vosk model
        log_message(f"Loading Vosk model from {config.vosk_model_path}...")
        self.model = Model(config.vosk_model_path)
        self.recognizer = KaldiRecognizer(self.model, config.sample_rate)
        self.recognizer.SetWords(True)
        log_message("Model loaded successfully")
        
        # Audio device selection
        self.device = self._select_mic_device()
        
        # Status flags
        self.stop_event = threading.Event()
        self.paused_event = threading.Event()
        self.pause_lock = threading.Lock()
        
        # Thread for processing control messages
        self.control_thread = None
        
        # Last recognized text to avoid duplicates
        self.last_text = ""
        
        # Voice activity detection 
        self.voice_frames = 0
        self.silence_frames = 0
        
        # Audio stream
        self.stream = None
    
    def _select_mic_device(self):
        """List and let user select a microphone"""
        log_message("\nAvailable Microphones:")
        
        devices = sd.query_devices()
        mic_devices = []
        
        for i, dev in enumerate(devices):
            if dev.get('max_input_channels', 0) > 0:
                try:
                    sd.check_input_settings(device=i, samplerate=self.config.sample_rate)
                    mic_devices.append((i, dev['name']))
                    log_message(f"[{len(mic_devices)-1}] {dev['name']} (device id: {i})")
                except:
                    # Skip devices that don't support our sample rate
                    pass
        
        if not mic_devices:
            raise RuntimeError("No compatible microphones found!")
        
        while True:
            try:
                choice = input("\nSelect microphone number: ")
                idx = int(choice)
                if 0 <= idx < len(mic_devices):
                    device_id = mic_devices[idx][0]
                    log_message(f"Selected: {mic_devices[idx][1]} (device id: {device_id})")
                    return device_id
                else:
                    print("Invalid selection. Try again.")
            except ValueError:
                print("Please enter a number.")
                
    def _control_thread_func(self):
        """Thread to listen for control messages from Redis"""
        if not self.redis_client or not self.pubsub:
            return
            
        while not self.stop_event.is_set():
            try:
                # Get and process control messages
                message = self.pubsub.get_message(timeout=0.1)
                if message and message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        action = data.get("action", "")
                        
                        if action == "pause":
                            self._handle_pause()
                        elif action == "resume":
                            self._handle_resume()
                        elif action == "stop":
                            self.stop_event.set()
                    except json.JSONDecodeError:
                        log_message("Invalid JSON in control message")
                    except Exception as e:
                        log_message(f"Error processing control message: {e}")
            except Exception as e:
                log_message(f"Error in control thread: {e}")
            
            time.sleep(0.01)
    
    def _handle_pause(self):
        """Pause speech recognition"""
        with self.pause_lock:
            if not self.paused_event.is_set():
                self.paused_event.set()
                log_message("[Speech recognition paused]")
    
    def _handle_resume(self):
        """Resume speech recognition"""
        with self.pause_lock:
            if self.paused_event.is_set():
                # Add a small delay before resuming to avoid cross-talk
                time.sleep(0.5)
                self.paused_event.clear()
                
                # Reset the recognizer state
                self.recognizer.Reset()
                
                # Reset voice activity detection counters
                self.voice_frames = 0
                self.silence_frames = 0
                
                log_message("[Speech recognition resumed]")
    
    def _send_recognized_text(self, text):
        """Send recognized text to Redis"""
        # Skip if empty or duplicate
        if not text or text == self.last_text:
            return
        
        # Update last recognized text
        self.last_text = text
        
        # Publish to Redis if available
        if self.redis_client:
            try:
                # Create message payload
                message = {
                    "type": "speech_recognition",
                    "text": text,
                    "timestamp": time.time()
                }
                
                # Publish to Redis
                self.redis_client.publish(
                    self.config.redis_channel_out,
                    json.dumps(message)
                )
                log_message(f"Published to Redis: '{text}'")
            except Exception as e:
                log_message(f"Error publishing to Redis: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            log_message(f"Audio status: {status}")
        
        # Convert to float for analysis
        audio_float = indata.copy().astype(np.float32) / 32768.0
        
        # Calculate audio statistics
        rms = np.sqrt(np.mean(audio_float**2))
        peak = np.max(np.abs(audio_float))
        
        # Update UI with audio levels
        self.ui.update_audio_levels(rms, peak)
        
        # Skip processing if paused or stopped
        if self.paused_event.is_set() or self.stop_event.is_set():
            return
        
        # Convert audio data to bytes for Vosk
        audio_data = bytes(indata)
        
        # Check for voice activity
        if rms > self.config.voice_threshold:
            self.voice_frames += 1
            self.silence_frames = 0
            
            # Indicate speaking status if crossed threshold
            if self.voice_frames >= self.config.min_voice_frames:
                speech_status_queue.put(True)
        else:
            self.silence_frames += 1
            if self.voice_frames > 0:
                self.voice_frames -= 1
                
            # Reset speaking status after enough silence
            if self.silence_frames > 10 and self.voice_frames < self.config.min_voice_frames:
                speech_status_queue.put(False)
        
        # Process with Vosk if voice activity detected
        if self.voice_frames >= self.config.min_voice_frames:
            if self.recognizer.AcceptWaveform(audio_data):
                result = json.loads(self.recognizer.Result())
                text = result.get("text", "").strip()
                
                if text:
                    recognition_queue.put((text, False))
                    self._send_recognized_text(text)
            else:
                # Get partial result
                partial = json.loads(self.recognizer.PartialResult())
                partial_text = partial.get("partial", "").strip()
                if partial_text:
                    recognition_queue.put((partial_text, True))
    
    def start(self):
        """Start the voice recognition service"""
        log_message("Starting enhanced voice recognition service...")
        log_message(f"Model: {self.config.vosk_model_path}")
        log_message(f"Device: {self.device}")
        log_message("Press Ctrl+C to stop")
        
        # Reset flags
        self.stop_event.clear()
        self.paused_event.clear()
        
        # Start control thread if Redis is available
        if self.redis_client:
            self.control_thread = threading.Thread(target=self._control_thread_func)
            self.control_thread.daemon = True
            self.control_thread.start()
        
        try:
            # Start audio stream
            self.stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                blocksize=self.config.block_size,
                device=self.device,
                dtype='int16',
                channels=1,
                callback=self._audio_callback
            )
            
            log_message("\n🎤 Listening... Speak into your microphone.")
            
            with self.stream:
                # Keep main thread alive
                while not self.stop_event.is_set():
                    # Check for keyboard input (for commands)
                    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                        cmd = sys.stdin.readline().strip().lower()
                        if cmd == 'q':
                            log_message("Quitting...")
                            self.stop_event.set()
                        elif cmd == 'p':
                            if self.paused_event.is_set():
                                self._handle_resume()
                            else:
                                self._handle_pause()
                    
                    time.sleep(0.1)
        
        except KeyboardInterrupt:
            log_message("\nStopped by user")
        except Exception as e:
            log_message(f"\nError in voice recognition: {e}")
        finally:
            # Clean up
            self.stop_event.set()
            
            if self.control_thread and self.control_thread.is_alive():
                self.control_thread.join(timeout=1)
            
            log_message("Voice recognition stopped")
    
    def stop(self):
        """Stop the voice recognition service"""
        self.stop_event.set()
        
        if self.stream:
            self.stream.close()
        
        log_message("Stopping voice recognition...")

# === Helper Functions ===
def log_message(message):
    """Add a message to the log queue"""
    log_queue.put(message)
    
# For keyboard input handling
import select

# === Main Function ===
def main():
    try:
        # Create config
        config = Config()
        
        # Create terminal UI
        ui = TerminalUI(config)
        ui.start()
        
        # Create and start service
        service = EnhancedVoiceService(config, ui)
        
        # Clear screen after device selection
        print(CLEAR_SCREEN)
        
        # Start voice recognition
        service.start()
        
        # Cleanup
        ui.stop()
        
    except KeyboardInterrupt:
        log_message("\nProgram terminated by user")
    except Exception as e:
        log_message(f"Error: {e}")

if __name__ == "__main__":
    main()

