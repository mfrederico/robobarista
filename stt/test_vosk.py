"""
Microphone diagnostics script for Python.
This script will help diagnose common issues with microphone access in Python.
"""
import sys
import platform
import subprocess
import os

# Check Python version and platform
def check_system():
    print("\n=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    
    # Platform-specific checks
    if platform.system() == "Windows":
        print("\nChecking Windows audio devices...")
        try:
            # List Windows audio devices using PowerShell
            cmd = "powershell -Command \"Get-WmiObject Win32_SoundDevice | Select-Object Name, Status\""
            output = subprocess.check_output(cmd, shell=True).decode()
            print(output)
        except Exception as e:
            print(f"Error getting Windows audio devices: {e}")
            
    elif platform.system() == "Darwin":  # macOS
        print("\nChecking macOS audio devices...")
        try:
            cmd = "system_profiler SPAudioDataType | grep -A 3 'Input'"
            output = subprocess.check_output(cmd, shell=True).decode()
            print(output)
        except Exception as e:
            print(f"Error getting macOS audio devices: {e}")
            
    elif platform.system() == "Linux":
        print("\nChecking Linux audio devices...")
        try:
            cmd = "arecord -l"
            output = subprocess.check_output(cmd, shell=True).decode()
            print(output)
        except Exception as e:
            print(f"Error getting Linux audio devices: {e}")

# Check Python audio packages
def check_packages():
    print("\n=== Python Audio Packages ===")
    packages = [
        "sounddevice",
        "pyaudio",
        "numpy",
        "vosk"
    ]
    
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, "__version__", "Unknown")
            print(f"✅ {package} is installed (version: {version})")
        except ImportError:
            print(f"❌ {package} is NOT installed")
        except Exception as e:
            print(f"⚠️ {package} error: {e}")

# Test basic sounddevice functionality
def test_sounddevice():
    print("\n=== Testing sounddevice ===")
    try:
        import sounddevice as sd
        
        # List devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        input_devices = []
        
        for i, dev in enumerate(devices):
            in_ch = dev.get('max_input_channels', 0)
            if in_ch > 0:
                print(f"[{i}] {dev['name']} (inputs: {in_ch})")
                input_devices.append(i)
                
        if not input_devices:
            print("❌ No input devices found!")
            return False
            
        # Check default device
        try:
            default_device = sd.query_devices(kind='input')
            print(f"\nDefault input device: [{sd.default.device[0]}] {default_device['name']}")
        except Exception as e:
            print(f"❌ Error getting default device: {e}")
            
        return True
            
    except Exception as e:
        print(f"❌ sounddevice error: {e}")
        return False

# Test PyAudio functionality
def test_pyaudio():
    print("\n=== Testing PyAudio ===")
    try:
        import pyaudio
        
        p = pyaudio.PyAudio()
        
        # Get device count
        device_count = p.get_device_count()
        input_devices = []
        
        print(f"\nFound {device_count} audio devices:")
        for i in range(device_count):
            dev_info = p.get_device_info_by_index(i)
            max_inputs = dev_info.get('maxInputChannels')
            if max_inputs > 0:
                print(f"[{i}] {dev_info['name']} (inputs: {max_inputs})")
                input_devices.append(i)
                
        if not input_devices:
            print("❌ No input devices found with PyAudio!")
            p.terminate()
            return False
            
        # Check default device
        try:
            default_index = p.get_default_input_device_info()['index']
            default_name = p.get_device_info_by_index(default_index)['name']
            print(f"\nDefault input device: [{default_index}] {default_name}")
        except Exception as e:
            print(f"❌ Error getting default device: {e}")
            
        p.terminate()
        return True
            
    except ImportError:
        print("❌ PyAudio is not installed")
        return False
    except Exception as e:
        print(f"❌ PyAudio error: {e}")
        return False

# Test recording with VU meter
def test_recording():
    print("\n=== Testing Audio Recording with VU Meter ===")
    try:
        import sounddevice as sd
        import numpy as np
        import time
        import threading
        import sys
        
        # Parameters
        sample_rate = 16000
        block_duration = 0.1  # Duration of each audio block in seconds
        block_size = int(sample_rate * block_duration)
        running = True
        
        # Function to display VU meter
        def vu_meter(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            # Convert to float for analysis
            audio_data = indata.copy().astype(np.float32) / 32768.0
            # Calculate RMS for this block
            rms = np.sqrt(np.mean(audio_data**2))
            # Calculate VU meter length (max 50 characters)
            meter_len = int(rms * 500)
            if meter_len > 50:
                meter_len = 50
            # Print VU meter
            meter = '|' + '█' * meter_len + ' ' * (50 - meter_len) + '| '
            # Add peak value indicator
            peak = np.max(np.abs(audio_data))
            peak_str = f"Peak: {peak:.6f}"
            rms_str = f"RMS: {rms:.6f}"
            level_indicator = "Low" if rms < 0.005 else "Good" if rms < 0.1 else "High"
            # Clear line and print meter
            print(f"\r{meter} {level_indicator} ({rms_str}, {peak_str})", end='')
        
        # Function to handle keypress
        def wait_for_keypress():
            nonlocal running
            input("\nPress Enter to stop recording...")
            running = False
        
        # Start keypress detection in a separate thread
        threading.Thread(target=wait_for_keypress, daemon=True).start()
        
        # Start the vu meter stream
        print("\nStarting microphone monitoring...")
        print("Please speak into your microphone to see the VU meter respond.")
        print("The bars should move when you speak. If they don't, your microphone might not be working.")
        
        # Track statistics for final report
        all_rms_values = []
        
        # Start the vu meter stream
        with sd.InputStream(callback=vu_meter, channels=1, samplerate=sample_rate, 
                            blocksize=block_size, dtype='int16'):
            
            # Keep monitoring until user presses Enter
            while running:
                # Record a small segment for statistics
                recording = sd.rec(block_size, samplerate=sample_rate, channels=1, dtype='int16')
                sd.wait()
                recording_float = recording.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(recording_float**2))
                all_rms_values.append(rms)
                time.sleep(0.1)  # Short delay to prevent CPU overload
        
        # Calculate overall statistics
        if all_rms_values:
            avg_rms = np.mean(all_rms_values)
            max_rms = np.max(all_rms_values)
            
            print("\n\nAudio Statistics from Monitoring Session:")
            print(f"- Average RMS value: {avg_rms:.6f}")
            print(f"- Maximum RMS value: {max_rms:.6f}")
            
            # Provide diagnostic info
            if max_rms < 0.001:
                print("\n⚠️ WARNING: Very low audio levels detected!")
                print("Your microphone might not be working correctly or is muted.")
                return False
            elif max_rms < 0.005:
                print("\n⚠️ WARNING: Low audio levels detected")
                print("Try speaking louder or adjusting microphone settings")
                return True
            else:
                print("\n✅ Audio recording successful! Microphone appears to be working")
                return True
        else:
            print("\n⚠️ No audio data was collected")
            return False
        
    except Exception as e:
        print(f"\n❌ Recording error: {e}")
        print("This suggests your microphone can't be accessed properly")
        return False

# Check for common OS permission issues
def check_permissions():
    print("\n=== Checking Permissions ===")
    
    if platform.system() == "Windows":
        print("On Windows, check if microphone access is enabled:")
        print("1. Go to Settings > Privacy > Microphone")
        print("2. Ensure 'Allow apps to access your microphone' is ON")
        
    elif platform.system() == "Darwin":  # macOS
        print("On macOS, check microphone permissions:")
        print("1. Go to System Preferences > Security & Privacy > Privacy > Microphone")
        print("2. Ensure Python/Terminal has permission to use the microphone")
        
    elif platform.system() == "Linux":
        print("On Linux, check audio group membership:")
        try:
            cmd = "groups"
            output = subprocess.check_output(cmd, shell=True).decode()
            print(f"Your user is in these groups: {output.strip()}")
            if "audio" in output:
                print("✅ User is in the 'audio' group")
            else:
                print("⚠️ User may not be in the 'audio' group. Consider running:")
                print("   sudo usermod -a -G audio $USER")
                print("   Then log out and back in")
        except Exception as e:
            print(f"Error checking groups: {e}")

# Provide suggestions based on diagnostic results
def provide_suggestions(sd_works, pa_works, recording_works):
    print("\n=== Suggestions ===")
    
    if not sd_works and not pa_works:
        print("❌ Major issues detected with audio libraries")
        print("Suggestions:")
        print("1. Reinstall audio packages:")
        print("   pip uninstall sounddevice pyaudio")
        print("   pip install sounddevice pyaudio")
        print("2. On Windows, ensure Microsoft Visual C++ is installed")
        print("3. On Linux, install audio development libraries:")
        print("   sudo apt-get install portaudio19-dev python3-all-dev")
        print("4. On macOS, install portaudio via Homebrew:")
        print("   brew install portaudio")
    
    elif not recording_works:
        print("❌ Libraries detected devices but recording failed")
        print("Suggestions:")
        print("1. Check system audio levels and ensure microphone is not muted")
        print("2. Try a different microphone or audio input device")
        print("3. Check OS permissions as described above")
        print("4. Restart your system to reset audio services")
        print("5. Check if any other applications are using the microphone")
    
    else:
        print("✅ Basic microphone functionality seems to be working")
        print("If you're still having issues with Vosk specifically:")
        print("1. Ensure Vosk model is properly downloaded and extracted")
        print("2. Check that the model path is correct")
        print("3. Try a different Vosk model (e.g., larger or specific to your language)")
        print("4. Make sure you're speaking clearly and directly into the microphone")

# Add a function to test Vosk if installed
def test_vosk():
    print("\n=== Testing Vosk Speech Recognition ===")
    try:
        from vosk import Model, KaldiRecognizer, SetLogLevel
        import json
        
        # Check if vosk model exists
        model_paths = [
            "vosk-model-small-en-us-0.15",
            "./vosk-model-small-en-us-0.15",
            os.path.expanduser("~/vosk-model-small-en-us-0.15")
        ]
        
        model_found = False
        model_path = None
        
        print("Looking for Vosk model...")
        for path in model_paths:
            if os.path.exists(path):
                model_found = True
                model_path = path
                print(f"✅ Found Vosk model at: {path}")
                break
                
        if not model_found:
            print("❌ Vosk model not found in common locations")
            print("Please provide the model path:")
            model_path = input("> ")
            if os.path.exists(model_path):
                model_found = True
                print(f"✅ Found Vosk model at: {model_path}")
            else:
                print(f"❌ Model not found at: {model_path}")
                print("Suggestions:")
                print("1. Download a model from https://alphacephei.com/vosk/models")
                print("2. Extract it to a known location")
                print("3. Make sure the path is correct")
                return False
        
        # Try to load the model
        print(f"Loading Vosk model from {model_path}...")
        model = Model(model_path)
        print("✅ Model loaded successfully")
        
        # Create recognizer
        recognizer = KaldiRecognizer(model, 16000)
        print("✅ Recognizer created successfully")
        
        print("\nVosk appears to be working correctly!")
        return True
        
    except ImportError:
        print("❌ Vosk is not installed")
        return False
    except Exception as e:
        print(f"❌ Vosk error: {e}")
        return False

# Main function
def main():
    print("Python Microphone Diagnostic Tool")
    print("================================")
    
    # Run all checks
    check_system()
    check_packages()
    check_permissions()
    
    # Test audio libraries
    sd_works = test_sounddevice()
    pa_works = test_pyaudio()
    
    # Test recording only if at least one library works
    recording_works = False
    if sd_works:
        recording_works = test_recording()
    
    # Test Vosk if needed
    vosk_works = False
    test_vosk_option = input("\nWould you like to test Vosk speech recognition? (y/n): ")
    if test_vosk_option.lower() == 'y':
        vosk_works = test_vosk()
    
    # Provide tailored suggestions
    provide_suggestions(sd_works, pa_works, recording_works)
    
    # Additional Vosk-specific suggestions
    if test_vosk_option.lower() == 'y' and not vosk_works:
        print("\n=== Vosk-Specific Suggestions ===")
        print("1. Ensure Vosk is properly installed: pip install vosk")
        print("2. Download the appropriate model from https://alphacephei.com/vosk/models")
        print("3. Extract the model to a directory and provide the correct path")
        print("4. Check if your Python architecture (32/64 bit) matches the Vosk package")
    
    print("\nDiagnostic Complete.")
    
if __name__ == "__main__":
    main()
