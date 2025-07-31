import cv2
from simple_facerec import SimpleFacerec
import socket
import whisper
import sounddevice as sd
from scipy.io.wavfile import write
import os

# Initialize face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Folder with known faces

# Load Whisper model (base, tiny, or medium as needed)
model = whisper.load_model("base")

# Utility to get local IP address
def get_local_ip_address():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.254.254.254', 1))
        ip_address = s.getsockname()[0]
    except Exception:
        ip_address = '127.0.0.1'
    finally:
        s.close()
    return ip_address

# Record voice and transcribe using Whisper
def record_and_transcribe(filename="command.wav", duration=5, samplerate=44100):
    print("🎙️ Listening for voice command...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(filename, samplerate, audio)

    result = model.transcribe(filename)
    os.remove(filename)
    return result["text"].lower()

# Listen for command like "search messi"
def listen_for_activation(known_names, keywords):
    while True:
        try:
            text = record_and_transcribe()
            print(f"🗣 You said: {text}")

            for keyword in keywords:
                if keyword in text:
                    for name in known_names:
                        if name.lower() in text:
                            print(f"✅ Command recognized: {keyword} {name}")
                            return name
                    print("❗ Name not found. Please try again.")
                    break
        except Exception as e:
            print(f"❌ Error: {e}")

# Face detection loop
def process_video(cap):
    activation_keywords = ["search", "locate", "present"]
    known_names = sfr.known_face_names
    detection_limit = 5
    ip = get_local_ip_address()

    while True:
        target_name = listen_for_activation(known_names, activation_keywords)
        detection_count = 0

        print(f"🔍 Scanning for {target_name}...")

        while detection_count < detection_limit:
            ret, frame = cap.read()
            if not ret:
                break

            face_locations, face_names = sfr.detect_known_faces(frame)

            for face_loc, name in zip(face_locations, face_names):
                if name == target_name:
                    y1, x2, y2, x1 = face_loc
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    print(f"📍 Detected '{name}' at IP: {ip}")
                    detection_count += 1

                    if detection_count >= detection_limit:
                        print(f"✅ Finished detecting {name} ({detection_limit} times).")
                        break

            cv2.imshow("🎦 AI Surveillance Feed", frame)
            if cv2.waitKey(1) == 27:  # ESC key
                return

# Entry point
if __name__ == "__main__":
    print("🚀 Starting Whisper-Based AI Surveillance System")
    print("1: Live Camera\n2: Video File")
    choice = input("Enter your choice (1 or 2): ")

    if choice == "1":
        cap = cv2.VideoCapture(0)
    elif choice == "2":
        path = input("Enter path to video file: ")
        cap = cv2.VideoCapture(path)
    else:
        print("❌ Invalid choice.")
        exit()

    process_video(cap)
    cap.release()
    cv2.destroyAllWindows()

