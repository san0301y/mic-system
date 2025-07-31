import cv2
from simple_facerec import SimpleFacerec
import socket
import speech_recognition as sr

# Initialize face recognizer
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# IP address utility
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

# Voice input logic
def listen_for_activation(known_names, keywords):
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source)
    
    while True:
        with mic as source:
            print("👂 Say a command...")
            try:
                audio = r.listen(source, timeout=5)
                text = r.recognize_google(audio).lower()
                print(f"🗣 Heard: {text}")

                for keyword in keywords:
                    if keyword in text:
                        for name in known_names:
                            if name.lower() in text:
                                print(f"✅ You said: \"{keyword} {name}\" — starting face detection.")
                                return name
                        print("⚠️ Name not recognized. Please try again.")
                        break
            except sr.WaitTimeoutError:
                print("⏱ Timeout: No voice detected.")
            except sr.UnknownValueError:
                print("⚠️ Could not understand audio.")
            except sr.RequestError:
                print("❌ Network error. Check your connection.")

# Main detection loop
def process_video(cap):
    activation_keywords = ["search", "locate", "present"]
    known_names = sfr.known_face_names
    detection_limit = 5
    ip = get_local_ip_address()

    while True:
        target_name = listen_for_activation(known_names, activation_keywords)
        detection_count = 0

        print(f"📷 Scanning for {target_name}...")

        while detection_count < detection_limit:
            ret, frame = cap.read()
            if not ret:
                break

            face_locations, face_names = sfr.detect_known_faces(frame)

            for face_loc, name in zip(face_locations, face_names):
                if name == target_name:
                    y1, x2, y2, x1 = face_loc
                    color = (0, 255, 0)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    print(f"📍 Detected '{name}' at IP: {ip}")
                    detection_count += 1

                    if detection_count >= detection_limit:
                        print(f"✅ Finished detecting {name} ({detection_limit} times).")
                        break

            cv2.imshow("AI Surveillance Feed", frame)
            if cv2.waitKey(1) == 27:  # ESC to exit
                return

# Entry point
if __name__ == "__main__":
    print("🔧 Starting AI Surveillance System")

    print("1: Live Camera Feed\n2: Recorded Video Feed")
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

