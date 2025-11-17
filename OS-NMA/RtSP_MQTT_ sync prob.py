import os
import cv2
import time
from datetime import datetime
from PIL import Image
import paho.mqtt.client as mqtt
import json
import piexif
import threading

# MQTT Broker settings
mqtt_server = "5.196.78.28"
mqtt_port = 1883
topic = "location_data"

# RTSP Stream settings
rtsp_url = "rtsp://user:192.168.0.160@192.168.164.106:8554/streaming/live/1"

# Output directory for geotagged images (ensure this directory already exists)
output_dir = r"C:/Users/User/Geotaged IMG"

# Ensure the directory exists (do not create a new one)
if not os.path.exists(output_dir):
    print(f"Directory does not exist: {output_dir}")
else:
    print(f"Directory exists: {output_dir}")

# Global variables for geolocation and UTC time
current_utc = "N/A"
current_lat = "N/A"
current_lng = "N/A"

# Function to get current UTC time
def get_utc():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# Function to add GPS metadata to an image
def add_gps_metadata(image_path, latitude, longitude, utc_time):
    try:
        img = Image.open(image_path)
        
        # Get the current EXIF data (if it exists)
        exif_dict = piexif.load(img.info.get("exif", b""))
        
        # GPS info: Degrees, Minutes, and Seconds
        gps_ifd = {
            piexif.GPSIFD.GPSLatitudeRef: 'N' if latitude >= 0 else 'S',
            piexif.GPSIFD.GPSLongitudeRef: 'E' if longitude >= 0 else 'W',
            piexif.GPSIFD.GPSLatitude: convert_to_dms(latitude),
            piexif.GPSIFD.GPSLongitude: convert_to_dms(longitude),
            piexif.GPSIFD.GPSAltitude: (0, 1),  # Assuming altitude is 0
            piexif.GPSIFD.GPSTimeStamp: convert_utc_to_gps_time(utc_time),
        }

        # Insert the GPS data into the EXIF metadata
        exif_dict["GPS"] = gps_ifd

        # Save the image with updated EXIF data
        exif_bytes = piexif.dump(exif_dict)
        img.save(image_path, exif=exif_bytes)
        print(f"Geotagged image saved: {image_path}")
    except Exception as e:
        print(f"Failed to save geotagged image: {e}")

# Helper functions to convert coordinates to DMS (Degrees, Minutes, Seconds)
def convert_to_dms(coord):
    degrees = int(coord)
    minutes = int((coord - degrees) * 60)
    seconds = int(((coord - degrees) * 60 - minutes) * 60)
    return ((degrees, 1), (minutes, 1), (seconds, 100))

# Helper function to convert UTC time to GPSTimeStamp (HH, MM, SS)
def convert_utc_to_gps_time(utc_time):
    utc = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S")
    return ((utc.hour, 1), (utc.minute, 1), (utc.second, 1))

# MQTT Callback when a message is received
def on_message(client, userdata, msg):
    global current_utc, current_lat, current_lng
    try:
        # Decode and parse the JSON message payload
        message = msg.payload.decode()
        data = json.loads(message)
        
        # Update current location and UTC time
        current_utc = data.get("utc_time", "N/A")
        current_lat = data.get("latitude", "N/A")
        current_lng = data.get("longitude", "N/A")
        
        print(f"Received message on topic {msg.topic}:")
        print(f"  UTC Time: {current_utc}")
        print(f"  Latitude: {current_lat}")
        print(f"  Longitude: {current_lng}")
        
    except Exception as e:
        print(f"Failed to parse message: {e}")

# Capture a frame from the RTSP stream and save with geotagging
def capture_and_save_frame():
    # Ensure that the geolocation data is available
    if current_lat == "N/A" or current_lng == "N/A" or current_utc == "N/A":
        print("Geolocation data is not available yet. Skipping frame capture.")
        return

    try:
        # Open RTSP stream (using FFmpeg backend)
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        ret, frame = cap.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f"{output_dir}\\image_{timestamp}.jpg"  # Save to the specified directory
            
            # Overlay UTC, Latitude, and Longitude on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_utc = f"UTC: {current_utc}"
            text_lat_lng = f"Lat: {current_lat}, Lng: {current_lng}"
            
            # Position for the text
            position_utc = (-10, frame.shape[0] - 50)
            position_lat_lng = (-10, frame.shape[0] - 10)
            
            # Add the text to the frame
            cv2.putText(frame, text_utc, position_utc, font, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, text_lat_lng, position_lat_lng, font, 0.7, (255, 255, 255), 2)

            # Save the frame as an image (screenshot)
            cv2.imwrite(file_name, frame)
            print(f"Image saved: {file_name}")
            
            # Add geotag metadata to the image
            add_gps_metadata(file_name, current_lat, current_lng, current_utc)
        
        cap.release()
    except Exception as e:
        print(f"Error capturing frame: {e}")

# Function to display RTSP stream with UTC, Latitude, and Longitude overlay
def display_rtsp_stream():
    # Open the RTSP stream with OpenCV for frame capture
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("Failed to open RTSP stream")
        return
    
    # Initialize last capture time
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from RTSP stream")
            break
        
        # Overlay UTC, Latitude, and Longitude at the bottom of the frame
        utc_time = get_utc()
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_utc = f"UTC: {utc_time}"
        text_lat_lng = f"Lat: {current_lat}, Lng: {current_lng}"
        
        # Position for the text
        position_utc = (10, frame.shape[0] - 50)
        position_lat_lng = (10, frame.shape[0] - 100)
        
        # Add the text to the frame
        cv2.putText(frame, text_utc, position_utc, font, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, text_lat_lng, position_lat_lng, font, 0.7, (255, 255, 255), 2)

        # Display the frame with the UTC time, Latitude, and Longitude overlay
        cv2.imshow("RTSP Stream with UTC, Latitude, Longitude", frame)

        # Capture and save frame every 5 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 5:  # Check if 5 seconds have passed
            capture_and_save_frame()  # Capture the frame and save it
            last_capture_time = current_time  # Update the last capture time

        # Check for key press to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# Create a new MQTT client instance
client = mqtt.Client()

# Assign the callback function for message handling
client.on_message = on_message

# Connect to the MQTT broker
print("Connecting to MQTT broker...")
client.connect(mqtt_server, mqtt_port, 60)

# Subscribe to the "location_data" topic
print(f"Subscribing to topic '{topic}'...")
client.subscribe(topic)

# Start the MQTT client loop in a separate thread
mqtt_thread = threading.Thread(target=client.loop_forever)
mqtt_thread.start()

# Display the RTSP stream in the main thread
display_rtsp_stream()
