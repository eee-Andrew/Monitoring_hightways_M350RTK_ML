

/*
 * ESP32 GNSS logger & MQTT publisher (independent from drone MSDK)
 *
 * This firmware runs autonomously on an ESP32 board mounted on the drone
 * and is completely independent of the DJI / Mobile SDK / flight controller.
 *
 * Responsibilities:
 *  - Read GNSS data from the connected GPS/OSNMA module via serial (TinyGPS++).
 *  - Obtain a UTC timestamp from an NTP server when WiFi is available.
 *  - Package {utc_time, latitude, longitude} into a JSON message.
 *  - Publish these messages to an MQTT broker on the ground (topic: "location_data").
 *
 * The drone’s MSDK and gimbal logic do NOT interact with this code directly.
 * Synchronization with vision / detection pipelines is done later on the ground
 * using the recorded timestamps.
 */


#include <WiFi.h>
#include <PubSubClient.h>
#include <NTPClient.h>
#include <WiFiUdp.h>
#include <TinyGPS++.h>

// GPS module and other setup
TinyGPSPlus gps;  // GPS object

// WiFi credentials
const char* ssid = "Andrikos";
const char* password = "";

// MQTT Broker settings
const char* mqtt_server = "5.196.78.28";  // Change to your broker IP or hostname
const int mqtt_port = 1883;  // Default MQTT port (non-secure)

// Create WiFi and MQTT client objects
WiFiClient espClient;
PubSubClient client(espClient);

// NTP Setup to get UTC time
WiFiUDP udp;
NTPClient timeClient(udp, "pool.ntp.org", 0, 60000);  // UTC time, update every 60 seconds

// Connect to WiFi
void setup_wifi() {
  Serial.begin(38400);  // Start Serial communication at 38400 baud for GPS and Serial Monitor
  delay(10);
  Serial.println("Connecting to WiFi...");
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("Connected to WiFi");
  Serial.print("connected");
}

// Connect to MQTT broker
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    // Set the client ID as mqtt-explorer-6f4a2eba
    String clientId = "mqtt-explorer-6f4a2eba";
    
    if (client.connect(clientId.c_str())) {
      Serial.println("connected");
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      delay(1000);  // Wait 1 second before retrying
    }
  }
}

void setup() {
  setup_wifi();  // Connect to WiFi
  client.setServer(mqtt_server, mqtt_port);  // Set MQTT broker address and port

  timeClient.begin();  // Start NTP client
}
unsigned long lastPublish = 0;
const unsigned long publishInterval = 1000; // 1 second

void loop() {
  if (!client.connected()) {
    reconnect();  // Reconnect to MQTT broker if not connected
  }
  client.loop();  // Keep the connection alive
  
 while (Serial.available()) {
    gps.encode(Serial.read());
  } 

static unsigned long lastNtpUpdate = 0;
  if (millis() - lastNtpUpdate > 60000) {
    timeClient.update();
    lastNtpUpdate = millis();
  }
 // Publish GPS data every second, if updated
  if (gps.location.isUpdated() && millis() - lastPublish > publishInterval) {
    lastPublish = millis();

    float lat = gps.location.lat();
    float lng = gps.location.lng();
    String utcTime = timeClient.getFormattedTime();

    String message = "{\"utc_time\":\"" + utcTime + "\",\"latitude\":" + String(lat, 8) + ",\"longitude\":" + String(lng, 8) + "}";
    client.publish("location_data", message.c_str());

    Serial.println("Sent message: " + message);
  }
}
