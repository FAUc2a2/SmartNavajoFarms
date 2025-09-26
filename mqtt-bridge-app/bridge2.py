import os
import json
import uuid
from datetime import datetime
import logging
import threading
import paho.mqtt.client as mqtt
from azure.eventgrid import EventGridPublisherClient, EventGridEvent
from azure.core.credentials import AzureKeyCredential
from http.server import BaseHTTPRequestHandler, HTTPServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration - Ideally, use environment variables or a config file
MQTT_BROKER = os.getenv('MQTT_BROKER', 'sensecap-openstream.seeed.cc')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = os.getenv('MQTT_TOPIC', '/device_sensor_data/441764963729152/+/+/+/+')

MQTT_USERNAME = os.getenv('MQTT_USERNAME', 'org-441764963729152')
MQTT_PASSWORD = os.getenv('MQTT_PASSWORD', '0980534BDC4E4EE0830FDD9F83840C39D6E59BFE681340BEA14B4FB10A28F0A1')
MQTT_CLIENT_ID_PREFIX = os.getenv('MQTT_CLIENT_ID_PREFIX', 'org-441764963729152-quickstart')

EVENT_GRID_ENDPOINT = os.getenv('EVENT_GRID_ENDPOINT', 'https://mqtt-ingest.eastus-1.eventgrid.azure.net/api/events')
EVENT_GRID_KEY = os.getenv('EVENT_GRID_KEY', 'G3bCreYg0tpYBKaLYYhWc1dRxgeXdwbKNvetPuTFi1Iop2yHxfIlJQQJ99AJACYeBjFXJ3w3AAABAZEGkDDM')

if not EVENT_GRID_ENDPOINT or not EVENT_GRID_KEY:
    logger.error("Event Grid Endpoint and Key must be set as environment variables.")
    raise ValueError("Event Grid Endpoint and Key must be set as environment variables.")

# Initialize Event Grid client
event_grid_client = EventGridPublisherClient(
    endpoint=EVENT_GRID_ENDPOINT,
    credential=AzureKeyCredential(EVENT_GRID_KEY)
)

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT Broker!")
        client.subscribe(MQTT_TOPIC)
        logger.info(f"Subscribed to topic: {MQTT_TOPIC}")
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

def on_message(client, userdata, msg):
    message_payload = msg.payload.decode()
    topic = msg.topic
    logger.info(f"Received message on topic {topic}: {message_payload}")

    event = EventGridEvent(
        subject=topic,
        event_type="MQTT.MessageReceived",
        data={"message": message_payload},
        data_version="1.0",
        id=str(uuid.uuid4()),
        event_time=datetime.utcnow()
    )

    try:
        event_grid_client.send(event)
        logger.info("Event sent to Event Grid successfully.")
    except Exception as e:
        logger.error(f"Failed to send event to Event Grid: {e}")

def run_mqtt_client():
    client_id = f"{MQTT_CLIENT_ID_PREFIX}{uuid.uuid4()}"
    client = mqtt.Client(client_id=client_id)
    client.on_connect = on_connect
    client.on_message = on_message

    if MQTT_USERNAME and MQTT_PASSWORD:
        client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_forever()
    except Exception as e:
        logger.error(f"Unable to connect to MQTT Broker: {e}")

class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b'OK')

def run_health_check_server():
    server_address = ('', 8080)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    logger.info("Starting health check server on port 8080")
    httpd.serve_forever()

def main():
    threading.Thread(target=run_health_check_server).start()
    run_mqtt_client()

if __name__ == "__main__":
    main()
