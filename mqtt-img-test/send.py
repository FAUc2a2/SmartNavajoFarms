import base64
import sys
import paho.mqtt.client as mqtt

client = mqtt.Client()

# connect to host address/port
client.connect("localhost", 1883)

# subscribe to relevant MQTT topic
client.subscribe("img")

# open image address within command line argument
with open(sys.argv[1], "rb") as image:
    img=image.read()

# encode image data in base64
bytes = base64.b64encode(img)
encoded = bytes.decode('utf-8')

# publish image to topic
client.publish('img', encoded)
