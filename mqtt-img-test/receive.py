import base64
import paho.mqtt.client as mqtt

client = mqtt.Client()

# connect to host address/port
client.connect("localhost", 1883)

# subscribe to relevant MQTT topic
client.subscribe("img")

# function to decode message and write to a jpg file
def on_message(client, userdata, message):
    encoded = str(message.payload.decode('utf-8'))
    decoded = encoded.encode("utf-8")
    imgdata = base64.b64decode(decoded)
    open('received_image.jpg', 'wb').write(imgdata)

#call function whenever a message is received
client.on_message=on_message

# continually listen for messages
client.loop_forever()