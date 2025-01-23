# Image over MQTT

This is a proof-of-concept project for sending and receiving images over MQTT.

The code written assumes an MQTT broker (usually Mosquitto) is already set-up locally, however can also easily be adapted to using an external MQTT broker. For demonstration, the MQTT topic `img` was configured as a target topic by default.

## Sending images

The `send.py` program is used to send an image over the configured MQTT topic. The name of the jpg file must be provided as an argument for the program to run properly. For example:

```
>python send.py example1.jpg
```

## Receiving images

The `receive.py` program is used to continually listen for images sent over the configured MQTT topic. Simply running the program is enough to begin listening, and the program will run and listen continually until it is stopped. Received images will save to a file named `received_image.jpg`.

*Note that this program will only listen for new messages sent while the program is running. During testing, `receive.py` should be run first.*

## Additional notes

- **This code is written for images saved in `.jpg` format.** For cases where only one image type can be expected, reconfiguration is as simple as changing the filetype extension of the saved file in `receive.py`. For cases where image types may be varied, a more robust implementation would be necessary.
- Two example image files have been included for testing. Note the small size of these images. Larger image files may introduce latency depending on use-case.
- The code presented is heavily adapted from [this](https://www.youtube.com/watch?v=dW-b5S7cOTw) video by user Iris Codes.
