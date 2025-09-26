FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1883 8080

ENV MQTT_BROKER=sensecap-openstream.seeed.cc
ENV MQTT_PORT=1883
ENV MQTT_TOPIC=/device_sensor_data/441764963729152/+/+/+/+
ENV MQTT_USERNAME=org-441764963729152
ENV MQTT_PASSWORD=0980534BDC4E4EE0830FDD9F83840C39D6E59BFE681340BEA14B4FB10A28F0A1
ENV EVENT_GRID_ENDPOINT=https://mqtt-ingest.eastus-1.eventgrid.azure.net/api/events
ENV EVENT_GRID_KEY=G3bCreYg0tpYBKaLYYhWc1dRxgeXdwbKNvetPuTFi1Iop2yHxfIlJQQJ99AJACYeBjFXJ3w3AAABAZEGkDDM

CMD ["python", "bridge2.py"]