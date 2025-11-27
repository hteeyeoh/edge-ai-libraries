
import logging
import threading
import time
import paho.mqtt.client as mqtt
from typing import Optional


class RabbitMQMQTTClient:
    """RabbitMQ MQTT Client to handle connections and messaging with robust lifecycle."""

    def __init__(
        self,
        broker: str,
        port: int,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        keepalive: int = 60,
        use_tls: bool = False,
        tls_kwargs: Optional[dict] = None,
        connect_timeout: float = 10.0,
        protocol=mqtt.MQTTv311,  # stick to v3.1.1 unless you need v5
    ):
        self.log = logging.getLogger('RABBITMQ_MQTT_CLIENT')
        # Set your desired level; INFO is fine for production, DEBUG for dev
        self.log.setLevel(logging.INFO)
        self.log.debug("Initializing RabbitMQMQTTClient")

        self.broker = broker
        self.port = port
        self.username = username
        self.password = password
        self.keepalive = keepalive
        self.connected = False
        self.publish_complete = False

        # Synchronization primitives
        self._connect_event = threading.Event()

        # Create client
        self.client = mqtt.Client(client_id=client_id, protocol=protocol)

        # Authentication
        if self.username is not None:
            self.client.username_pw_set(self.username, self.password or "")

        # TLS if required
        if use_tls:
            tls_kwargs = tls_kwargs or {}
            # Example: {'ca_certs': '/path/ca.pem', 'certfile': '/path/cert.pem', 'keyfile': '/path/key.pem', 'tls_version': ssl.PROTOCOL_TLS}
            self.client.tls_set(**tls_kwargs)

        # Callbacks
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_publish = self.on_publish

        # Reconnect strategy
        # (min_delay=1s, max_delay=60s backoff)
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)

        # Start the network loop and connect
        self.log.info(f"Connecting to MQTT Broker at {self.broker}:{self.port}")
        self.client.connect(self.broker, self.port, keepalive=self.keepalive)
        self.client.loop_start()
        self.log.info("Network loop started.")

        # Wait for on_connect (or timeout)
        if not self._connect_event.wait(timeout=connect_timeout):
            self.log.error(f"Connection to {self.broker}:{self.port} timed out after {connect_timeout}s")

    # --- Callbacks ---

    def on_connect(self, client, userdata, flags, rc):
        """Handle the connection result."""
        # rc == 0 means successful connect for MQTTv3.1.1
        if rc == mqtt.MQTT_ERR_SUCCESS:
            self.connected = True
            self._connect_event.set()
            self.log.info(f"Connected successfully to {self.broker}:{self.port}")
        else:
            # Non-zero rc indicates failure; still set event so init doesn't hang.
            self._connect_event.set()
            self.connected = False
            self.log.error(f"Connection failed (rc={rc})")

    def on_disconnect(self, client, userdata, rc):
        """Handle disconnection from the broker."""
        self.connected = False
        # rc == 0 means clean disconnect; non-zero means unexpected
        if rc == 0:
            self.log.info("Disconnected cleanly from broker.")
        else:
            self.log.warning(f"Unexpected disconnection (rc={rc}). Auto-reconnect will be attempted if loop is running.")

    def on_publish(self, client, userdata, mid):
        """Handle publish confirmation."""
        self.log.info(f"Publish confirmed (mid={mid})")
        self.publish_complete = True

    # --- API ---

    def is_connected(self) -> bool:
        """Return the connection status."""
        return self.connected

    def publish(self, topic: str, message: str, qos: int = 0, retain: bool = False, timeout: float = 5.0) -> bool:
        """
        Publish a message. Returns True if the message was queued and acknowledged (for QoS>0),
        or queued successfully (for QoS=0).
        """
        if not topic:
            raise ValueError("Topic must be non-empty")

        self.publish_complete = False
        info = self.client.publish(topic, payload=message, qos=qos, retain=retain)

        # info is MQTTMessageInfo; info.rc indicates immediate queuing status
        if info.rc == mqtt.MQTT_ERR_SUCCESS:
            self.log.debug(f"Publishing message to topic '{topic}': {message}")
        else:
            self.log.error(f"Failed to queue publish to topic '{topic}' (rc={info.rc})")
            return False

        # For QoS 0, there's no PUBACK; consider queued == success
        if qos == 0:
            return True

        # For QoS 1/2, wait for on_publish confirmation
        start = time.time()
        while not self.publish_complete and (time.time() - start) < timeout:
            time.sleep(0.01)
        if not self.publish_complete:
            self.log.warning(f"Publish to '{topic}' not confirmed within {timeout}s")
            return False
        return True

    def disconnect(self):
        """Disconnect from the broker."""
        # loop_stop stops the network thread; do it after disconnect
        try:
            self.client.disconnect()
            self.log.info("Disconnect requested.")
        finally:
            self.connected = False

    def stop(self):
        """Stop the MQTT client loop and disconnect."""
        try:
            self.disconnect()
        finally:
            self.client.loop_stop()
            self.log.info("MQTT client loop stopped.")
