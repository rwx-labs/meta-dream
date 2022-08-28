import logging
import functools
import os
from typing import Union
import json


import pika
from pika.exchange_type import ExchangeType

from meta_dream.stable_diffusion import StableDiffusionPipeline

LOGGER = logging.getLogger(__name__)


class Worker:
    QUEUE_SUFFIX = "stable_diffusion_v1_4.meta-dream.labs.rwx.im"
    EXCHANGE = "stable_diffusion_v1_4.meta-dream.labs.rwx.im"
    EXCHANGE_TYPE = ExchangeType.topic
    ROUTING_KEY_SUFFIX = "prompt"

    def __init__(
        self,
        amqp_url: str,
        hugging_face_token: str,
        model_cache_dir: Union[str, bytes, os.PathLike],
    ):
        self.url = amqp_url
        self.channel = None
        self.connection = None
        self.closing = False

        LOGGER.info("Loading Stable Diffusion pipeline")

        self.pipeline = StableDiffusionPipeline(
            hugging_face_token, model_cache_dir=model_cache_dir
        )

        scheduler_name = self.pipeline.scheduler_name()
        self.routing_key = f"prompt"
        self.queue = f"{scheduler_name}.{self.QUEUE_SUFFIX}"

    def on_open(self):
        print("on_open")

    def connect(self):
        """This method connects to RabbitMQ, returning the connection handle.
        When the connection is established, the on_connection_open method
        will be invoked by pika.

        :rtype: pika.SelectConnection

        """
        LOGGER.info("Connecting to %s", self.url)

        return pika.SelectConnection(
            parameters=pika.URLParameters(self.url),
            on_open_callback=self.on_connection_open,
            on_open_error_callback=self.on_connection_open_error,
            on_close_callback=self.on_connection_closed,
        )

    def run(self):
        self.connection = self.connect()
        self.connection.ioloop.start()

    def on_connection_open(self, _unused_connection):
        LOGGER.info("Connection opened")

        LOGGER.info("Creating new channel")
        self.connection.channel(on_open_callback=self.on_channel_open)

    def on_channel_open(self, channel):
        LOGGER.info("Channel opened")

        self.channel = channel
        self.channel.add_on_close_callback(self.on_channel_closed)

        queue_name = self.queue
        cb = functools.partial(self.on_exchange_declareok, userdata=queue_name)
        self.channel.exchange_declare(
            exchange=self.EXCHANGE, exchange_type=self.EXCHANGE_TYPE, callback=cb
        )

    def on_exchange_declareok(self, _unused_frame, userdata):
        LOGGER.info("Exchange declared: %s", userdata)

        queue_name = self.queue
        LOGGER.info("Declaring queue %s", queue_name)

        cb = functools.partial(self.on_queue_declareok, userdata=queue_name)
        self.channel.basic_qos(prefetch_count=1)
        self.channel.queue_declare(queue=queue_name, callback=cb)

    def on_queue_declareok(self, _unused_frame, userdata):
        queue_name = userdata
        LOGGER.info(
            "Binding %s to %s with %s", self.EXCHANGE, queue_name, self.routing_key
        )
        cb = functools.partial(self.on_bindok, userdata=queue_name)
        self.channel.queue_bind(
            queue_name, self.EXCHANGE, routing_key=self.routing_key, callback=cb
        )

        self.start_consuming()

    def on_bindok(self, _unused_frame, userdata):
        LOGGER.info("Queue bound: %s", userdata)

    def on_channel_closed(self, channel, reason):
        LOGGER.warning("Channel %i was closed: %s", channel, reason)

    def on_connection_open_error(self, _unused_connection, err):
        LOGGER.error("Connection open failed: %s", err)

    def on_connection_closed(self, _unused_connection, reason):
        self.channel = None

        if self.closing:
            self.connection.ioloop.stop()
        else:
            LOGGER.warning("Connection closed, reconnect necessary: %s", reason)

    def on_message(self, _unused_channel, basic_deliver, properties, body):
        LOGGER.info(
            "Received message # %s from %s: %s",
            basic_deliver.delivery_tag,
            properties.app_id,
            body,
        )

        try:
            data = json.loads(body)
            prompt = data.get("prompt", None)
            seed = data.get("seed", None)

            result = self.pipeline(prompt=prompt, seed=seed)
            self.acknowledge_message(basic_deliver.delivery_tag)
        except json.JSONDecodeError as e:
            LOGGER.warn("Could not deserialize RabbitMQ message as JSON")

    def acknowledge_message(self, delivery_tag):
        """Acknowledge the message delivery from RabbitMQ by sending a
        Basic.Ack RPC method for the delivery tag.
        :param int delivery_tag: The delivery tag from the Basic.Deliver frame
        """
        LOGGER.info("Acknowledging message %s", delivery_tag)
        self.channel.basic_ack(delivery_tag)

    def start_consuming(self):
        self.channel.basic_consume(self.queue, self.on_message)

    def stop(self):
        LOGGER.debug("stopping")
