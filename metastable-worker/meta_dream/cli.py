#!/usr/bin/env python

import os
import logging


import click

import config

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("pika").setLevel(logging.ERROR)
logger = logging.getLogger("meta_dream")


def model_cache_dir():
    return os.path.join(os.getcwd(), "model_cache")


@click.group()
def cli():
    pass


@click.command()
@click.option("--host", default="0.0.0.0", help="Local address to bind to.")
@click.option("--port", default=5005, help="Local port to bind to.")
def server(host, port):
    from meta_dream.server import app

    app.run(host=host, port=port)


@click.command()
@click.option(
    "--hugging-face-token",
    envvar="HUGGINGFACE_TOKEN",
    help="Hugging Face user access token.",
)
@click.option(
    "--model-cache-dir",
    envvar="MODEL_CACHE_DIR",
    default=model_cache_dir(),
    help="The directory where the models are cached.",
)
def worker(hugging_face_token, model_cache_dir):
    from meta_dream.stable_diffusion import StableDiffusionPipeline
    from meta_dream.worker import Worker

    worker = Worker(config.AMQP_URL, hugging_face_token, model_cache_dir)

    while True:
        try:
            worker.run()
        except KeyboardInterrupt:
            worker.stop()
            break


cli.add_command(server)
cli.add_command(worker)

if __name__ == "__main__":
    cli()
