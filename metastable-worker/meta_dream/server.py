#!/usr/bin/env python

from flask import Flask, request
import pika
import json

from meta_dream.stable_diffusion import StableDiffusionPipeline
import config

app = Flask(__name__)

connection = pika.BlockingConnection(pika.URLParameters(config.AMQP_URL))

channel = connection.channel()
assert channel

EXCHANGE = "k_lms.stable_diffusion_v1_4.meta-dream.labs.rwx.im"

channel.exchange_declare(exchange=EXCHANGE, exchange_type="topic")

for scheduler in ["k_lms", "pndm", "ddim"]:
    result = channel.queue_declare(queue=f"{scheduler}.{EXCHANGE}")
    queue_name = result.method.queue
    channel.queue_bind(exchange=EXCHANGE, queue=queue_name, routing_key="prompt")


def valid_dream(json):
    if not "prompt" in json:
        return False

    return True


@app.route("/")
def index():
    return ""


def job_not_found(job_id):
    return ({"error": "the job was not found"}, 404)


def job_to_dict(job):
    return {
        "id": job.id,
        "status": job.get_status(),
        "result": job.result,
        "enqueued_at": job.enqueued_at,
        "started_at": job.started_at,
        "ended_at": job.ended_at,
    }


def worker_to_dict(worker):
    return {
        "name": worker.name,
        "hostname": worker.hostname,
        "pid": worker.pid,
        "queues": [queue.name for queue in worker.queues],
        "state": worker.state,
        "current_job_id": worker.get_current_job_id(),
        "last_heartbeat": worker.last_heartbeat,
        "birth_date": worker.birth_date,
        "successful_job_count": worker.successful_job_count,
        "failed_job_count": worker.failed_job_count,
        "total_working_time": worker.total_working_time,
    }


@app.route("/api/v1/jobs")
def list_jobs():
    return ([], 200)


@app.route("/api/v1/jobs/finished")
def list_finished_jobs():
    return ([], 200)


@app.route("/api/v1/jobs/started")
def list_started_jobs():
    return ([], 200)


@app.route("/api/v1/jobs/failed")
def list_failed_jobs():
    return ([], 200)


@app.route("/api/v1/jobs/canceled")
def list_canceled_jobs():
    return ([], 200)


@app.route("/api/v1/jobs/<job_id>")
def get_job(job_id):
    return job_not_found(job_id)


@app.route("/api/v1/workers")
def list_workers():
    return ([], 200)


@app.route("/api/v1/dream", methods=["POST"])
def post_dream():
    if not request.is_json:
        return ({"message": "invalid json"}, 400)

    request_json = request.get_json()

    if not request_json:
        return ({"error": "invalid request"}, 422)

    print(request_json)

    if valid_dream(request_json):
        prompt = request_json["prompt"]
        seed = int(request_json.get("seed", 1))
        message = {"prompt": prompt, "seed": seed}
        channel.basic_publish(
            exchange=EXCHANGE,
            routing_key="prompt",
            body=json.dumps(message),
            properties=pika.BasicProperties(
                delivery_mode=pika.spec.PERSISTENT_DELIVERY_MODE
            ),
        )

        return ({}, 201)

    return ({"error": "invalid request"}, 422)
