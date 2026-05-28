import os
import random

import nltk

from vastai import (
    Worker,
    WorkerConfig,
    HandlerConfig,
    LogActionConfig,
    BenchmarkConfig,
)

# --- Model configuration ------------------------------------------------------

MODEL_SERVER_URL  = "http://127.0.0.1"
MODEL_SERVER_PORT = 1800
MODEL_LOG_FILE    = "/var/log/portal/sglang.log"
MODEL_HEALTHCHECK_ENDPOINT = "/health"

# vLLM-specific log messages
MODEL_LOAD_LOG_MSG = [
    "The server is fired up and ready to roll!",
]

MODEL_ERROR_LOG_MSGS = [
    "INFO:     Application shutdown complete.",
    "INFO: Application shutdown complete.",
    "Traceback (most recent call last):",
    "Initialization failed. warmup error: Traceback (most recent call last):",
    "Waiting for application shutdown."
]

MODEL_INFO_LOG_MSGS = [
        "Init torch distributed begin."
]

# --- Benchmark data generation -----------------------------------------------

# For this example we use NLTK's word list to create random prompts
nltk.download("words")
WORD_LIST = nltk.corpus.words.words()

def completions_benchmark_generator() -> dict:
    """Generate one benchmark payload for the /v1/completions endpoint.
    This shape should match what your vLLM server expects.
    """
    prompt = " ".join(random.choices(WORD_LIST, k=int(250)))

    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    return {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 500,
    }

# --- Worker configuration -----------------------------------------------------

worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        # /v1/completions: also used as the benchmark handler
        HandlerConfig(
            route="/v1/completions",

            # Allow vLLM to schedule parallel requests internally
            allow_parallel_requests=True,

            # Maximum time a request may sit in any internal queue before being rejected
            max_queue_time=60.0,

            # Workload: use max_tokens as a simple cost proxy
            workload_calculator=lambda payload: float(payload.get("max_tokens", 0)),

            benchmark_config=BenchmarkConfig(
                # Use our generator to produce payloads
                generator=completions_benchmark_generator,
                runs=8,
                concurrency=10,
            ),
        ),

        # /v1/chat/completions: similar behavior but no benchmark_config
        HandlerConfig(
            route="/v1/chat/completions",
            allow_parallel_requests=True,
            max_queue_time=60.0,
            workload_calculator=lambda payload: float(payload.get("max_tokens", 0)),
        ),
    ],

    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS,
    ),
)

# Run the worker synchronously
Worker(worker_config).run()

# Or run asynchronously if you need to do other Python work:
# import asyncio
# asyncio.run(Worker(worker_config).run_async())
