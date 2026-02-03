"""Centralized logging configuration for ml-service."""

import logging

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

log = logging.getLogger("alphazero")
