"""Config loading and logging setup."""

import logging
import sys

import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(cfg):
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cfg["logging"]["file"], mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    return logging.getLogger("litkg")
