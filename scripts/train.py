"""Train, tune, evaluate, and track the fraud ensemble."""

from __future__ import annotations

import json

from fraud_detection.training import train_and_evaluate

if __name__ == "__main__":
    print(json.dumps(train_and_evaluate(), indent=2))
