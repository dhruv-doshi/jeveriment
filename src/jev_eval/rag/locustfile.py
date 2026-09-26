"""Optional Locust workload against the internal RAG service.

Set RAG_LOAD_CASES to a JSON list of AnswerRequest payloads and SERVICE_API_KEY.
Use only after the owner authorizes a real provider load run.
"""

import json
import os
import random
from pathlib import Path

from locust import HttpUser, between, task


class RagUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        path = os.environ.get("RAG_LOAD_CASES")
        if not path:
            raise ValueError("RAG_LOAD_CASES must point to a prepared JSON workload")
        self.cases = json.loads(Path(path).read_text())
        if not self.cases:
            raise ValueError("RAG_LOAD_CASES is empty")
        self.headers = {"x-api-key": os.environ["SERVICE_API_KEY"]}

    @task
    def answer(self):
        self.client.post("/v1/answer", json=random.choice(self.cases), headers=self.headers,
                         name="/v1/answer")
