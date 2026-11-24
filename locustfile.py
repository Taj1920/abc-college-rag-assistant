import random

from locust import HttpUser, task, between

class SmartCampusUser(HttpUser):
    # Simulate a student waiting 1–3 seconds between questions
    wait_time = between(1, 3)

    questions = [
        "What is the examination fee for 2026-2027?",
        "What courses are offered by ABC College?",
        "When are the upcoming examinations?",
        "What is the admission process?",
        "Tell me about the college facilities.",
    ]

    @task
    def ask_question(self):
        question = random.choice(self.questions)

        with self.client.post(
            "/api/v1/chat",
            json={"question": question},
            name="/api/v1/chat",
            timeout=60,
            catch_response=True,
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}: {response.text[:200]}"
                )
                return

            try:
                data = response.json()

                if not data.get("answer"):
                    response.failure("Response has no answer")
                elif not isinstance(data.get("sources"), list):
                    response.failure("Invalid sources format")
                else:
                    response.success()

            except ValueError:
                response.failure("Invalid JSON response")
