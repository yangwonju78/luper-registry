import json
import os
import unittest
from unittest.mock import patch

from flask import Flask

from revu_collector import revu_bp


class FakeHeaders:
    def get_content_type(self):
        return "application/json"


class FakeResponse:
    status = 200
    headers = FakeHeaders()

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RevuCollectorTest(unittest.TestCase):
    def setUp(self):
        self.previous_key = os.environ.get("REVU_COLLECTOR_KEY")
        os.environ["REVU_COLLECTOR_KEY"] = "test-secret"
        app = Flask(__name__)
        app.register_blueprint(revu_bp)
        self.client = app.test_client()

    def tearDown(self):
        if self.previous_key is None:
            os.environ.pop("REVU_COLLECTOR_KEY", None)
        else:
            os.environ["REVU_COLLECTOR_KEY"] = self.previous_key

    def test_health_reports_stateless_mode(self):
        response = self.client.get("/api/revu/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["mode"], "stateless")

    def test_collection_requires_key(self):
        response = self.client.post("/api/revu/collect/baemin", json={})
        self.assertEqual(response.status_code, 401)

    def test_rejects_non_baemin_url(self):
        response = self.client.post(
            "/api/revu/collect/baemin",
            headers={"X-Revu-Key": "test-secret"},
            json={"url": "https://example.com/private"},
        )
        self.assertEqual(response.status_code, 400)

    @patch("revu_collector._open_no_redirect")
    def test_collects_and_deduplicates_pages(self, mocked_open):
        mocked_open.side_effect = [
            FakeResponse({"reviews": [{"id": 1}, {"id": 2}], "next": True}),
            FakeResponse({"reviews": [{"id": 2}, {"id": 3}], "next": False}),
        ]
        response = self.client.post(
            "/api/revu/collect/baemin",
            headers={"X-Revu-Key": "test-secret"},
            json={
                "url": "https://self-api.baemin.com/v1/review/shops/123/reviews?from=2026-01-01&to=2026-09-22",
                "headers": {"Authorization": "Bearer temporary", "Host": "evil.example"},
                "pageSize": 2,
            },
        )
        result = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(result["count"], 3)
        self.assertEqual(result["pages"], 2)
        self.assertFalse(result["persisted"])
        second_url = mocked_open.call_args_list[1].args[0].full_url
        self.assertIn("offset=2", second_url)
        sent_headers = dict(mocked_open.call_args_list[0].args[0].header_items())
        self.assertNotIn("Host", sent_headers)


if __name__ == "__main__":
    unittest.main()
