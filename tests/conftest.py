import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    return flask_app.test_client()


@pytest.fixture
def logged_in_client(client):
    res = client.post("/login", json={"username": "admin", "password": "1234"})
    assert res.status_code == 200
    return client
