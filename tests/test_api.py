def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert body["models_loaded"] == 7


def test_home_page_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"<html" in res.data.lower()


def test_login_success_does_not_leak_password(client):
    res = client.post("/login", json={"username": "admin", "password": "1234"})
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert "login_details" not in body
    assert "1234" not in res.get_data(as_text=True)


def test_login_failure(client):
    res = client.post("/login", json={"username": "admin", "password": "wrong"})
    assert res.status_code == 401
    assert res.get_json()["success"] is False


def test_predict_requires_login(client):
    res = client.post("/predict", json={"answers": [3] * 15})
    assert res.status_code == 401


def test_predict_rejects_wrong_answer_count(logged_in_client):
    res = logged_in_client.post("/predict", json={"answers": [3] * 5})
    assert res.status_code == 400


def test_predict_happy_path(logged_in_client):
    res = logged_in_client.post("/predict", json={"answers": [3] * 15})
    assert res.status_code == 200
    body = res.get_json()
    for key in ["confidence", "discipline", "leadership", "neuroticism",
                "openness", "agreeableness", "extroversion"]:
        assert 0 <= body[key] <= 100
    assert body["total_points"] == 45
    assert body["attempt_number"] == 1
    assert len(body["changes_needed"]) == 3


def test_predict_accepts_free_text_answers(logged_in_client):
    answers = [{"type": "other", "text": "I would lead the team and help calmly"}] + [3] * 14
    res = logged_in_client.post("/predict", json={"answers": answers})
    assert res.status_code == 200


def test_attempt_counter_increments(logged_in_client):
    logged_in_client.post("/predict", json={"answers": [3] * 15})
    res = logged_in_client.post("/predict", json={"answers": [4] * 15})
    assert res.get_json()["attempt_number"] == 2


def test_logout_clears_session(logged_in_client):
    logged_in_client.post("/logout")
    res = logged_in_client.post("/predict", json={"answers": [3] * 15})
    assert res.status_code == 401
