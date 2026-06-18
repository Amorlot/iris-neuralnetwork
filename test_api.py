"""
Test della pipeline API Iris.
Avviare prima il server con:  python api/app.py

Gli endpoint devono essere chiamati in quest'ordine:
  1. POST /data/load
  2. POST /model/configure
  3. POST /model/build
  4. POST /model/train
  5. POST /predict  (ripetibile)
"""

import sys
import requests

BASE_URL = "http://localhost:5000"

_passed = 0
_failed = 0


def run(name, fn):
    global _passed, _failed
    try:
        fn()
        print(f"[PASS] {name}")
        _passed += 1
    except AssertionError as e:
        print(f"[FAIL] {name}: {e}")
        _failed += 1


# ── Utility ────────────────────────────────────────────────────────────────────

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_status_idle():
    r = requests.get(f"{BASE_URL}/status")
    assert r.status_code == 200
    assert r.json()["step"] == "idle"


# ── Out-of-order (chiamate anticipate devono restituire 409) ───────────────────

def test_configure_before_load():
    r = requests.post(f"{BASE_URL}/model/configure", json={"optimizer": "adam"})
    assert r.status_code == 409, f"expected 409, got {r.status_code}"


def test_build_before_load():
    r = requests.post(f"{BASE_URL}/model/build")
    assert r.status_code == 409


def test_train_before_load():
    r = requests.post(f"{BASE_URL}/model/train")
    assert r.status_code == 409


def test_predict_before_train():
    r = requests.post(f"{BASE_URL}/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 409


# ── Step 1: /data/load ─────────────────────────────────────────────────────────

def test_load_data():
    r = requests.post(f"{BASE_URL}/data/load")
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == "data_loaded"
    assert body["train_samples"] == 120
    assert body["test_samples"] == 30


def test_load_data_twice():
    r = requests.post(f"{BASE_URL}/data/load")
    assert r.status_code == 409


# ── Step 2: /model/configure ───────────────────────────────────────────────────

def test_configure_invalid_optimizer():
    r = requests.post(f"{BASE_URL}/model/configure", json={"optimizer": "rmsprop"})
    assert r.status_code == 400


def test_configure():
    r = requests.post(f"{BASE_URL}/model/configure", json={
        "optimizer": "adam",
        "learning_rate": 0.001,
        "epochs": 80,
        "batch_size": 16,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == "configured"
    assert body["optimizer"] == "adam"


def test_configure_twice():
    r = requests.post(f"{BASE_URL}/model/configure", json={"optimizer": "sgd"})
    assert r.status_code == 409


# ── Step 3: /model/build ───────────────────────────────────────────────────────

def test_build():
    r = requests.post(f"{BASE_URL}/model/build")
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == "built"
    assert body["total_params"] > 0


def test_build_twice():
    r = requests.post(f"{BASE_URL}/model/build")
    assert r.status_code == 409


# ── Step 4: /model/train ───────────────────────────────────────────────────────

def test_train():
    r = requests.post(f"{BASE_URL}/model/train", timeout=120)
    assert r.status_code == 200
    body = r.json()
    assert body["step"] == "trained"
    assert body["test_accuracy"] > 0.8, f"accuracy too low: {body['test_accuracy']}"


def test_train_twice():
    r = requests.post(f"{BASE_URL}/model/train", timeout=120)
    assert r.status_code == 409


# ── Step 5: /predict ───────────────────────────────────────────────────────────

def test_predict_setosa():
    r = requests.post(f"{BASE_URL}/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == "setosa"


def test_predict_versicolor():
    r = requests.post(f"{BASE_URL}/predict", json={"features": [6.0, 2.7, 4.1, 1.0]})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == "versicolor"


def test_predict_virginica():
    r = requests.post(f"{BASE_URL}/predict", json={"features": [6.9, 3.1, 5.4, 2.1]})
    assert r.status_code == 200
    assert r.json()["predicted_class"] == "virginica"


def test_predict_missing_features():
    r = requests.post(f"{BASE_URL}/predict", json={})
    assert r.status_code == 400


def test_predict_wrong_count():
    r = requests.post(f"{BASE_URL}/predict", json={"features": [1.0, 2.0]})
    assert r.status_code == 400


def test_predict_no_body():
    r = requests.post(f"{BASE_URL}/predict")
    assert r.status_code == 400


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Testing API at {BASE_URL}")
    print("Pipeline: /data/load → /model/configure → /model/build → /model/train → /predict\n")

    run("GET  /health", test_health)
    run("GET  /status (step=idle)", test_status_idle)

    print("\n-- Out-of-order (409 atteso) --")
    run("POST /model/configure  prima di /data/load", test_configure_before_load)
    run("POST /model/build      prima di /data/load", test_build_before_load)
    run("POST /model/train      prima di /data/load", test_train_before_load)
    run("POST /predict          prima di /model/train", test_predict_before_train)

    print("\n-- Step 1: /data/load --")
    run("POST /data/load", test_load_data)
    run("POST /data/load  (secondo tentativo → 409)", test_load_data_twice)

    print("\n-- Step 2: /model/configure --")
    run("POST /model/configure  optimizer sconosciuto → 400", test_configure_invalid_optimizer)
    run("POST /model/configure  optimizer=adam", test_configure)
    run("POST /model/configure  (secondo tentativo → 409)", test_configure_twice)

    print("\n-- Step 3: /model/build --")
    run("POST /model/build", test_build)
    run("POST /model/build  (secondo tentativo → 409)", test_build_twice)

    print("\n-- Step 4: /model/train (attendi ~30s) --")
    run("POST /model/train", test_train)
    run("POST /model/train  (secondo tentativo → 409)", test_train_twice)

    print("\n-- Step 5: /predict --")
    run("POST /predict  → setosa", test_predict_setosa)
    run("POST /predict  → versicolor", test_predict_versicolor)
    run("POST /predict  → virginica", test_predict_virginica)
    run("POST /predict  features mancanti → 400", test_predict_missing_features)
    run("POST /predict  features errate → 400", test_predict_wrong_count)
    run("POST /predict  body assente → 400", test_predict_no_body)

    print(f"\n{_passed} passed, {_failed} failed")
    sys.exit(0 if _failed == 0 else 1)


if __name__ == "__main__":
    main()
