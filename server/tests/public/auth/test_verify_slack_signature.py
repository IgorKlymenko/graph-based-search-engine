from src.security.slack import slack_sign
import time

def test_verify_signature_valid(client):
    SIGNING_SECRET = b"mysecret"
    body = b'{"type":"url_verification","challenge":"abc"}'
    ts, sig = slack_sign(body, SIGNING_SECRET)

    HEADERS = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": ts, 
        "X-Slack-Signature": sig
    }

    response = client.post("/slack/event", 
                            content=body, 
                            headers=HEADERS)
    assert response.status_code == 200

def test_verify_signature_invalid(client):
    SIGNING_SECRET = b"mysecret"
    body = b'{"type":"url_verification","challenge":"abc"}'
    ts, sig = slack_sign(body, SIGNING_SECRET)

    malicious_body = b'{"type":"url_verification","challenge":"abcd"}'

    HEADERS = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": ts, 
        "X-Slack-Signature": sig
    }

    response = client.post("/slack/event", 
                            content=malicious_body, 
                            headers=HEADERS)
    assert response.status_code == 401

def test_verify_missing_signature(client):
    SIGNING_SECRET = b"mysecret"
    body = b'{"type":"url_verification","challenge":"abc"}'
    ts, sig = slack_sign(body, SIGNING_SECRET, "bad_timestamp")

    HEADERS = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": ts
    }

    response = client.post("/slack/event", 
                            content=body, 
                            headers=HEADERS)
    assert response.status_code == 401

def test_verify_timestamp_invalid(client):
    SIGNING_SECRET = b"mysecret"
    body = b'{"type":"url_verification","challenge":"abc"}'
    ts, sig = slack_sign(body, SIGNING_SECRET, "bad_timestamp")

    HEADERS = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": ts, 
        "X-Slack-Signature": sig
    }

    response = client.post("/slack/event", 
                            content=body, 
                            headers=HEADERS)
    assert response.status_code == 401

def test_verify_timestamp_stale(client):
    SIGNING_SECRET = b"mysecret"
    body = b'{"type":"url_verification","challenge":"abc"}'

    stale_ts = str(time.time() - (60 * 5 + 1))
    ts, sig = slack_sign(body, SIGNING_SECRET, stale_ts)

    HEADERS = {
        "Content-Type": "application/json",
        "X-Slack-Request-Timestamp": ts, 
        "X-Slack-Signature": sig
    }

    response = client.post("/slack/event", 
                            content=body, 
                            headers=HEADERS)
    assert response.status_code == 401


def test_verify_missing_timestamp(client):
    SIGNING_SECRET = b"mysecret"
    body = b'{"type":"url_verification","challenge":"abc"}'
    ts, sig = slack_sign(body, SIGNING_SECRET, "bad_timestamp")

    HEADERS = {
        "Content-Type": "application/json",
        "X-Slack-Signature": sig
    }

    response = client.post("/slack/event", 
                            content=body, 
                            headers=HEADERS)
    assert response.status_code == 401

