#!/usr/bin/env python3
"""
Quick test to verify streaming is working.
Run this after starting the server with: python -m smart_interviewer.main
"""
import requests
import json

BASE_URL = "http://localhost:8000"
SESSION_ID = "test-streaming-123"

def test_streaming_start():
    """Test streaming question generation on start."""
    print("Testing /v1/interview/start/stream")
    print("-" * 60)

    response = requests.post(
        f"{BASE_URL}/v1/interview/start/stream",
        headers={"X-Session-Id": SESSION_ID},
        stream=True
    )

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return

    question_tokens = []
    final_state = None

    for line in response.iter_lines():
        if not line:
            continue

        data = json.loads(line)
        event_type = data.get("type")

        if event_type == "question_token":
            token = data.get("token", "")
            question_tokens.append(token)
            print(token, end="", flush=True)

        elif event_type == "final_state":
            final_state = data.get("data", {})
            print("\n")

    print("-" * 60)
    print(f"✓ Received {len(question_tokens)} question tokens")
    print(f"✓ Final question: {final_state.get('current_question', 'N/A')}")
    print(f"✓ Phase: {final_state.get('phase', 'N/A')}")
    print()

def test_health():
    """Test server health."""
    response = requests.get(f"{BASE_URL}/")
    print(f"Server health: {response.json()}")
    print()

if __name__ == "__main__":
    print("=" * 60)
    print("STREAMING TEST")
    print("=" * 60)
    print()

    try:
        test_health()
        test_streaming_start()
        print("✓ All tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
