#!/usr/bin/env python3
"""
Simple streaming test - run this while dev-interview is running.
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"
SESSION_ID = f"test-{int(time.time())}"

print("=" * 70)
print("STREAMING TEST")
print("=" * 70)

# Step 1: Check server health
print("\n1. Checking server health...")
try:
    r = requests.get(f"{BASE_URL}/")
    print(f"   ✓ Server is up: {r.json()}")
except Exception as e:
    print(f"   ✗ Server not responding: {e}")
    print("   Make sure to run: dev-interview")
    exit(1)

# Step 2: Initialize session (this will use non-streaming)
print(f"\n2. Initializing session: {SESSION_ID}")
try:
    r = requests.get(
        f"{BASE_URL}/v1/session/state",
        headers={"X-Session-Id": SESSION_ID}
    )
    print(f"   ✓ Session initialized: phase={r.json().get('phase')}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    exit(1)

# Step 3: Test streaming start
print("\n3. Testing streaming start endpoint...")
print("   Watch the server logs for: ✨ STREAMING")
print("-" * 70)

try:
    r = requests.post(
        f"{BASE_URL}/v1/interview/start/stream",
        headers={"X-Session-Id": SESSION_ID},
        stream=True,
        timeout=30
    )

    if r.status_code != 200:
        print(f"   ✗ Error {r.status_code}: {r.text}")
        exit(1)

    token_count = 0
    question_text = ""

    print("   Receiving stream:")
    for line in r.iter_lines():
        if not line:
            continue

        data = json.loads(line)
        event_type = data.get("type")

        if event_type == "question_token":
            token = data.get("token", "")
            question_text += token
            token_count += 1
            print(token, end="", flush=True)

        elif event_type == "final_state":
            print("\n")
            final_state = data.get("data", {})
            print(f"\n   ✓ Stream complete!")
            print(f"   ✓ Received {token_count} tokens")
            print(f"   ✓ Final question: {final_state.get('current_question', 'N/A')[:60]}...")
            print(f"   ✓ Phase: {final_state.get('phase')}")

            if token_count > 0:
                print("\n" + "=" * 70)
                print("SUCCESS! Streaming is working correctly! ✨")
                print("=" * 70)
            else:
                print("\n" + "=" * 70)
                print("WARNING: No tokens received. Check server logs for errors.")
                print("=" * 70)

    if token_count == 0:
        print("\n   ⚠️  No streaming tokens received.")
        print("   Check the server logs - you should see:")
        print("      ✨ STREAMING question for turn 1, level 1")
        print("   If you see:")
        print("      ⚡ NON-STREAMING for turn 1, level 1")
        print("   Then the writer is not being passed to the node.")

except Exception as e:
    print(f"\n   ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
