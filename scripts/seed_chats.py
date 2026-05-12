import httpx, time, random

BACKEND = "http://localhost:8080"

# These are realistic support queries a chatbot would receive
conversations = [
    ["How do I reset my password?", "I tried but the email never arrived"],
    ["Where is my order?", "It says delivered but I got nothing"],
    ["I want to cancel my subscription"],
    ["My account is locked, I can't get in"],
    ["What are your business hours?"],
    ["I need a refund for order #4421"],
    ["How do I update my payment method?", "The form keeps showing an error"],
    ["I can't log in at all", "Reset didn't work either"],
    ["What is your return policy?"],
    ["My payment was declined but my card is fine"],
    ["I never received my confirmation email"],
    ["How do I change my delivery address?"],
    ["The app keeps crashing when I open it"],
    ["I was charged twice for the same order"],
    ["Can I change my subscription plan?"],
]

for i in range(40):
    session_id = None
    convo = random.choice(conversations)

    for message in convo:
        try:
            r = httpx.post(
                f"{BACKEND}/chat",
                json={"query": message, "session_id": session_id, "stream": False},
                timeout=90,
            )
            if r.status_code == 200:
                session_id = r.json().get("session_id")
                print(f"  [{i+1}] {message[:50]}")
        except Exception as e:
            print(f"  Error: {e}")

        time.sleep(2)   # stay within Gemini free tier RPM

print("Done")