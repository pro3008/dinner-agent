#!/usr/bin/env python3
"""
app.py — WhatsApp webhook for the Dinner Planning Agent
=========================================================
Receives WhatsApp messages via Twilio, processes them with Claude,
and sends back responses. Deploy to Railway or Render.

Uses background threading so WhatsApp doesn't time out while
Claude is thinking (Claude can take 20-30 seconds for a full menu).
"""

import os
import threading
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from agent import chat, load_data, save_data

app = Flask(__name__)

# ── Config from environment variables ─────────────────────
TWILIO_AUTH_TOKEN  = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_ACCOUNT_SID = os.environ.get("TWILIO_ACCOUNT_SID", "")

AUTHORIZED_NUMBERS_RAW = os.environ.get("AUTHORIZED_NUMBERS", "")
AUTHORIZED_NUMBERS = [n.strip() for n in AUTHORIZED_NUMBERS_RAW.split(",") if n.strip()]


# ── Helpers ───────────────────────────────────────────────

def split_message(text: str, max_len: int = 1500) -> list[str]:
    """Split a long message into WhatsApp-safe chunks."""
    if len(text) <= max_len:
        return [text]
    chunks = []
    paragraphs = text.split("\n\n")
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > max_len:
            if current:
                chunks.append(current.strip())
            current = para
        else:
            current += ("\n\n" if current else "") + para
    if current:
        chunks.append(current.strip())
    return chunks


def process_and_reply(sender: str, to_number: str, message: str):
    """
    Process message in background thread and send reply via Twilio REST API.
    This runs AFTER the webhook has already returned to WhatsApp,
    so there's no timeout issue.
    """
    twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    try:
        data = load_data()
        response_text = chat(message, sender, data)
        save_data(data)
    except Exception as e:
        response_text = f"Oops, something went wrong 😅 Please try again.\n\n_{str(e)[:120]}_"

    # Send the reply back via Twilio REST API
    chunks = split_message(response_text)
    for chunk in chunks:
        try:
            twilio_client.messages.create(
                from_=to_number,   # The Twilio sandbox number
                to=sender,         # The user's WhatsApp number
                body=chunk
            )
        except Exception as e:
            print(f"Failed to send chunk: {e}")


# ── Routes ────────────────────────────────────────────────

@app.route("/health")
def health():
    return {"status": "ok", "service": "Dinner Planning Agent"}, 200


@app.route("/webhook", methods=["POST"])
def webhook():
    sender     = request.form.get("From", "")   # e.g. "whatsapp:+358XXXXXXXXX"
    message    = request.form.get("Body", "").strip()
    to_number  = request.form.get("To", "")     # The Twilio sandbox number

    # Authorisation check
    if AUTHORIZED_NUMBERS and sender not in AUTHORIZED_NUMBERS:
        resp = MessagingResponse()
        resp.message("Sorry, this is a private family bot 🔒")
        return str(resp), 200

    if not message:
        return str(MessagingResponse()), 200

    # Send immediate acknowledgment so WhatsApp doesn't time out
    thread = threading.Thread(
        target=process_and_reply,
        args=(sender, to_number, message)
    )
    thread.daemon = True
    thread.start()

    # Return empty response immediately — reply comes separately
    return str(MessagingResponse()), 200


# ── Run locally for testing ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🍽️  Dinner Agent webhook running on port {port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
