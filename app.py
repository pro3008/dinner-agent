#!/usr/bin/env python3
"""
app.py — WhatsApp webhook for the Dinner Planning Agent
=========================================================
Receives WhatsApp messages via Twilio, processes them with Claude,
and sends back responses. Deploy to Railway or Render.
"""

import os
from flask import Flask, request, abort
from twilio.twiml.messaging_response import MessagingResponse
from twilio.request_validator import RequestValidator
from agent import chat, load_data, save_data

app = Flask(__name__)

# ── Config from environment variables ─────────────────────
TWILIO_AUTH_TOKEN   = os.environ.get("TWILIO_AUTH_TOKEN", "")
TWILIO_ACCOUNT_SID  = os.environ.get("TWILIO_ACCOUNT_SID", "")

# Comma-separated list of authorised WhatsApp numbers
# Format: "whatsapp:+358XXXXXXXXX,whatsapp:+358YYYYYYYYY"
AUTHORIZED_NUMBERS_RAW = os.environ.get("AUTHORIZED_NUMBERS", "")
AUTHORIZED_NUMBERS = [n.strip() for n in AUTHORIZED_NUMBERS_RAW.split(",") if n.strip()]

WHATSAPP_SANDBOX = os.environ.get("WHATSAPP_SANDBOX", "true").lower() == "true"


# ── Helpers ───────────────────────────────────────────────

def split_message(text: str, max_len: int = 1500) -> list[str]:
    """Split a long message into WhatsApp-safe chunks."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    # Try to split on double newlines (sections)
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


def validate_twilio_request() -> bool:
    """Validate that the request genuinely comes from Twilio."""
    if WHATSAPP_SANDBOX or not TWILIO_AUTH_TOKEN:
        return True   # Skip validation in sandbox / dev mode

    validator = RequestValidator(TWILIO_AUTH_TOKEN)
    signature = request.headers.get("X-Twilio-Signature", "")
    url = request.url
    params = request.form.to_dict()
    return validator.validate(url, params, signature)


# ── Routes ────────────────────────────────────────────────

@app.route("/health")
def health():
    return {"status": "ok", "service": "Dinner Planning Agent"}, 200


@app.route("/webhook", methods=["POST"])
def webhook():
    # Security check
    if not validate_twilio_request():
        abort(403)

    sender  = request.form.get("From", "")    # e.g. "whatsapp:+358XXXXXXXXX"
    message = request.form.get("Body", "").strip()

    # Authorisation check (skip if list is empty = allow all)
    if AUTHORIZED_NUMBERS and sender not in AUTHORIZED_NUMBERS:
        resp = MessagingResponse()
        resp.message("Sorry, this is a private family bot 🔒")
        return str(resp), 200

    if not message:
        return str(MessagingResponse()), 200

    # Process the message
    data = load_data()
    try:
        response_text = chat(message, sender, data)
    except Exception as e:
        response_text = f"Oops, something went wrong 😅 Please try again.\n\n_Error: {str(e)[:100]}_"

    save_data(data)

    # Build Twilio response (handle long messages by splitting)
    resp = MessagingResponse()
    chunks = split_message(response_text)
    for chunk in chunks:
        resp.message(chunk)

    return str(resp), 200


# ── Run locally for testing ───────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🍽️  Dinner Agent webhook running on port {port}")
    print("Use ngrok or deploy to Railway to expose it to Twilio.\n")
    app.run(host="0.0.0.0", port=port, debug=True)
