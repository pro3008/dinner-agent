#!/usr/bin/env python3
"""
agent.py — Core dinner planning agent with learning and memory
==============================================================
Handles: plan generation, day swapping, pantry tracking,
         shopping lists, and learning family preferences over time.
"""

import anthropic
import json
import os
from datetime import datetime
from pathlib import Path

# ── Config ────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "your-key-here")
DATA_FILE = os.environ.get("DATA_FILE", "data.json")

DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

FAMILY_CONTEXT = """Family profile — Finnish-Indian family of 4 in Helsinki, Finland:
- 2 adults who enjoy cooking and trying cuisines from both cultures
- 2 toddlers aged 2 and 4 (need mild seasoning, toddler-safe textures)
- NO meat: no chicken, beef, or pork
- Fish and seafood: loved by everyone — include 2-3 times per week
- Eggs: loved by everyone — great for quick weeknight meals
- Dairy: cheese, paneer, butter, cream — all fine

Cuisine preferences:
- Indian: dal (lentils), rice dishes, coconut curry, palak paneer, aloo dishes
- Mexican: fish tacos, egg tacos, bean quesadillas, guacamole
- Finnish: lohikeitto (salmon soup), oven-baked fish, potato dishes
- Finnish-Indian fusion is warmly encouraged!

Practical notes:
- Groceries available at S-Market, K-Market, Prisma, Lidl in Helsinki
- Weeknight dinners (Mon-Fri): max 40 minutes total
- Weekend dinners: up to 90 minutes ok, can be more elaborate"""

# ── Tool definitions ──────────────────────────────────────
TOOLS = [
    {
        "name": "generate_plan",
        "description": "Generate a complete 7-day dinner plan, incorporating pantry items and learned family preferences.",
        "input_schema": {
            "type": "object",
            "properties": {
                "special_requests": {
                    "type": "string",
                    "description": "Any specific requests for this week (e.g. 'include something Finnish', 'quick meals this week')"
                }
            },
            "required": []
        }
    },
    {
        "name": "swap_days",
        "description": "Move a specific cuisine or meal type to a different day, or swap two days with each other.",
        "input_schema": {
            "type": "object",
            "properties": {
                "day1": {"type": "string", "description": "The day to change", "enum": DAYS},
                "day2": {"type": "string", "description": "Swap with this day (optional)", "enum": DAYS},
                "new_meal_request": {"type": "string", "description": "What to put on day1 instead (e.g. 'Indian dal', 'quick fish dish')"}
            },
            "required": ["day1"]
        }
    },
    {
        "name": "update_pantry",
        "description": "Add or remove items from the family's pantry/fridge inventory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "add": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items to add (e.g. ['salmon fillet', 'coconut milk', 'eggs'])"
                },
                "remove": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Items to remove (already used up)"
                }
            }
        }
    },
    {
        "name": "get_shopping_list",
        "description": "Generate a categorized shopping list for the current week's plan, automatically excluding pantry items.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "record_feedback",
        "description": "Save feedback about a meal (loved it, kids hated it, too spicy, etc.) to improve future plans.",
        "input_schema": {
            "type": "object",
            "properties": {
                "meal_name": {"type": "string", "description": "Name of the meal"},
                "sentiment": {
                    "type": "string",
                    "enum": ["loved", "liked", "ok", "disliked", "kids_loved", "kids_disliked"],
                    "description": "How the family felt about it"
                },
                "notes": {"type": "string", "description": "Specific notes, e.g. 'too spicy for kids', 'will make again'"}
            },
            "required": ["meal_name", "sentiment"]
        }
    },
    {
        "name": "show_plan",
        "description": "Show the current weekly plan in a readable format.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    }
]

# ── Data persistence ──────────────────────────────────────

EMPTY_DATA = {
    "plan": {},
    "pantry": [],
    "preferences": {
        "loved_meals": [],
        "disliked_meals": [],
        "notes": [],
        "feedback_history": []
    },
    "conversations": {},
    "last_generated": None
}

def load_data() -> dict:
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            # File is corrupted — delete it and start fresh
            os.remove(DATA_FILE)
    return dict(EMPTY_DATA)


def save_data(data: dict) -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Plan generation ───────────────────────────────────────

def _build_preferences_context(prefs: dict) -> str:
    """Convert stored preferences into a context string for the prompt."""
    lines = []
    if prefs.get("loved_meals"):
        lines.append(f"FAMILY LOVED THESE (make again): {', '.join(prefs['loved_meals'][-10:])}")
    if prefs.get("disliked_meals"):
        lines.append(f"FAMILY DISLIKED THESE (avoid): {', '.join(prefs['disliked_meals'][-10:])}")
    if prefs.get("notes"):
        lines.append(f"IMPORTANT NOTES FROM FAMILY: {'; '.join(prefs['notes'][-5:])}")
    return "\n".join(lines) if lines else "No history yet — first week!"


def call_generate_plan(data: dict, special_requests: str = "") -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    pantry = data.get("pantry", [])
    pantry_str = ", ".join(pantry) if pantry else "nothing specific noted"
    prefs_context = _build_preferences_context(data.get("preferences", {}))
    week_str = datetime.now().strftime("%B %d, %Y")

    prompt = f"""Generate a 7-day dinner menu for this family. Week of {week_str}.

{FAMILY_CONTEXT}

PANTRY / FRIDGE RIGHT NOW (prioritize using these):
{pantry_str}

LEARNED FAMILY PREFERENCES:
{prefs_context}

SPECIAL REQUESTS THIS WEEK:
{special_requests if special_requests else "None"}

Return ONLY valid JSON (no other text, no markdown fences):
{{
  "monday":    {{"name": "...", "cuisine": "Indian/Mexican/Finnish/Finnish-Indian/Other", "description": "One warm sentence.", "prep_time": 30, "kid_tip": "Toddler-friendly tip.", "ingredients": ["..."], "uses_pantry": ["pantry items used"]}},
  "tuesday":   {{...}},
  "wednesday": {{...}},
  "thursday":  {{...}},
  "friday":    {{...}},
  "saturday":  {{...}},
  "sunday":    {{...}}
}}"""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
        if text.endswith("```"):
            text = text[:-3].strip()
    return json.loads(text)


def call_swap_days(data: dict, day1: str, day2: str = None, new_meal_request: str = None) -> dict:
    plan = data.get("plan", {})

    if day2 and day2 in plan and day1 in plan:
        plan[day1], plan[day2] = plan[day2], plan[day1]
        return plan

    if new_meal_request and day1:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        pantry_str = ", ".join(data.get("pantry", [])) or "none"
        prompt = f"""Generate ONE dinner for {day1.capitalize()} for this family:

{FAMILY_CONTEXT}

The meal should be: {new_meal_request}
Pantry available: {pantry_str}

Return ONLY valid JSON (no other text):
{{"name":"...","cuisine":"...","description":"...","prep_time":30,"kid_tip":"...","ingredients":["..."],"uses_pantry":[]}}"""
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
        plan[day1] = json.loads(text)
    return plan


def call_get_shopping_list(data: dict) -> dict:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    plan = data.get("plan", {})
    pantry = data.get("pantry", [])

    all_ingredients = []
    for day_data in plan.values():
        all_ingredients.extend(day_data.get("ingredients", []))

    pantry_str = ", ".join(pantry) if pantry else "none"
    ingredients_str = "\n".join(f"- {i}" for i in set(all_ingredients))

    prompt = f"""Organize a shopping list for a family in Helsinki, Finland.

ALL INGREDIENTS NEEDED:
{ingredients_str}

ALREADY IN PANTRY (do NOT include):
{pantry_str}

Return ONLY valid JSON (no other text):
{{
  "produce": ["..."],
  "fish_eggs_dairy": ["..."],
  "pantry_dry": ["..."],
  "frozen": ["..."],
  "spices_condiments": ["..."]
}}
Consolidate duplicates. Use realistic Finnish grocery quantities."""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:]).rstrip("`").strip()
    return json.loads(text)


# ── Format plan for WhatsApp ──────────────────────────────

EMOJI_MAP = {
    "Indian": "🇮🇳", "Mexican": "🌮", "Finnish": "🇫🇮",
    "Finnish-Indian": "🌟", "Italian": "🍝", "Other": "🍽️"
}

def format_plan_whatsapp(plan: dict) -> str:
    lines = ["*📅 This Week's Dinner Menu*\n"]
    for day in DAYS:
        if day not in plan:
            continue
        m = plan[day]
        emoji = EMOJI_MAP.get(m.get("cuisine", ""), "🍽️")
        lines.append(f"{emoji} *{day.capitalize()}* — {m['name']}")
        lines.append(f"   _{m.get('description', '')}_")
        lines.append(f"   ⏱ {m.get('prep_time', '?')}min  |  👶 {m.get('kid_tip', '')}")
        if m.get("uses_pantry"):
            lines.append(f"   ♻️ Uses: {', '.join(m['uses_pantry'])}")
        lines.append("")
    return "\n".join(lines)


def format_shopping_whatsapp(shopping: dict) -> str:
    sections = {
        "produce": "🥦 *Produce & Herbs*",
        "fish_eggs_dairy": "🐟 *Fish, Eggs & Dairy*",
        "pantry_dry": "🥫 *Pantry & Dry*",
        "frozen": "🧊 *Frozen*",
        "spices_condiments": "🌶 *Spices & Condiments*"
    }
    lines = ["*🛒 Shopping List*\n"]
    for key, label in sections.items():
        items = shopping.get(key, [])
        if items:
            lines.append(label)
            for item in items:
                lines.append(f"  ▢ {item}")
            lines.append("")
    lines.append("_Hyvää ruokahalua! 🇫🇮🇮🇳_")
    return "\n".join(lines)


# ── Helper: make Claude response blocks JSON-safe ─────────

def serialize_content(content):
    """Convert Anthropic SDK objects to plain dicts so they can be saved to JSON."""
    if not isinstance(content, list):
        return content
    result = []
    for block in content:
        if hasattr(block, "type"):
            if block.type == "text":
                result.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                result.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input
                })
        else:
            result.append(block)
    return result


# ── Main chat function ────────────────────────────────────

def chat(user_message: str, sender_number: str, data: dict) -> str:
    """
    Process a WhatsApp message and return a response.
    Updates data in-place (caller must save).
    """
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Per-user conversation history (for context), shared plan/pantry
    conversations = data.setdefault("conversations", {})
    history = conversations.get(sender_number, [])
    history.append({"role": "user", "content": user_message})

    # Build system prompt with current state
    plan_summary = "No plan yet."
    if data.get("plan"):
        plan_summary = "\n".join(
            f"- {d.capitalize()}: {v['name']}" for d, v in data["plan"].items()
        )

    pantry_summary = ", ".join(data.get("pantry", [])) or "Nothing noted"
    prefs = data.get("preferences", {})
    loved = ", ".join(prefs.get("loved_meals", [])[-5:]) or "None yet"
    disliked = ", ".join(prefs.get("disliked_meals", [])[-5:]) or "None yet"

    system = f"""You are a warm, practical family dinner planning assistant for a Finnish-Indian family in Helsinki. You communicate via WhatsApp.

CURRENT WEEK'S PLAN:
{plan_summary}

PANTRY / FRIDGE:
{pantry_summary}

MEALS FAMILY HAS LOVED: {loved}
MEALS FAMILY DISLIKED: {disliked}

You have tools to: generate_plan, swap_days, update_pantry, get_shopping_list, record_feedback, show_plan.

Rules:
- Be concise and warm — this is WhatsApp, not email
- Use *bold* and _italic_ for WhatsApp formatting
- When you use a tool that updates the plan, confirm it briefly
- When recording feedback, always confirm what you learned
- Both family members (adults) may message you — treat them as equal decision makers
- If asked to show the plan or shopping list, always call the appropriate tool"""

    response = client.messages.create(
        model="claude-opus-4-5-20251101",
        max_tokens=1500,
        system=system,
        tools=TOOLS,
        messages=history
    )

    # Handle tool calls in a loop
    result_text = ""
    while response.stop_reason == "tool_use":
        tool_results = []

        for block in response.content:
            if block.type == "tool_use":
                tool_name = block.name
                tool_input = block.input
                tool_result = ""

                if tool_name == "generate_plan":
                    plan = call_generate_plan(data, tool_input.get("special_requests", ""))
                    data["plan"] = plan
                    data["last_generated"] = datetime.now().isoformat()
                    tool_result = format_plan_whatsapp(plan)

                elif tool_name == "swap_days":
                    plan = call_swap_days(
                        data,
                        day1=tool_input.get("day1"),
                        day2=tool_input.get("day2"),
                        new_meal_request=tool_input.get("new_meal_request")
                    )
                    data["plan"] = plan
                    tool_result = format_plan_whatsapp(plan)

                elif tool_name == "update_pantry":
                    pantry = data.get("pantry", [])
                    for item in tool_input.get("add", []):
                        if item.lower() not in [p.lower() for p in pantry]:
                            pantry.append(item)
                    for item in tool_input.get("remove", []):
                        pantry = [p for p in pantry if item.lower() not in p.lower()]
                    data["pantry"] = pantry
                    tool_result = json.dumps({"pantry": pantry})

                elif tool_name == "get_shopping_list":
                    shopping = call_get_shopping_list(data)
                    tool_result = format_shopping_whatsapp(shopping)

                elif tool_name == "record_feedback":
                    meal = tool_input.get("meal_name", "")
                    sentiment = tool_input.get("sentiment", "ok")
                    notes = tool_input.get("notes", "")
                    prefs = data.setdefault("preferences", {})

                    if sentiment in ("loved", "kids_loved"):
                        loved_list = prefs.setdefault("loved_meals", [])
                        if meal not in loved_list:
                            loved_list.append(meal)
                    elif sentiment in ("disliked", "kids_disliked"):
                        disliked_list = prefs.setdefault("disliked_meals", [])
                        if meal not in disliked_list:
                            disliked_list.append(meal)

                    if notes:
                        prefs.setdefault("notes", []).append(f"{meal}: {notes}")

                    prefs.setdefault("feedback_history", []).append({
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "meal": meal,
                        "sentiment": sentiment,
                        "notes": notes,
                        "from": sender_number[-4:]  # last 4 digits only for privacy
                    })
                    tool_result = json.dumps({"recorded": True, "meal": meal, "sentiment": sentiment})

                elif tool_name == "show_plan":
                    if data.get("plan"):
                        tool_result = format_plan_whatsapp(data["plan"])
                    else:
                        tool_result = "No plan yet."

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result
                })

        history.append({"role": "assistant", "content": serialize_content(response.content)})
        history.append({"role": "user", "content": tool_results})

        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=1500,
            system=system,
            tools=TOOLS,
            messages=history
        )

    for block in response.content:
        if hasattr(block, "text"):
            result_text = block.text
            break

    history.append({"role": "assistant", "content": result_text})

    # Keep history from growing too large
    if len(history) > 24:
        history = history[-24:]
    conversations[sender_number] = history

    return result_text
