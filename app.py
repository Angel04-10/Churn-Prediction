"""Interactive chat interface for the churn model.

Run locally:   python app.py   (then open http://localhost:7860)
"""

from __future__ import annotations

import gradio as gr

import churn_model
from question_parser import is_reset, parse

BUNDLE = churn_model.load()

LABELS = {
    "Age": ("Age", "{:.0f} yrs"),
    "Gender": ("Gender", "{}"),
    "Metro": ("Metro city", None),
    "Account_Age_Months": ("Account age", "{:.0f} months"),
    "Days_Since_Purchase": ("Days since purchase", "{:.0f}"),
    "Purchase_Frequency_Per_Month": ("Purchase frequency / month", "{:.2f}"),
    "Avg_Purchase_Amount_INR": ("Avg purchase amount", "₹{:,.0f}"),
    "Return_Frequency": ("Return frequency", "{:.2f}"),
    "Review_Rating": ("Review rating", "{:.1f} / 5"),
    "Customer_Service_Interactions": ("Customer service interactions", "{:.0f}"),
    "Subscription_Status": ("Subscription status", "{}"),
    "Discount_Sensitivity": ("Discount sensitivity", "{:.2f}"),
    "Days_Since_Last_Discount": ("Days since last discount", "{:.0f}"),
    "Mobile_App_User": ("Mobile app user", None),
}

HELP = f"""I predict the **probability that a customer churns** with a calibrated
{BUNDLE['model_name']} model (see `MODEL_REPORT.md`), and flag customers at or above
**{BUNDLE['threshold']:.0%}** for a retention offer — a threshold tuned to catch most
churners ({BUNDLE['evaluation']['at_threshold']['recall']:.0%} on held-out data). Describe a customer in plain English, e.g.

> *What is the probability of churn if days since purchase is 123, purchase frequency
> per month is 2.21, return frequency is 0.26 and subscription status is regular?*

Things I understand: **days since purchase**, **purchase frequency per month**,
**return frequency** (0–1 or %), **subscription** (premium / regular / none),
**review rating**, **avg purchase amount**, **customer service interactions**,
**age**, **gender**, **city / metro**, **account age (months)**, **discount sensitivity**,
**days since last discount**, **mobile app user**.

Anything you don't mention is filled with a typical customer's value. Follow-ups such as
*"what if they were premium?"* change only that detail; say **"new customer"** to start over."""


def _fmt(feature: str, value) -> str:
    label, fmt = LABELS[feature]
    if fmt is None:
        return f"{label}: {'yes' if value else 'no'}"
    return f"{label}: {fmt.format(value)}"


def _risk(p: float) -> tuple[str, str]:
    if p >= 0.5:
        return "🔴 High risk", "**Target now** — prioritise a win-back / retention offer."
    if p >= BUNDLE["threshold"]:
        return "🟠 Elevated risk", "**Target** — include in the retention campaign."
    return "🟢 Low risk", "Not targeted — keep nurturing."


def _user_texts(history) -> list[str]:
    texts = []
    for turn in history or []:
        if isinstance(turn, dict):
            if turn.get("role") != "user":
                continue
            content = turn.get("content")
        else:  # (user, assistant) tuples from older Gradio versions
            content = turn[0]
        if isinstance(content, list):
            content = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
        if isinstance(content, str):
            texts.append(content)
    return texts


def _customer_from_conversation(message: str, history) -> dict:
    """Merge details from earlier turns so follow-up 'what if' questions work."""
    customer: dict = {}
    for text in _user_texts(history) + [message]:
        if is_reset(text):
            customer = {}
        customer.update(parse(text))
    return customer


def respond(message: str, history, *_ignored) -> str:
    if not message.strip() or message.strip().lower() in {"help", "?", "hi", "hello", "hey"}:
        return HELP

    new_details = parse(message)
    customer = _customer_from_conversation(message, history)
    if not new_details and not is_reset(message):
        return (
            "I couldn't find any customer details in that question. 🤔\n\n" + HELP
        )
    if not customer:
        return "Okay, starting fresh. Describe the new customer."

    p = churn_model.predict(BUNDLE, customer)
    risk, action = _risk(p)

    lines = [
        f"### Churn probability: **{p:.0%}**",
        f"{risk} — {action}",
        "",
        "**Customer details used**",
    ]
    lines += [f"- {_fmt(f, customer[f])}" for f in churn_model.RAW_FEATURES if f in customer]

    # Which of the given details push the risk up or down (vs. a typical customer)?
    effects = []
    for f in customer:
        baseline = {**customer, f: BUNDLE["defaults"][f]}
        delta = p - churn_model.predict(BUNDLE, baseline)
        if abs(delta) >= 0.02:
            effects.append((delta, f))
    if effects:
        lines += ["", "**What's driving it** (vs. a typical value)"]
        for delta, f in sorted(effects, key=lambda e: -abs(e[0])):
            arrow = "⬆️ raises" if delta > 0 else "⬇️ lowers"
            lines.append(f"- {LABELS[f][0]} {arrow} risk by {abs(delta):.0%} points")

    warnings = []
    for f, v in customer.items():
        if f in BUNDLE["ranges"]:
            lo, hi = BUNDLE["ranges"][f]
            if not lo <= v <= hi:
                warnings.append(f"{LABELS[f][0]} = {v} is outside the training range ({lo:g}–{hi:g})")
    if warnings:
        lines += ["", "⚠️ " + "; ".join(warnings) + ". Treat the estimate with caution."]

    missing = [LABELS[f][0].lower() for f in churn_model.RAW_FEATURES if f not in customer]
    if missing:
        lines += [
            "",
            f"<sub>Not specified, so typical values were assumed: {', '.join(missing)}. "
            f"Model: {BUNDLE['model_name']}, retention threshold {BUNDLE['threshold']:.0%}. "
            f"Average churn rate in the data is {BUNDLE['base_churn_rate']:.0%}.</sub>",
        ]
    return "\n".join(lines)


EXAMPLES = [
    "What is the probability of churn if days since purchase is 123, purchase frequency per month is 2.21, return frequency is 0.26 and subscription status is regular?",
    "A 28 year old woman in Mumbai, last purchase 10 days ago, 3.5 purchases per month, premium subscription, rating 4.7",
    "Customer with no subscription, 95 days since purchase, 35% returns, 5 customer service calls",
    "What if they had a premium subscription?",
]

demo = gr.ChatInterface(
    fn=respond,
    title="🛒 Customer Churn Predictor",
    description="Ask in plain English about a customer and get their probability of churning. Type **help** for everything I understand.",
    examples=EXAMPLES,
    cache_examples=False,
)

if __name__ == "__main__":
    import os

    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
