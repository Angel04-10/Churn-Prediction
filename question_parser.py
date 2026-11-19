"""Turn a free-text question into customer feature values.

Rule-based (regex) so it runs offline with no API key. It understands phrasing like:
  "what is the probability of churn if days since purchase is 123,
   purchase frequency per month is 2.21, return frequency is 0.26,
   subscription status is regular"
  "35 year old male in Mumbai, 2 purchases per month, 26% returns, premium member"
"""

from __future__ import annotations

import re

NUM = r"(?P<num>\d+(?:\.\d+)?)\s*(?P<pct>%|percent\b)?"
# Filler words allowed between a feature name and its value: "is", "of about", "=", ...
FILLER = r"(?:\s*(?:\b(?:is|was|are|of|at|around|about|approx(?:imately)?|equals?|equal\s+to|being|roughly|nearly|=|:|-)\b|[=:\-])\s*)*\s*"

# Order matters: more specific names first, and each match is blanked out
# so e.g. "days since last discount" is never read as "days since purchase".
FEATURE_PATTERNS: list[tuple[str, str]] = [
    ("Days_Since_Last_Discount", r"days?\s+since\s+(?:the\s+|their\s+|his\s+|her\s+)?(?:last\s+)?discount|last\s+discount(?:\s+days?)?"),
    ("Discount_Sensitivity", r"discount\s+sensitivity|sensitivity\s+to\s+discounts?"),
    ("Days_Since_Purchase", r"days?\s+since\s+(?:the\s+|their\s+|his\s+|her\s+)?(?:last\s+)?(?:purchase|order|buy)|purchase\s+recency|recency|days?\s+(?:inactive|of\s+inactivity)|last\s+(?:purchase|order)(?:d)?(?:\s+days?)?(?:\s+ago)?|days?\s+ago"),
    ("Purchase_Frequency_Per_Month", r"purchase\s+frequency(?:\s+(?:per|a|each)\s+month)?|(?:purchases?|orders?|buys?|transactions?)\s+(?:per|a|each|every)\s+month|monthly\s+(?:purchases?|orders?|purchase\s+frequency)"),
    ("Return_Frequency", r"return\s+(?:frequency|rate|ratio)|returns?(?:\s+(?:frequency|rate))?"),
    ("Avg_Purchase_Amount_INR", r"(?:average|avg\.?|mean)\s+(?:purchase|order|basket)\s+(?:amount|value|size)(?:\s+(?:inr|in\s+inr|rs\.?|rupees|₹))?|(?:purchase|order)\s+(?:amount|value)|(?:average\s+)?spend(?:s|ing)?|basket\s+(?:size|value)"),
    ("Review_Rating", r"review\s+rating|ratings?|reviews?|stars?"),
    ("Customer_Service_Interactions", r"customer\s+(?:service|support|care)\s+(?:interactions?|calls?|contacts?|tickets?|complaints?)|(?:support|service)\s+(?:interactions?|calls?|tickets?|contacts?)|cs\s+interactions?|complaints?"),
    ("Account_Age_Months", r"account\s+age(?:\s+(?:in\s+)?months?)?|tenure(?:\s+(?:in\s+)?months?)?|months?\s+(?:as\s+a\s+customer|with\s+us|on\s+the\s+platform|old\s+account|account)|account\s+(?:is\s+)?(?:months?\s+)?old"),
    ("Age", r"\bage(?:d)?\b|years?\s+old|yrs?\s+old|year[-\s]old|yo\b"),
]

METRO_CITIES = ["mumbai", "delhi", "bangalore", "bengaluru", "new delhi"]
NON_METRO_CITIES = ["hyderabad", "pune", "chennai", "kolkata", "ahmedabad", "jaipur", "lucknow"]

INT_FEATURES = {"Age", "Account_Age_Months", "Days_Since_Purchase",
                "Customer_Service_Interactions", "Days_Since_Last_Discount"}
FRACTION_FEATURES = {"Return_Frequency", "Discount_Sensitivity"}


def _blank(text: str, span: tuple[int, int]) -> str:
    a, b = span
    return text[:a] + " " * (b - a) + text[b:]


def _to_value(feature: str, m: re.Match) -> float:
    v = float(m.group("num").replace(",", ""))
    if feature in FRACTION_FEATURES and (m.group("pct") or v > 1):
        v /= 100.0
    if feature in INT_FEATURES:
        v = int(round(v))
    return v


def parse(question: str) -> dict:
    """Extract every feature mentioned in `question`."""
    text = question.lower().replace("₹", " ").replace("rs.", " ")
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)  # 2,500 -> 2500
    found: dict = {}

    for feature, kw in FEATURE_PATTERNS:
        after = re.compile(rf"(?:{kw}){FILLER}(?:₹|rs\.?|inr)?\s*{NUM}")
        before = re.compile(rf"{NUM}\s*(?:days?\s+|months?\s+|times?\s+|stars?\s+|-)?(?:{kw})")
        for rx in (after, before):
            m = rx.search(text)
            if m:
                found[feature] = _to_value(feature, m)
                text = _blank(text, m.span())
                break

    # Number of months phrased like "customer for 12 months"
    if "Account_Age_Months" not in found:
        m = re.search(rf"(?:customer|member|user|account)\s+(?:for|since)\s+{NUM}\s*months?", text)
        if m:
            found["Account_Age_Months"] = _to_value("Account_Age_Months", m)
            text = _blank(text, m.span())
        else:
            m = re.search(rf"(?:customer|member|user|account)\s+(?:for|since)\s+{NUM}\s*years?", text)
            if m:
                found["Account_Age_Months"] = int(round(float(m.group("num")) * 12))

    # Subscription status
    if re.search(
        r"\b(?:no\s+(?:subscription|membership|plan)|without\s+(?:a\s+|any\s+)?(?:subscription|membership)|not\s+subscribed|unsubscribed|non[-\s]?(?:subscriber|member)|not\s+a\s+(?:subscriber|member)|(?:subscription|membership)(?:\s+status)?\s*(?:is\s+|=\s*|:\s*)?(?:none|no)\b)",
        text,
    ):
        found["Subscription_Status"] = "None"
    elif re.search(r"\bpremium\b|\bprime\b|\bgold\b|\bvip\b", text):
        found["Subscription_Status"] = "Premium"
    elif re.search(r"\bregular\b|\bbasic\b|\bstandard\b|\bsubscri(?:bed|ber|ption\s+(?:status\s+)?(?:is\s+)?(?:active|yes))\b|\bhas\s+(?:a\s+)?subscription\b", text):
        found["Subscription_Status"] = "Regular"

    # Gender
    if re.search(r"\b(?:female|woman|lady|girl|she)\b", text):
        found["Gender"] = "Female"
    elif re.search(r"\b(?:male|man|guy|boy|he)\b", text):
        found["Gender"] = "Male"

    # Metro / city
    if re.search(r"\bnon[-\s]?metro\b|\btier[-\s]?(?:2|3|ii|iii)\b|\bnot\s+(?:in\s+)?(?:a\s+)?metro\b", text) or any(
        re.search(rf"\b{c}\b", text) for c in NON_METRO_CITIES
    ):
        found["Metro"] = 0
    elif re.search(r"\bmetro\b|\btier[-\s]?(?:1|i)\b", text) or any(re.search(rf"\b{c}\b", text) for c in METRO_CITIES):
        found["Metro"] = 1

    # Mobile app usage
    if re.search(r"\b(?:no|not|doesn'?t|does\s+not|never|without)\b(?:\s+\w+){0,3}\s+(?:mobile\s+)?app\b|\bnon[-\s]?app\b|\bweb(?:site)?\s+(?:only|user)\b|\bdesktop\b", text):
        found["Mobile_App_User"] = 0
    elif re.search(r"\b(?:mobile\s+)?app\s+user\b|\buses?\s+(?:the\s+)?(?:mobile\s+)?app\b|\bmobile\s+app\b", text):
        found["Mobile_App_User"] = 1

    return found


def is_reset(question: str) -> bool:
    """User wants to start describing a brand-new customer."""
    return bool(re.search(r"\b(?:new\s+customer|another\s+customer|different\s+customer|reset|start\s+over|clear)\b", question.lower()))
