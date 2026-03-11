import json
import re
import requests
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="AI Financial Advisor", layout="wide")

st.title("AI Financial Advisor")
st.caption("Dynamic budgeting with local LLM support via Ollama")

# =========================
# Ollama config
# =========================
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# =========================
# Default values for a good demo
# =========================
DEFAULT_TOTAL_BUDGET = 2200

DEFAULT_BUDGETS = {
    "Rent": 900,
    "Groceries": 300,
    "Dining": 180,
    "Transport": 180,
    "Entertainment": 140,
    "Savings": 500,
}

DEFAULT_SPENDING = {
    "Rent": 900,
    "Groceries": 320,
    "Dining": 290,
    "Transport": 110,
    "Entertainment": 220,
    "Savings": 360,
}

DEFAULT_PRIORITIES = {
    "Rent": "High",
    "Groceries": "High",
    "Dining": "Low",
    "Transport": "Medium",
    "Entertainment": "Low",
    "Savings": "Medium",
}

CATEGORIES = list(DEFAULT_BUDGETS.keys())
PRIORITY_SCORE = {"High": 3, "Medium": 2, "Low": 1}

# =========================
# Helper functions
# =========================
def build_dataframe(category_budgets, current_spending):
    df = pd.DataFrame({
        "Category": CATEGORIES,
        "Budget": [category_budgets[c] for c in CATEGORIES],
        "Spent": [current_spending[c] for c in CATEGORIES],
    })
    df["Remaining"] = df["Budget"] - df["Spent"]
    df["Status"] = df["Remaining"].apply(
        lambda x: "Overspent" if x < 0 else ("On track" if x == 0 else "Under budget")
    )
    return df


def compute_reallocation(df, priorities):
    deficits = []
    donors = []

    for _, row in df.iterrows():
        category = row["Category"]
        remaining = row["Remaining"]

        item = {
            "category": category,
            "amount": abs(float(remaining)),
            "priority": priorities[category],
            "priority_score": PRIORITY_SCORE[priorities[category]],
        }

        if remaining < 0:
            deficits.append(item)
        elif remaining > 0:
            donors.append({
                "category": category,
                "amount": float(remaining),
                "priority": priorities[category],
                "priority_score": PRIORITY_SCORE[priorities[category]],
            })

    # First cover deficits in higher-priority categories
    deficits = sorted(deficits, key=lambda x: (-x["priority_score"], -x["amount"]))
    # First use extra money from lower-priority categories
    donors = sorted(donors, key=lambda x: (x["priority_score"], -x["amount"]))

    moves = []

    for deficit in deficits:
        needed = deficit["amount"]

        for donor in donors:
            if donor["amount"] <= 0:
                continue

            # Do not move from a higher-priority category to a lower-priority one
            if donor["priority_score"] > deficit["priority_score"]:
                continue

            move = min(donor["amount"], needed)

            if move > 0:
                moves.append({
                    "from": donor["category"],
                    "to": deficit["category"],
                    "amount": round(move, 2)
                })
                donor["amount"] -= move
                needed -= move

            if needed <= 0:
                break

    return moves


def apply_reallocation(df, moves):
    adjusted = df.copy()
    adjusted["Adjusted Budget"] = adjusted["Budget"].astype(float)

    for move in moves:
        adjusted.loc[adjusted["Category"] == move["from"], "Adjusted Budget"] -= move["amount"]
        adjusted.loc[adjusted["Category"] == move["to"], "Adjusted Budget"] += move["amount"]

    adjusted["Adjusted Remaining"] = adjusted["Adjusted Budget"] - adjusted["Spent"]
    return adjusted


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([€$£])\s+", r"\1", text)
    text = re.sub(r"(\d)([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    return text.strip()


def create_prompt(df, priorities, moves):
    summary = []
    for _, row in df.iterrows():
        summary.append({
            "category": row["Category"],
            "budget": float(row["Budget"]),
            "spent": float(row["Spent"]),
            "remaining": float(row["Remaining"]),
            "status": row["Status"],
            "priority": priorities[row["Category"]],
        })

    payload = {
        "budget_summary": summary,
        "proposed_reallocations": moves
    }

    return f"""
You are a financial budgeting assistant.

Analyze the budget data and proposed reallocations.

IMPORTANT RULES:
- Be concise and practical.
- Do not use italics, markdown, or special formatting.
- Do not invent categories or amounts.
- Only refer to the categories and reallocations provided.
- Explain clearly why the reallocation makes sense.
- If a category is under budget, do not suggest increasing it unless there is a strong reason.
- Prioritize protecting High priority categories.
- Prefer moving unused budget from Low or Medium priority categories.
- Do not recommend increasing a category that is already under budget unless it supports a high-priority category.
- Write in plain English.

Return ONLY valid JSON with this exact structure:
{{
  "overview": "2-3 short sentences",
  "risks": [
    "risk 1",
    "risk 2"
  ],
  "actions": [
    "action 1",
    "action 2",
    "action 3"
  ]
}}

Data:
{json.dumps(payload, indent=2)}
"""


def call_ollama(prompt, model=OLLAMA_MODEL):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:
            return False, f"Ollama error ({response.status_code}): {response.text}"

        result = response.json()
        raw_text = result.get("response", "").strip()

        if not raw_text:
            return False, "Ollama returned an empty response."

        try:
            parsed = json.loads(raw_text)
            return True, parsed
        except Exception:
            return False, f"The model returned invalid JSON:\n\n{raw_text}"

    except requests.exceptions.ConnectionError:
        return False, (
            "No connection to Ollama. Make sure Ollama is installed, open, and running."
        )
    except Exception as e:
        return False, f"Request failed: {str(e)}"


# =========================
# Sidebar
# =========================
st.sidebar.header("Budget setup")

total_budget = st.sidebar.number_input(
    "Total monthly budget (€)",
    min_value=0,
    value=DEFAULT_TOTAL_BUDGET,
    step=50
)

st.sidebar.subheader("Category budgets")
category_budgets = {}
for cat in CATEGORIES:
    category_budgets[cat] = st.sidebar.number_input(
        f"{cat} budget (€)",
        min_value=0,
        value=DEFAULT_BUDGETS[cat],
        step=10
    )

st.sidebar.subheader("Priorities")
priorities = {}
for cat in CATEGORIES:
    default_idx = ["Low", "Medium", "High"].index(DEFAULT_PRIORITIES[cat])
    priorities[cat] = st.sidebar.selectbox(
        f"{cat} priority",
        options=["Low", "Medium", "High"],
        index=default_idx
    )

st.sidebar.subheader("Current spending")
current_spending = {}
for cat in CATEGORIES:
    current_spending[cat] = st.sidebar.number_input(
        f"{cat} spent (€)",
        min_value=0,
        value=DEFAULT_SPENDING[cat],
        step=10
    )

# =========================
# Main data
# =========================
df = build_dataframe(category_budgets, current_spending)
moves = compute_reallocation(df, priorities)
adjusted_df = apply_reallocation(df, moves)

spent_total = float(df["Spent"].sum())
remaining_total = total_budget - spent_total

# =========================
# Top metrics
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Total budget", f"€{total_budget:,.0f}")
col2.metric("Total spent", f"€{spent_total:,.0f}")
col3.metric("Budget left", f"€{remaining_total:,.0f}")

# =========================
# Current status table
# =========================
st.subheader("Current budget status")
st.dataframe(df, use_container_width=True)

overspent_df = df[df["Remaining"] < 0]
if not overspent_df.empty:
    st.warning("Some categories are overspending.")
else:
    st.success("No categories are overspending.")

# =========================
# Reallocation suggestions
# =========================
st.subheader("Suggested reallocation")

if moves:
    for move in moves:
        st.write(f"Move **€{move['amount']:.0f}** from **{move['from']}** to **{move['to']}**")
else:
    st.info("No reallocation is needed with the current values.")

# =========================
# Adjusted budget table
# =========================
st.subheader("Budget after reallocation")
st.dataframe(
    adjusted_df[["Category", "Spent", "Budget", "Adjusted Budget", "Remaining", "Adjusted Remaining"]],
    use_container_width=True
)

# =========================
# Side-by-side bar chart
# =========================
st.subheader("Budget vs spent")

chart_data = df.melt(
    id_vars="Category",
    value_vars=["Budget", "Spent"],
    var_name="Type",
    value_name="Amount"
)

chart = alt.Chart(chart_data).mark_bar().encode(
    x=alt.X("Category:N", title="Category"),
    xOffset="Type:N",
    y=alt.Y("Amount:Q", title="Amount (€)"),
    color=alt.Color("Type:N", title=""),
    tooltip=["Category", "Type", "Amount"]
).properties(
    height=420
)

st.altair_chart(chart, use_container_width=True)

# =========================
# AI advice
# =========================
st.subheader("AI financial advice")
st.caption("This version uses a local LLM through Ollama, so it does not require a paid API.")

if st.button("Generate AI Advice"):
    prompt = create_prompt(df, priorities, moves)

    with st.spinner("Generating advice with local LLM..."):
        ok, result = call_ollama(prompt)

        if ok:
            overview = clean_text(result.get("overview", ""))
            risks = result.get("risks", [])
            actions = result.get("actions", [])

            st.markdown("### Short overview")
            st.write(overview)

            st.markdown("### Two risks")
            for i, risk in enumerate(risks[:2], start=1):
                st.write(f"{i}. {clean_text(risk)}")

            st.markdown("### Three actions")
            for i, action in enumerate(actions[:3], start=1):
                st.write(f"{i}. {clean_text(action)}")
        else:
            st.error(result)
            st.info("If this happens, try again. Local models sometimes fail to follow JSON format on the first attempt.")