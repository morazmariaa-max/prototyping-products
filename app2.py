import streamlit as st
import pandas as pd

st.set_page_config(page_title="Priority-Aware Budget Assistant", layout="centered")

st.title("Priority-Aware Budget Assistant")
st.caption("Prototype of an AI-assisted dynamic budgeting feature for a banking app.")

# -----------------------------
# 1) Budget setup (inputs)
# -----------------------------
st.header("1) Budget setup")

col1, col2 = st.columns(2)
with col1:
    monthly_income = st.number_input("Monthly income (€)", min_value=0.0, value=2500.0, step=50.0)
with col2:
    total_budget = st.number_input("Total monthly spending budget (€)", min_value=0.0, value=1600.0, step=50.0)

st.markdown("Define category budgets and priorities (1 = low priority, 5 = high priority).")

default_categories = [
    {"category": "Groceries", "budget": 350.0, "priority": 5},
    {"category": "Eating out", "budget": 250.0, "priority": 3},
    {"category": "Leisure", "budget": 200.0, "priority": 2},
    {"category": "Transport", "budget": 100.0, "priority": 4},
]

# Let user set number of categories (simple control, no “dynamic add” yet)
n_categories = st.slider("Number of categories", min_value=2, max_value=6, value=4)

# Build category inputs
categories = []
for i in range(n_categories):
    st.subheader(f"Category {i+1}")

    # Use defaults if available
    default_name = default_categories[i]["category"] if i < len(default_categories) else f"Category {i+1}"
    default_budget = default_categories[i]["budget"] if i < len(default_categories) else 100.0
    default_priority = default_categories[i]["priority"] if i < len(default_categories) else 3

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Name", value=default_name, key=f"name_{i}")
    with c2:
        budget = st.number_input(
            "Budget (€)", min_value=0.0, value=float(default_budget), step=10.0, key=f"budget_{i}"
        )
    with c3:
        priority = st.slider("Priority", 1, 5, int(default_priority), key=f"prio_{i}")

    categories.append({"category": name.strip(), "budget": float(budget), "priority": int(priority)})

# Clean empty names
categories = [c for c in categories if c["category"] != ""]

# If user somehow deletes all category names, stop safely
if len(categories) == 0:
    st.error("Please add at least one category (category name cannot be empty).")
    st.stop()

budgets_df = pd.DataFrame(categories)

# ✅ Compute total planned BEFORE using it
total_planned = budgets_df["budget"].sum()

# ---- Allocation check: do category budgets fit within total budget? ----
allocation_gap = total_budget - total_planned  # + = still available, - = overallocated

st.subheader("Allocation check")

cA, cB = st.columns(2)
cA.metric("Total allocated across categories (€)", f"{total_planned:,.0f}")
cB.metric("Remaining to allocate (€)", f"{allocation_gap:,.0f}")

if allocation_gap > 0:
    st.info(
        f"You still have **€{allocation_gap:,.0f}** unallocated. "
        "You can distribute it across categories."
    )
elif allocation_gap < 0:
    st.error(
        f"You are **€{abs(allocation_gap):,.0f}** over budget. "
        "Reduce one or more category budgets to stay within the total limit."
    )
else:
    st.success("Perfect! Your category budgets add up exactly to your total monthly budget.")

# Visual progress toward allocation (caps at 100% for display)
if total_budget > 0:
    pct = min(total_planned / total_budget, 1.0)
    st.progress(pct, text=f"Allocated {total_planned:,.0f} / {total_budget:,.0f} (€)")

st.divider()

# -----------------------------
# 2) Spending monitoring (simulated for prototype)
# -----------------------------
st.header("2) Spending monitoring (prototype simulation)")

st.write(
    "For prototyping purposes, you can simulate current spending per category. "
    "Later, this could be replaced by real transaction data."
)

spending = []
for i, row in budgets_df.iterrows():
    spent = st.number_input(
        f"Current spent in {row['category']} (€)",
        min_value=0.0,
        value=max(0.0, row["budget"] * 0.6),
        step=10.0,
        key=f"spent_{i}",
    )
    spending.append(float(spent))

budgets_df["spent_so_far"] = spending
budgets_df["remaining"] = budgets_df["budget"] - budgets_df["spent_so_far"]

total_spent = budgets_df["spent_so_far"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Planned category budgets (€)", f"{total_planned:,.0f}")
c2.metric("Total spent so far (€)", f"{total_spent:,.0f}")
c3.metric("Total budget limit (€)", f"{total_budget:,.0f}")

st.divider()

# -----------------------------
# 3) Risk detection (simple logic)
# -----------------------------
st.header("3) Risk detection")

# Simple projection rule (prototype): assume user keeps same spending pace for the rest of the month.
day = st.slider("Day of month", 1, 31, 15)
days_in_month = 30  # simplification for prototype

pace_factor = days_in_month / day
projected_total = total_spent * pace_factor

st.write(f"**Projected end-of-month spending:** €{projected_total:,.0f}")

if projected_total <= total_budget:
    st.success("You are currently on track to stay within your total monthly budget.")
else:
    st.warning("You may exceed your total monthly budget if you keep this spending pace.")

st.divider()

# -----------------------------
# 4) Recommendation: reallocate budgets based on priorities
# -----------------------------
st.header("4) Recommended budget adjustment")

st.write(
    "If there is a risk of exceeding the total budget, the assistant suggests reallocating "
    "budget from lower-priority categories to protect higher-priority ones."
)

def suggest_reallocation(df: pd.DataFrame, total_budget_limit: float):
    """
    Prototype recommendation:
    - If category budgets exceed the total budget limit, suggest reductions.
    - Reduce lowest-priority categories first (and avoid extreme reductions).
    """
    planned_sum = df["budget"].sum()
    overflow = planned_sum - total_budget_limit

    if overflow <= 0:
        return overflow, pd.DataFrame()

    # Sort by priority asc (lowest priority first), then by largest budget
    candidates = df.sort_values(by=["priority", "budget"], ascending=[True, False]).copy()

    adjustments = []
    remaining_overflow = overflow

    for _, r in candidates.iterrows():
        if remaining_overflow <= 0:
            break

        # Reduce at most 30% of a category budget in this prototype
        max_reducible = 0.30 * r["budget"]
        reduction = min(max_reducible, remaining_overflow)

        if reduction > 0:
            adjustments.append(
                {
                    "category": r["category"],
                    "priority": r["priority"],
                    "current_budget": r["budget"],
                    "suggested_budget": r["budget"] - reduction,
                    "reduction": reduction,
                }
            )
            remaining_overflow -= reduction

    return overflow, pd.DataFrame(adjustments)

overflow, rec_df = suggest_reallocation(budgets_df, total_budget)

if overflow <= 0:
    st.info("No reallocation needed: your planned category budgets fit within your total budget.")
else:
    st.write(f"Your planned category budgets exceed the total budget by **€{overflow:,.0f}**.")
    if rec_df.empty:
        st.error("The prototype could not generate a safe reallocation suggestion with the current rules.")
    else:
        st.dataframe(rec_df, use_container_width=True)

        st.subheader("Apply suggestion?")
        apply = st.checkbox("Apply recommended budgets (prototype action)")

        if apply:
            new_df = budgets_df.copy()
            for _, adj in rec_df.iterrows():
                new_df.loc[new_df["category"] == adj["category"], "budget"] = adj["suggested_budget"]

            new_total = new_df["budget"].sum()
            st.success("Suggestion applied (prototype).")
            st.write(f"**New planned budgets total:** €{new_total:,.0f} (target: €{total_budget:,.0f})")

            st.write("✅ This keeps your overall plan aligned while protecting your highest-priority categories.")

st.divider()

# -----------------------------
# 5) What would be AI in a real version?
# -----------------------------
with st.expander("How AI would improve this in a real product (optional explanation)"):
    st.markdown(
        """
- **Transaction classification**: Automatically assign each expense to a category.
- **Personalized forecasting**: Predict end-of-month spending using your historical patterns (seasonality, habits).
- **Smarter recommendations**: Suggest reallocations that fit your behavior (not only simple rules).
- **User preferences learning**: Learn which suggestions you tend to accept and adapt future recommendations.
        """
    )
