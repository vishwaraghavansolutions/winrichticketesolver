import os
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import streamlit as st
from rapidfuzz import process, fuzz

# ============================================================
# CONFIG
# ============================================================

PORTFOLIO_PATH = "data/Datawarehouse_MutualFunds_2026_01_01_mutualfunds.csv"
SCHEME_MASTER_PATH = "data/SchemeData2301262313SS.csv"

REQUIRED_PORTFOLIO_COLUMNS = [
    "h_name", "c_name", "sCode", "s_name", "foliono", "Nature",
    "FolioStartDate", "BalUnit", "AvgCost", "InvAmt", "TotalInvAmt",
    "CurNAV", "CurValue", "DivAmt", "NotionalGain", "ActualGain",
    "FolioXIRR", "NatureXIRR", "ClientXIRR", "NatureAbs", "ClientAbs",
    "absReturn", "ValueDate", "ReportDate", "Email", "Mobile"
]

SENSITIVE_FIELDS = ["Email", "Mobile"]

REQUIRED_SCHEME_COLUMNS = [
    "AMC", "Code", "Scheme Name", "Scheme Type", "Scheme Category",
    "Scheme NAV Name", "Scheme Minimum Amount", "Launch Date",
    "Closure Date", "ISIN Div Payout/ ISIN GrowthISIN Div Reinvestment"
]

# ------------------------------------------------------------
# PLACEHOLDERS – plug in your real implementations
# ------------------------------------------------------------

RANKING_FILES = {
    # "Large Cap": "data/ranking_large_cap.csv",
    # "Flexi Cap": "data/ranking_flexi_cap.csv",
    # ...
}

def load_ranking_file(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

def get_bottom_half_funds(df):
    if df.empty:
        return set()
    # assumes df has a 'Rank' column where lower is better
    df_sorted = df.sort_values("Rank")
    half = len(df_sorted) // 2
    bottom = df_sorted.iloc[half:]
    # assumes df has a 'Scheme Name' column
    return set(bottom["Scheme Name"].tolist())

def build_customer_analysis(filtered_customer_master: pd.DataFrame) -> pd.DataFrame:
    # Your existing implementation
    return filtered_customer_master.copy()

def load_and_override_targets(scheme_master: pd.DataFrame):
    # Your existing implementation
    # Should return: asset_class_targets, fund_type_targets
    return {}, {}

def validate_targets(asset_class_targets, fund_type_targets) -> bool:
    # Your existing implementation
    return True

def update_selected_customer_analysis(selected_customer_analysis_master,
                                      filtered_customer_master,
                                      asset_class_targets,
                                      fund_type_targets):
    # Your existing implementation
    return selected_customer_analysis_master


# ============================================================
# HELPERS
# ============================================================

def derive_asset_class_from_category(category):
    if not isinstance(category, str):
        return "Unknown"

    cat = category.lower()

    if "equity" in cat:
        return "Equity"
    if "debt" in cat:
        return "Debt"
    if "hybrid" in cat:
        return "Hybrid"
    if "solution" in cat:
        return "Solution Oriented"
    if "other" in cat or "fof" in cat or "fund of funds" in cat:
        return "Other"

    return "Other"


def format_in_indian(value):
    try:
        n = float(value)
    except (TypeError, ValueError):
        return ""

    n = int(round(n))
    s = str(abs(n))
    sign = "-" if n < 0 else ""

    if len(s) <= 3:
        return f"{sign}₹{s}"

    last3 = s[-3:]
    rest = s[:-3]

    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)

    parts.reverse()
    formatted = ",".join(parts) + "," + last3

    return f"{sign}₹{formatted}"


def load_portfolio_local(path):
    if not os.path.exists(path):
        st.error(f"Portfolio file not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_PORTFOLIO_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Portfolio CSV missing columns: {missing}")
        return pd.DataFrame()

    df = df.drop(columns=SENSITIVE_FIELDS, errors="ignore")
    return df


def load_scheme_master_local(path):
    if not os.path.exists(path):
        st.error(f"Scheme Master file not found at: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_SCHEME_COLUMNS if c not in df.columns]
    if missing:
        st.error(f"Scheme Master CSV missing columns: {missing}")
        return pd.DataFrame()

    def split_category(cat):
        if not isinstance(cat, str):
            return "Unknown Class", "Unknown Type"
        parts = [p.strip() for p in cat.split("-", 1)]
        if len(parts) == 1:
            return parts[0], "Unknown Type"
        return parts[0], parts[1]

    df["fund_class"], df["fund_type"] = zip(*df["Scheme Category"].apply(split_category))
    df["asset_class"] = df["Scheme Category"].apply(derive_asset_class_from_category)

    return df


def select_customer_before_classification(customer_master):
    st.subheader("Select Customer")

    customers = sorted(customer_master["c_name"].unique())
    selected = st.selectbox("Choose customer", customers)

    filtered = customer_master[customer_master["c_name"] == selected].copy()

    st.write(f"Selected: {selected}")
    return selected, filtered


# ============================================================
# STEP 3 — Async fuzzy matching
# ============================================================

executor = ThreadPoolExecutor(max_workers=10)
fund_match_cache = {}


def fuzzy_match_with_index(fund_name, scheme_master, scheme_names):
    if not isinstance(fund_name, str) or fund_name.strip() == "":
        return ("Unknown", "Unknown", "Unknown", "Unknown")

    key = fund_name.lower().strip()
    if key in fund_match_cache:
        return fund_match_cache[key]

    match, score, idx = process.extractOne(
        key,
        scheme_names,
        scorer=fuzz.WRatio,
        processor=None,
        score_cutoff=80
    )

    if match is None:
        result = ("Unknown", "Unknown", "Unknown", "Unknown")
    else:
        sm_row = scheme_master.iloc[idx]
        asset_class = derive_asset_class_from_category(sm_row["Scheme Category"])
        result = (
            asset_class,
            sm_row["fund_type"],
            sm_row["fund_class"],
            sm_row["AMC"]
        )

    fund_match_cache[key] = result
    return result


async def fuzzy_match_async(fund_name, scheme_master, scheme_names):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        fuzzy_match_with_index,
        fund_name,
        scheme_master,
        scheme_names
    )


async def add_classifications_async(filtered_df, scheme_master):
    scheme_names = scheme_master["Scheme Name"].str.lower().str.strip().tolist()
    tasks = [
        fuzzy_match_async(f, scheme_master, scheme_names)
        for f in filtered_df["s_name"]
    ]
    results = await asyncio.gather(*tasks)

    filtered_df["asset_class"] = [r[0] for r in results]
    filtered_df["fund_type"] = [r[1] for r in results]
    filtered_df["fund_class"] = [r[2] for r in results]
    filtered_df["AMC"] = [r[3] for r in results]

    return filtered_df


# ============================================================
# STEP 5 — Target Model Editor (per-customer)
# ============================================================

def step5_target_model_editor(row):
    customer_model_path = "customer_target_model.json"

    if os.path.exists(customer_model_path):
        with open(customer_model_path, "r") as f:
            customer_target_model = json.load(f)
    else:
        customer_target_model = {}

    customer_key = row["customer_id"]
    saved_model = customer_target_model.get(customer_key, None)

    # Auto-populate from customer's actual portfolio
    auto_ac_targets = {
        ac: round(data["actual_pct"] * 100, 2)
        for ac, data in row["asset_class"].items()
    }

    auto_ft_targets = {}
    for ac, ac_data in row["asset_class"].items():
        total_ac_value = ac_data["actual_value"]
        ft_values = {}

        for fname, fdata in row["funds"].items():
            if fdata["asset_class"] == ac:
                ft = fdata["fund_type"]
                ft_values[ft] = ft_values.get(ft, 0) + fdata["actual_value"]

        ft_pct = {
            ft: round((val / total_ac_value) * 100, 2) if total_ac_value > 0 else 0
            for ft, val in ft_values.items()
        }

        auto_ft_targets[ac] = ft_pct

    if saved_model:
        default_ac_targets = saved_model.get("asset_class_targets", auto_ac_targets)
        default_ft_targets = saved_model.get("fund_type_targets", auto_ft_targets)
        saved_notes = saved_model.get("advisor_notes", "")
        saved_risk_profile = saved_model.get("risk_profile", "")
    else:
        default_ac_targets = auto_ac_targets
        default_ft_targets = auto_ft_targets
        saved_notes = ""
        saved_risk_profile = ""

    st.header("Step 5 — Target Model Editor")

    updated_ac_targets = {}
    updated_ft_targets = {}

    # Asset Class Targets
    with st.expander("Asset Class Targets", expanded=True):
        st.subheader("Asset Class Allocation (Must total 100%)")

        for ac, pct in default_ac_targets.items():
            updated_ac_targets[ac] = st.number_input(
                f"{ac} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(pct),
                step=0.5,
                key=f"ac_{ac}"
            )

        ac_total = sum(updated_ac_targets.values())
        st.write(f"**Total: {ac_total}%**")

        if ac_total != 100:
            st.error("Asset class total must equal 100%.")

    # Fund-Type Targets
    for ac, ft_dict in default_ft_targets.items():
        with st.expander(f"Fund Types for {ac}", expanded=False):
            st.subheader(f"{ac} — Fund-Type Allocation (Must total 100%)")

            overridden = {}
            for ft, pct in ft_dict.items():
                overridden[ft] = st.number_input(
                    f"{ft} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(pct),
                    step=0.5,
                    key=f"ft_{ac}_{ft}"
                )

            ft_total = sum(overridden.values())
            st.write(f"**Total for {ac}: {ft_total}%**")

            if ft_total != 100:
                st.error(f"Fund-type total for {ac} must equal 100%.")

            updated_ft_targets[ac] = overridden

    # Advisor Notes
    with st.expander("Advisor Notes / Comments", expanded=False):
        advisor_notes = st.text_area(
            "Notes for this customer",
            value=saved_notes,
            height=150,
            key="advisor_notes_box"
        )

    # Risk Profile
    with st.expander("Risk Profile", expanded=False):
        risk_options = [
            "Conservative",
            "Moderately Conservative",
            "Moderate",
            "Moderately Aggressive",
            "Aggressive"
        ]

        risk_profile = st.selectbox(
            "Select Customer Risk Profile",
            options=risk_options,
            index=risk_options.index(saved_risk_profile) if saved_risk_profile in risk_options else 2,
            key="risk_profile_box"
        )

    if st.button("Save Model"):
        if ac_total != 100:
            st.error("Cannot save. Asset class total must equal 100%.")
        else:
            valid = True
            for ac, ft_dict in updated_ft_targets.items():
                if sum(ft_dict.values()) != 100:
                    st.error(f"Cannot save. Fund-type total for {ac} must equal 100%.")
                    valid = False

            if valid:
                customer_target_model[customer_key] = {
                    "asset_class_targets": updated_ac_targets,
                    "fund_type_targets": updated_ft_targets,
                    "advisor_notes": advisor_notes,
                    "risk_profile": risk_profile
                }

                with open(customer_model_path, "w") as f:
                    json.dump(customer_target_model, f, indent=4)

                st.success("Model, notes, and risk profile saved successfully!")

    return updated_ac_targets, updated_ft_targets


# ============================================================
# STEP 6 — Breach Summary
# ============================================================

def step6_breach_summary(row, asset_class_targets, fund_type_targets):
    st.header("Step 6 — Breach Summary")

    breach_rows = []

    # Asset Class Breaches
    for ac, data in row["asset_class"].items():
        actual_pct = data["actual_pct"] * 100
        target_pct = asset_class_targets.get(ac, 0)
        diff = actual_pct - target_pct

        status = "OK"
        if diff > 1:
            status = "Overweight"
        elif diff < -1:
            status = "Underweight"

        breach_rows.append({
            "Level": "Asset Class",
            "Category": ac,
            "Actual %": round(actual_pct, 2),
            "Target %": target_pct,
            "Difference": round(diff, 2),
            "Status": status
        })

    # Fund Type Breaches
    for ac, ft_dict in fund_type_targets.items():
        if ac not in row["asset_class"]:
            continue

        ac_total = row["asset_class"][ac]["actual_value"]

        for ft, target_pct in ft_dict.items():
            actual_value = sum(
                f["actual_value"]
                for f in row["funds"].values()
                if f["asset_class"] == ac and f["fund_type"] == ft
            )
            actual_pct = (actual_value / ac_total * 100) if ac_total > 0 else 0
            diff = actual_pct - target_pct

            status = "OK"
            if diff > 1:
                status = "Overweight"
            elif diff < -1:
                status = "Underweight"

            breach_rows.append({
                "Level": "Fund Type",
                "Category": f"{ac} → {ft}",
                "Actual %": round(actual_pct, 2),
                "Target %": target_pct,
                "Difference": round(diff, 2),
                "Status": status
            })

    breach_df = pd.DataFrame(breach_rows)
    st.dataframe(breach_df)

    return breach_df


# ============================================================
# STEP 7 — Redeployment Recommendations
# ============================================================

def step7_redeployment(row, asset_class_targets):
    st.header("Step 7 — Redeployment Recommendations")

    total_value = sum(ac["actual_value"] for ac in row["asset_class"].values())

    rec_rows = []

    for ac, data in row["asset_class"].items():
        actual_value = data["actual_value"]
        target_pct = asset_class_targets.get(ac, 0)
        target_value = total_value * (target_pct / 100)

        diff = target_value - actual_value

        action = "Hold"
        if diff > 1000:
            action = "Buy"
        elif diff < -1000:
            action = "Sell"

        rec_rows.append({
            "Asset Class": ac,
            "Actual Value": format_in_indian(actual_value),
            "Target Value": format_in_indian(target_value),
            "Required Change": format_in_indian(diff),
            "Action": action
        })

    rec_df = pd.DataFrame(rec_rows)
    st.dataframe(rec_df)

    return rec_df


# ============================================================
# STEP 8 — Underperformer Detection
# ============================================================

def step8_underperformers(row, bottom_half_by_type):
    st.header("Step 8 — Underperformer Detection")

    under_rows = []

    for fname, fdata in row["funds"].items():
        ft = fdata["fund_type"]

        if ft in bottom_half_by_type:
            if fname in bottom_half_by_type[ft]:
                under_rows.append({
                    "Fund Name": fname,
                    "Fund Type": ft,
                    "AMC": fdata["AMC"],
                    "Actual Value": format_in_indian(fdata["actual_value"]),
                    "Status": "Bottom 50%",
                    "Action": "Consider Redeployment"
                })

    if under_rows:
        st.dataframe(pd.DataFrame(under_rows))
    else:
        st.success("No underperformers detected.")

    return under_rows


# ============================================================
# MAIN APP
# ============================================================

st.title("Portfolio Analysis")
if "role" not in st.session_state:
    st.write("Please login to access this page.")
    st.stop()

customer_master = load_portfolio_local(PORTFOLIO_PATH)
if customer_master.empty:
    st.stop()

scheme_master = load_scheme_master_local(SCHEME_MASTER_PATH)
if scheme_master.empty:
    st.stop()

# Step 2.5 — select customer
selected_customer_name, filtered_customer_master = select_customer_before_classification(customer_master)

# Step 3 — classify
filtered_customer_master = asyncio.run(
    add_classifications_async(filtered_customer_master, scheme_master)
)

# Existing analysis pipeline
selected_customer_analysis_master = build_customer_analysis(filtered_customer_master)

asset_class_targets_base, fund_type_targets_base = load_and_override_targets(scheme_master)
targets_valid = validate_targets(asset_class_targets_base, fund_type_targets_base)
if not targets_valid:
    st.stop()

selected_customer_analysis_master = update_selected_customer_analysis(
    selected_customer_analysis_master,
    filtered_customer_master,
    asset_class_targets_base,
    fund_type_targets_base
)

# Build row for Steps 5–9
asset_class_summary = {}
total_portfolio_value = filtered_customer_master["CurValue"].sum()

for ac in filtered_customer_master["asset_class"].unique():
    ac_df = filtered_customer_master[filtered_customer_master["asset_class"] == ac]
    total_value = ac_df["CurValue"].sum()
    actual_pct = total_value / total_portfolio_value if total_portfolio_value > 0 else 0

    asset_class_summary[ac] = {
        "actual_value": total_value,
        "actual_pct": actual_pct
    }

funds_dict = {}
for _, r in filtered_customer_master.iterrows():
    funds_dict[r["s_name"]] = {
        "asset_class": r["asset_class"],
        "fund_type": r["fund_type"],
        "fund_class": r["fund_class"],
        "AMC": r["AMC"],
        "actual_value": r["CurValue"]
    }

row = {
    "customer_id": selected_customer_name,
    "asset_class": asset_class_summary,
    "funds": funds_dict
}

# Step 5 — Target Model Editor
asset_class_targets_step5, fund_type_targets_step5 = step5_target_model_editor(row)

# Load ranking data once (for Steps 8 & 9)
ranking_data = {ft: load_ranking_file(path) for ft, path in RANKING_FILES.items()}
bottom_half_by_type = {
    ft: get_bottom_half_funds(df)
    for ft, df in ranking_data.items()
}

# Step 6 — Breach Summary
breach_df = step6_breach_summary(row, asset_class_targets_step5, fund_type_targets_step5)

# Step 7 — Redeployment Recommendations
redeploy_df = step7_redeployment(row, asset_class_targets_step5)

# Step 8 — Underperformer Detection
under_df = step8_underperformers(row, bottom_half_by_type)