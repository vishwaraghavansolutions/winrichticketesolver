import streamlit as st
import pandas as pd
import datetime
import asyncio
from rapidfuzz import process, fuzz
from concurrent.futures import ThreadPoolExecutor
import os
import json

st.set_page_config(page_title="Portfolio Analysis Engine", layout="wide")

# ============================================================
# STEP 1 — Load Portfolio CSV from local file
# ============================================================

PORTFOLIO_PATH = "data/Datawarehouse_MutualFunds_2026_01_01_mutualfunds.csv"

REQUIRED_PORTFOLIO_COLUMNS = [
    "h_name", "c_name", "sCode", "s_name", "foliono", "Nature",
    "FolioStartDate", "BalUnit", "AvgCost", "InvAmt", "TotalInvAmt",
    "CurNAV", "CurValue", "DivAmt", "NotionalGain", "ActualGain",
    "FolioXIRR", "NatureXIRR", "ClientXIRR", "NatureAbs", "ClientAbs",
    "absReturn", "ValueDate", "ReportDate", "Email", "Mobile"
]

SENSITIVE_FIELDS = ["Email", "Mobile"]


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


# ============================================================
# STEP 2 — Load Scheme Master CSV from local file
# ============================================================

SCHEME_MASTER_PATH = "data/SchemeData2301262313SS.csv"

REQUIRED_SCHEME_COLUMNS = [
    "AMC", "Code", "Scheme Name", "Scheme Type", "Scheme Category",
    "Scheme NAV Name", "Scheme Minimum Amount", "Launch Date",
    "Closure Date", "ISIN Div Payout/ ISIN GrowthISIN Div Reinvestment"
]


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

    # Round to nearest rupee
    n = int(round(n))
    s = str(abs(n))
    sign = "-" if n < 0 else ""

    # If <= 3 digits, no special grouping
    if len(s) <= 3:
        return f"{sign}₹{s}"

    # Last 3 digits stay together
    last3 = s[-3:]
    rest = s[:-3]

    # Group rest in 2s
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)

    parts.reverse()
    formatted = ",".join(parts) + "," + last3

    return f"{sign}₹{formatted}"

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

    # Split Scheme Category → fund_class + fund_type
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


# ============================================================
# STEP 2.5 — Select Customer BEFORE classification
# ============================================================

def select_customer_before_classification(customer_master):
    #st.header("Step 2.5: Select Customer")
    st.subheader("Select Customer")

    customers = sorted(customer_master["c_name"].unique())
    selected = st.selectbox("Choose customer", customers)

    filtered = customer_master[customer_master["c_name"] == selected].copy()

    st.write(f"Selected: {selected}")
    return selected, filtered


# ============================================================
# STEP 3 — Async fuzzy matching (Scheme Category–based, includes AMC)
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
        row = scheme_master.iloc[idx]
        asset_class = derive_asset_class_from_category(row["Scheme Category"])
        result = (
            asset_class,
            row["fund_type"],
            row["fund_class"],
            row["AMC"]
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


async def add_classifications_async(filtered_customer_master, scheme_master):
    scheme_names = scheme_master["Scheme Name"].str.lower().str.strip().tolist()
    tasks = [fuzzy_match_async(f, scheme_master, scheme_names) for f in filtered_customer_master["s_name"]]
    results = await asyncio.gather(*tasks)

    filtered_customer_master["asset_class"] = [r[0] for r in results]
    filtered_customer_master["fund_type"] = [r[1] for r in results]
    filtered_customer_master["fund_class"] = [r[2] for r in results]
    filtered_customer_master["AMC"] = [r[3] for r in results]

    return filtered_customer_master


# ============================================================
# STEP 4 — Build Analysis_master (Asset Class Hierarchy)
# ============================================================

def build_customer_analysis(customer_master, analysis_by="XXX"):
    customer_id = customer_master["c_name"].iloc[0]
    analysis_date = datetime.date.today().isoformat()

    total_value = customer_master["CurValue"].sum()
    total_invested = customer_master["TotalInvAmt"].sum()
    if total_value == 0:
        total_value = 1e-9

    # Asset Class
    ac_group = customer_master.groupby("asset_class")["CurValue"].sum()
    asset_class_dict = {
        ac: {
            "name": ac,
            "target_pct": 0,
            "target_value": 0,
            "actual_pct": val / total_value,
            "actual_value": val
        }
        for ac, val in ac_group.items()
    }

    # Fund Type
    ft_group = customer_master.groupby(["fund_type", "asset_class"])["CurValue"].sum()
    fund_type_dict = {
        ft: {
            "name": ft,
            "asset_class": ac,
            "target_pct": 0,
            "target_value": 0,
            "actual_pct": val / total_value,
            "actual_value": val
        }
        for (ft, ac), val in ft_group.items()
    }

    # Funds
    f_group = customer_master.groupby(["s_name", "asset_class", "fund_type"])["CurValue"].sum()
    fund_dict = {
        fname: {
            "name": fname,
            "asset_class": ac,
            "fund_type": ft,
            "target_pct": 0,
            "target_value": 0,
            "actual_pct": val / total_value,
            "actual_value": val
        }
        for (fname, ac, ft), val in f_group.items()
    }

    return pd.DataFrame([{
        "customer_id": customer_id,
        "analysis_date": analysis_date,
        "analysis_by": analysis_by,
        "value": total_value,
        "invested": total_invested,
        "asset_class": asset_class_dict,
        "fund_type": fund_type_dict,
        "funds": fund_dict
    }])


# ============================================================
# STEP 5 — JSON Target Model + UI Override + Validation
# ============================================================

TARGET_JSON_PATH = "data/target_model.json"


def load_target_model(path):
    if not os.path.exists(path):
        st.error(f"Target model JSON not found at: {path}")
        return {}
    with open(path, "r") as f:
        return json.load(f)


def load_and_override_targets(scheme_master):
    #st.header("Step 5: Target Model")

    model = load_target_model(TARGET_JSON_PATH)
    if not model:
        st.stop()

    asset_class_targets = model["asset_class_targets"]
    fund_type_targets_json = model["fund_type_targets"]

    # ------------------------------------------------------------
    # STEP 5 — Editable Target Model (4 Expanders)
    # ------------------------------------------------------------

    # 1) Asset Class Targets
    with st.expander("Edit Asset Class Targets", expanded=False):

        st.subheader("Asset Class Targets")

        updated_asset_class_targets = {}
        for ac, default_pct in asset_class_targets.items():
            updated_asset_class_targets[ac] = st.number_input(
                f"{ac} (%)",
                min_value=0.0,
                max_value=100.0,
                value=float(default_pct),
                step=0.5,
                key=f"asset_class_{ac}"
            )

        st.write(f"Total Asset Class Allocation: **{sum(updated_asset_class_targets.values())}%**")


    # ------------------------------------------------------------
    # 2) Fund-Type Targets — One Expander Per Asset Class
    # ------------------------------------------------------------

    updated_fund_type_targets = {}

    asset_classes = sorted(scheme_master["asset_class"].unique())

    for ac in asset_classes:

        # Skip asset classes that have no fund-type targets in JSON
        if ac not in fund_type_targets_json:
            continue

        with st.expander(f"Edit Fund Types for {ac}", expanded=False):

            st.subheader(f"{ac} — Fund-Type Targets")

            ac_targets = fund_type_targets_json[ac]
            overridden = {}

            for ft, default_pct in ac_targets.items():
                overridden[ft] = st.number_input(
                    f"{ft} (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(default_pct),
                    step=0.5,
                    key=f"{ac}_{ft}"
                )

            updated_fund_type_targets[ac] = overridden
            st.write(f"Total for {ac}: **{sum(overridden.values())}%**")


    return updated_asset_class_targets, updated_fund_type_targets


def validate_targets(asset_class_targets, fund_type_targets):
    st.header("Validation")

    valid = True

    # Validate asset class totals
    ac_total = sum(asset_class_targets.values())
    if abs(ac_total - 100) > 0.01:
        st.error(f"Asset Class totals must sum to 100%. Current: {ac_total}%")
        valid = False

    # Validate fund types under each asset class
    for ac, ft_dict in fund_type_targets.items():
        ft_total = sum(ft_dict.values())
        if abs(ft_total - 100) > 0.01:
            st.error(f"Fund Types under '{ac}' must total 100%. Current: {ft_total}%")
            valid = False

    if valid:
        st.success("All targets validated successfully.")

    return valid


# ============================================================
# STEP 6 — Apply Hierarchical Targets
# ============================================================

def update_selected_customer_analysis(
    analysis_master,
    filtered_customer_master,
    asset_class_targets,
    fund_type_targets
):
    row = analysis_master.iloc[0]

    total_value = filtered_customer_master["CurValue"].sum()
    total_invested = filtered_customer_master["TotalInvAmt"].sum()

    row["value"] = total_value
    row["invested"] = total_invested

    # Asset Class
    ac_group = filtered_customer_master.groupby("asset_class")["CurValue"].sum()
    for ac, actual_value in ac_group.items():
        target_pct = asset_class_targets.get(ac, 0)
        row["asset_class"][ac] = {
            "name": ac,
            "target_pct": target_pct,
            "target_value": (target_pct / 100) * total_value,
            "actual_pct": actual_value / total_value,
            "actual_value": actual_value
        }

    # Fund Type
    ft_group = filtered_customer_master.groupby(["fund_type", "asset_class"])["CurValue"].sum()
    for (ft, ac), actual_value in ft_group.items():
        target_pct = fund_type_targets.get(ac, {}).get(ft, 0)
        row["fund_type"][ft] = {
            "name": ft,
            "asset_class": ac,
            "target_pct": target_pct,
            "target_value": (target_pct / 100) * row["asset_class"][ac]["target_value"],
            "actual_pct": actual_value / total_value,
            "actual_value": actual_value
        }

    # Funds (placeholder target logic)
    f_group = filtered_customer_master.groupby(["s_name", "asset_class", "fund_type"])["CurValue"].sum()
    for (fname, ac, ft), actual_value in f_group.items():
        row["funds"][fname] = {
            "name": fname,
            "asset_class": ac,
            "fund_type": ft,
            "target_pct": 15,
            "target_value": 0.15 * total_value,
            "actual_pct": actual_value / total_value,
            "actual_value": actual_value
        }

    return pd.DataFrame([row])


# ============================================================
# STEP 8 — Ranking helpers (shared with Step 9)
# ============================================================

RANKING_FILES = {
    "large_cap": "data/rankings/large_cap.csv",
    "mid_cap": "data/rankings/mid_cap.csv",
    "small_cap": "data/rankings/small_cap.csv",
    "flexi_cap": "data/rankings/flexi_cap.csv"
}


def load_ranking_file(path):
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Fund Name" not in df.columns or "winrichrank" not in df.columns:
        st.error(f"Ranking file {path} must contain: Fund Name, winrichrank")
        return None

    df = df.sort_values("winrichrank")
    return df


def get_bottom_half_funds(df):
    if df is None or df.empty:
        return set()
    midpoint = len(df) // 2
    return set(df.iloc[midpoint:]["Fund Name"].str.lower().tolist())


def map_fund_type_to_bucket(fund_type: str):
    if not isinstance(fund_type, str):
        return None
    ft = fund_type.strip().lower()

    # Large Cap mapping
    large_cap_types = [
        "large cap fund",
        "large cap",
        "largecap",
        "large cap - regular",
        "large cap - growth"
    ]
    # Mid Cap mapping
    mid_cap_types = [
        "mid cap fund",
        "mid cap",
        "midcap"
    ]
    # Small Cap mapping
    small_cap_types = [
        "small cap fund",
        "small cap",
        "smallcap"
    ]
    # Flexi Cap mapping
    flexi_cap_types = [
        "flexi cap fund",
        "flexi cap",
        "flexicap"
    ]

    if ft in [t.lower() for t in large_cap_types]:
        return "large_cap"
    if ft in [t.lower() for t in mid_cap_types]:
        return "mid_cap"
    if ft in [t.lower() for t in small_cap_types]:
        return "small_cap"
    if ft in [t.lower() for t in flexi_cap_types]:
        return "flexi_cap"

    return None


def color_breach(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v < 0:
        return "color: green; font-weight: bold;"
    if v > 0:
        return "color: red; font-weight: bold;"
    return ""


def detect_breaches(row, bottom_half_by_type):
    breaches = {
        "asset_class": [],
        "fund_type": [],
        "funds": [],
        "ranking_breaches": []
    }

    # Overweight: Asset Class
    for ac, data in row["asset_class"].items():
        target = data["target_pct"]
        actual = data["actual_pct"] * 100
        excess = actual - target

        if actual > target:
            breaches["asset_class"].append({
                "Asset Class": ac,
                "Target %": target,
                "Actual %": actual,
                "Excess %": excess
            })

    # Overweight: Fund Type
    for ft, data in row["fund_type"].items():
        target = data["target_pct"]
        actual = data["actual_pct"] * 100
        excess = actual - target

        if actual > target:
            breaches["fund_type"].append({
                "Fund Type": ft,
                "Asset Class": data["asset_class"],
                "Target %": target,
                "Actual %": actual,
                "Excess %": excess
            })

    # Fund Level Overweight
    for fname, data in row["funds"].items():
        target = data["target_pct"]
        actual = data["actual_pct"] * 100
        excess = actual - target

        if actual > target:
            breaches["funds"].append({
                "Fund Name": fname,
                "Asset Class": data["asset_class"],
                "Fund Type": data["fund_type"],
                "Target %": target,
                "Actual %": actual,
                "Excess %": excess
            })

    # Ranking Breach (Bottom Half by Fund Type bucket)
    for fname, data in row["funds"].items():
        bucket = map_fund_type_to_bucket(data["fund_type"])
        if bucket and bucket in bottom_half_by_type:
            if fname.lower() in bottom_half_by_type[bucket]:
                breaches["ranking_breaches"].append({
                    "Fund Name": fname,
                    "Fund Type": data["fund_type"],
                    "Asset Class": data["asset_class"],
                    "Reason": "Fund is in bottom half of ranking for its category"
                })

    return breaches


# ============================================================
# STEP 9 — Rebalancing helpers
# ============================================================

def compute_asset_class_drift(row):
    records = []
    for ac, data in row["asset_class"].items():
        target_val = data["target_value"]
        actual_val = data["actual_value"]
        diff = actual_val - target_val
        records.append({
            "Asset Class": ac,
            "Target Value": target_val,
            "Actual Value": actual_val,
            "Diff": diff
        })
    df = pd.DataFrame(records)
    df_over = df[df["Diff"] > 0].copy()
    df_under = df[df["Diff"] < 0].copy()
    return df_over, df_under


def compute_fund_type_drift(row):
    records = []
    for ft, data in row["fund_type"].items():
        target_val = data["target_value"]
        actual_val = data["actual_value"]
        diff = actual_val - target_val
        records.append({
            "Fund Type": ft,
            "Asset Class": data["asset_class"],
            "Target Value": target_val,
            "Actual Value": actual_val,
            "Diff": diff
        })
    df = pd.DataFrame(records)
    df_over = df[df["Diff"] > 0].copy()
    df_under = df[df["Diff"] < 0].copy()
    return df_over, df_under


def compute_fund_drift(row):
    records = []
    for fname, data in row["funds"].items():
        target_val = data["target_value"]
        actual_val = data["actual_value"]
        diff = actual_val - target_val
        records.append({
            "Fund Name": fname,
            "Asset Class": data["asset_class"],
            "Fund Type": data["fund_type"],
            "Target Value": target_val,
            "Actual Value": actual_val,
            "Diff": diff
        })
    df = pd.DataFrame(records)
    df_over = df[df["Diff"] > 0].copy()
    df_under = df[df["Diff"] < 0].copy()
    return df_over, df_under


def get_top_30_funds_for_bucket(bucket, ranking_data):
    df = ranking_data.get(bucket)
    if df is None or df.empty:
        return []
    cutoff = max(1, int(len(df) * 0.3))
    return df.iloc[:cutoff]["Fund Name"].str.lower().tolist(), df.iloc[:cutoff].copy()


def allocate_buy_equal(total_amount, funds_list):
    if not funds_list or total_amount <= 0:
        return {}
    n = len(funds_list)
    per = total_amount / n
    return {f: per for f in funds_list}


def allocate_buy_top_only(total_amount, funds_list):
    if not funds_list or total_amount <= 0:
        return {}
    return {funds_list[0]: total_amount}


def allocate_buy_proportional(total_amount, ranking_df):
    if ranking_df is None or ranking_df.empty or total_amount <= 0:
        return {}
    # Use inverse rank as weight (rank 1 gets highest weight)
    ranking_df = ranking_df.copy()
    ranking_df["weight"] = 1 / ranking_df["winrichrank"].astype(float)
    total_w = ranking_df["weight"].sum()
    if total_w == 0:
        return {}
    alloc = {}
    for _, r in ranking_df.iterrows():
        amt = total_amount * (r["weight"] / total_w)
        alloc[r["Fund Name"].lower()] = amt
    return alloc


def is_equity_asset_class(ac_name: str):
    return isinstance(ac_name, str) and ac_name.lower() == "equity"


def build_net_cashflow_summary(sell_df, buy_df):
    total_sell = sell_df["Amount"].sum() if not sell_df.empty else 0.0
    total_buy = buy_df["Amount"].sum() if not buy_df.empty else 0.0
    net = total_sell - total_buy
    return pd.DataFrame([
        {"Category": "Total Sell", "Amount": total_sell},
        {"Category": "Total Buy", "Amount": total_buy},
        {"Category": "Net Difference (Sell - Buy)", "Amount": net},
    ])


# ============================================================
# MAIN APP LAYOUT WITH TABS
# ============================================================

st.title("Portfolio Analysis")
if "role" not in st.session_state:
    st.write("Please login to access this page.")
    st.stop

customer_master = load_portfolio_local(PORTFOLIO_PATH)
if customer_master.empty:
    st.stop()

scheme_master = load_scheme_master_local(SCHEME_MASTER_PATH)
if scheme_master.empty:
    st.stop()

selected_customer_name, filtered_customer_master = (
    select_customer_before_classification(customer_master)
)

filtered_customer_master = asyncio.run(
    add_classifications_async(filtered_customer_master, scheme_master)
)

selected_customer_analysis_master = build_customer_analysis(filtered_customer_master)

asset_class_targets, fund_type_targets = load_and_override_targets(scheme_master)
targets_valid = validate_targets(asset_class_targets, fund_type_targets)
if not targets_valid:
    st.stop()

selected_customer_analysis_master = update_selected_customer_analysis(
    selected_customer_analysis_master,
    filtered_customer_master,
    asset_class_targets,
    fund_type_targets
)

row = selected_customer_analysis_master.iloc[0]

# Load ranking data once (for Steps 8 & 9)
ranking_data = {ft: load_ranking_file(path) for ft, path in RANKING_FILES.items()}
bottom_half_by_type = {
    ft: get_bottom_half_funds(df)
    for ft, df in ranking_data.items()
}


tab_analysis, tab_rebalancing = st.tabs(["Analysis", "Rebalancing"])

# ============================================================
# ANALYSIS TAB — Steps 7 & 8
# ============================================================

with tab_analysis:
    #st.header("Step 7: Customer Summary")
    st.subheader(f"Customer Summary: {selected_customer_name}")

    # ------------------------------------------------------------
    # PORTFOLIO SUMMARY (Place this BEFORE Asset Class Summary)
    # ------------------------------------------------------------
    st.subheader("Portfolio Summary")

    # Total portfolio value
    total_value = sum(data["actual_value"] for data in row["asset_class"].values())

    # Number of asset classes
    num_asset_classes = len(row["asset_class"])

    # Number of funds
    num_funds = len(row["funds"])

    # Largest & smallest asset class
    ac_values = {ac: data["actual_value"] for ac, data in row["asset_class"].items()}
    largest_ac = max(ac_values, key=ac_values.get)
    smallest_ac = min(ac_values, key=ac_values.get)

    # Largest & smallest fund
    fund_values = {fname: data["actual_value"] for fname, data in row["funds"].items()}
    largest_fund = max(fund_values, key=fund_values.get)
    smallest_fund = min(fund_values, key=fund_values.get)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Portfolio Value", format_in_indian(total_value))
        st.metric("Asset Classes", num_asset_classes)

    with col2:
        st.metric("Total Funds", num_funds)
        st.metric("Largest Asset Class", f"{largest_ac} {format_in_indian(ac_values[largest_ac])}")

    with col3:
        st.metric("Largest Fund", f"{largest_fund} (₹{fund_values[largest_fund]:,.0f})")
        st.metric("Smallest Fund", f"{smallest_fund} (₹{fund_values[smallest_fund]:,.0f})")

    st.markdown("---")
    # Asset Class Summary
    st.subheader("Asset Class Summary")

    ac_rows = []
    for ac, data in row["asset_class"].items():
        ac_rows.append({
            "Asset Class": ac,
            "Target %": data["target_pct"],
            "Target Value": data["target_value"],
            "Actual %": data["actual_pct"] * 100,
            "Actual Value": data["actual_value"]
        })

    df_ac = pd.DataFrame(ac_rows)
    st.dataframe(
        df_ac.style.format({
            "Target %": "{:.2f}%",
            "Actual %": "{:.2f}%",
            "Target Value": format_in_indian,
            "Actual Value": format_in_indian
        }),
        use_container_width=True
    )

    # Fund Type Summary
    st.subheader("Fund Type Summary")

    ft_rows = []
    for ft, data in row["fund_type"].items():
        ft_rows.append({
            "Fund Type": ft,
            "Asset Class": data["asset_class"],
            "Target %": data["target_pct"],
            "Target Value": data["target_value"],
            "Actual %": data["actual_pct"] * 100,
            "Actual Value": data["actual_value"]
        })

    df_ft = pd.DataFrame(ft_rows)
    st.dataframe(
        df_ft.style.format({
            "Target %": "{:.2f}%",
            "Actual %": "{:.2f}%",
            "Target Value": format_in_indian,
            "Actual Value": format_in_indian
        }),
        use_container_width=True
    )

    # Fund Summary
    st.subheader("Fund Summary")

    fund_rows = []
    for fname, data in row["funds"].items():
        fund_rows.append({
            "Fund Name": fname,
            "Asset Class": data["asset_class"],
            "Fund Type": data["fund_type"],
            "Target %": data["target_pct"],
            "Target Value": data["target_value"],
            "Actual %": data["actual_pct"] * 100,
            "Actual Value": data["actual_value"]
        })

    df_funds = pd.DataFrame(fund_rows)
    st.dataframe(
        df_funds.style.format({
            "Target %": "{:.2f}%",
            "Actual %": "{:.2f}%",
            "Target Value": format_in_indian,
            "Actual Value": format_in_indian
        }),
        use_container_width=True
    )

    # Step 8 — Breach Detection
    #st.header("Step 8: Breach Detection")
    st.subheader("Breach Detection")

    breaches = detect_breaches(row, bottom_half_by_type)

    st.subheader("Asset Class that exceed Targets")
    if breaches["asset_class"]:
        df = pd.DataFrame(breaches["asset_class"])
        df["Excess %"] = df["Excess %"].round(2)
        st.dataframe(
            df.style
            .applymap(color_breach, subset=["Excess %"])
            .format({
                "Target %": "{:.2f}%",
                "Actual %": "{:.2f}%",
                "Excess %": "{:.2f}%"
            }),
            use_container_width=True
        )
    else:
        st.success("No asset class breaches detected.")

    st.subheader("Fund Type that exceed Targets")
    if breaches["fund_type"]:
        df = pd.DataFrame(breaches["fund_type"])
        df["Excess %"] = df["Excess %"].round(2)
        st.dataframe(
            df.style
            .applymap(color_breach, subset=["Excess %"])
            .format({
                "Target %": "{:.2f}%",
                "Actual %": "{:.2f}%",
                "Excess %": "{:.2f}%"
            }),
            use_container_width=True
        )
    else:
        st.success("No fund-type breaches detected.")

    st.subheader("Funds that have more than 15% allocation or exceed Targets")
    if breaches["funds"]:
        df = pd.DataFrame(breaches["funds"])
        df["Excess %"] = df["Excess %"].round(2)
        st.dataframe(
            df.style
            .applymap(color_breach, subset=["Excess %"])
            .format({
                "Target %": "{:.2f}%",
                "Actual %": "{:.2f}%",
                "Excess %": "{:.2f}%"
            }),
            use_container_width=True
        )
    else:
        st.success("No fund-level overweight breaches detected.")

    st.subheader("Fund Ranking Breaches (Bottom Half by Category)")
    if breaches["ranking_breaches"]:
        st.dataframe(pd.DataFrame(breaches["ranking_breaches"]), use_container_width=True)
    else:
        st.success("No ranking breaches detected.")

# ============================================================
# REBALANCING TAB — Step 9
# ============================================================

# ============================================================
# STEP 9 — Rebalancing Engine (Updated with Fund-Level Buy/Sell)
# ============================================================

with tab_rebalancing:
    #st.header("Step 9: Rebalancing Engine --- to be build")
    st.subheader("Rebalancing Engine --- to be built")

    # mode = st.radio(
    #     "Choose Rebalancing Mode",
    #     [
    #         "A: Asset Class Rebalancing",
    #         "B: Fund Type Rebalancing",
    #         "C: Fund Level Rebalancing (Overweight + Ranking)",
    #         "D: Ranking-Only Rebalancing (Exit bottom → reinvest top)"
    #     ]
    # )

    # buy_strategy = None
    # if mode.startswith("C") or mode.startswith("D"):
    #     buy_strategy = st.radio(
    #         "Choose Buy Allocation Strategy",
    #         [
    #             "1: Equal Split",
    #             "2: Proportional to Ranking",
    #             "3: Top-Ranked Only"
    #         ]
    #     )

    # # --------------------------------------------------------
    # # Helper: SELL distribution (S-A + S-B)
    # # --------------------------------------------------------
    # def distribute_sell_to_funds(bucket_funds, bucket_over_amt):
    #     """
    #     bucket_funds: list of dicts with keys:
    #         - name
    #         - actual_value
    #         - target_value
    #     bucket_over_amt: total amount to sell at bucket level

    #     Rules:
    #     S-A: If any fund is overweight → sell proportional to overweight
    #     S-B: Else → sell proportional to fund weights
    #     """
    #     df = pd.DataFrame(bucket_funds)

    #     # Compute overweight per fund
    #     df["over"] = df["actual_value"] - df["target_value"]
    #     df_over = df[df["over"] > 0]

    #     allocations = {}

    #     if not df_over.empty:
    #         # S-A: proportional to overweight
    #         total_over = df_over["over"].sum()
    #         for _, r in df_over.iterrows():
    #             amt = bucket_over_amt * (r["over"] / total_over)
    #             allocations[r["name"]] = amt
    #     else:
    #         # S-B: proportional to fund weights
    #         total_val = df["actual_value"].sum()
    #         for _, r in df.iterrows():
    #             amt = bucket_over_amt * (r["actual_value"] / total_val)
    #             allocations[r["name"]] = amt

    #     return allocations

    # # --------------------------------------------------------
    # # Helper: BUY distribution for Options A & B
    # # --------------------------------------------------------
    # def distribute_buy_to_funds(ac, ft, bucket_buy_amt):
    #     """
    #     ac: asset class
    #     ft: fund type (may be None for asset class mode)
    #     bucket_buy_amt: total amount to buy at bucket level
    #     """
    #     # Get funds inside this bucket
    #     funds = []
    #     for fname, data in row["funds"].items():
    #         if data["asset_class"] == ac and (ft is None or data["fund_type"] == ft):
    #             funds.append(fname)

    #     if not funds:
    #         return {}

    #     # Equity → ranking-based buy
    #     if is_equity_asset_class(ac):
    #         bucket = map_fund_type_to_bucket(ft) if ft else None
    #         if bucket:
    #             top_funds, top_df = get_top_30_funds_for_bucket(bucket, ranking_data)
    #             if top_funds:
    #                 if buy_strategy and buy_strategy.startswith("1"):
    #                     return allocate_buy_equal(bucket_buy_amt, top_funds)
    #                 elif buy_strategy and buy_strategy.startswith("2"):
    #                     return allocate_buy_proportional(bucket_buy_amt, top_df)
    #                 elif buy_strategy and buy_strategy.startswith("3"):
    #                     return allocate_buy_top_only(bucket_buy_amt, top_funds)

    #     # Non-equity → equal split
    #     per = bucket_buy_amt / len(funds)
    #     return {f.lower(): per for f in funds}

    # # --------------------------------------------------------
    # # Rebalancing Logic
    # # --------------------------------------------------------
    # sell_recs = []
    # buy_recs = []

    # # ------------------------------------------------------------
    # # MODE A — ASSET CLASS REBALANCING
    # # ------------------------------------------------------------
    # if mode.startswith("A"):

    #     # Compute over/under allocation at asset-class level
    #     df_over, df_under = compute_asset_class_drift(row)

    #     # --------------------------------------------------------
    #     # SELL LOGIC — proportional inside asset class
    #     # --------------------------------------------------------
    #     for _, r in df_over.iterrows():
    #         ac = r["Asset Class"]
    #         amt = r["Diff"]   # negative number

    #         # Collect all funds belonging to this asset class
    #         bucket_funds = []
    #         for fname, data in row["funds"].items():
    #             if data["asset_class"] == ac:
    #                 bucket_funds.append({
    #                     "name": fname,
    #                     "fund_type": data["fund_type"],
    #                     "actual_value": data["actual_value"],
    #                     "target_value": data["target_value"]
    #                 })

    #         # SELL allocation (proportional)
    #         alloc = distribute_sell_to_funds(bucket_funds, abs(amt))

    #         # Build SELL records
    #         for fname, a in alloc.items():
    #             sell_recs.append({
    #                 "Level": "Fund",
    #                 "Asset Class": ac,
    #                 "Fund Type": row["funds"][fname]["fund_type"],
    #                 "Fund Name": fname,
    #                 "Amount": a
    #             })

    #     # --------------------------------------------------------
    #     # BUY LOGIC — ranking-aware inside asset class
    #     # --------------------------------------------------------
    #     for _, r in df_under.iterrows():
    #         ac = r["Asset Class"]
    #         amt = abs(r["Diff"])   # positive number

    #         # Collect all funds belonging to this asset class
    #         bucket_funds = []
    #         for fname, data in row["funds"].items():
    #             if data["asset_class"] == ac:
    #                 bucket_funds.append({
    #                     "name": fname,
    #                     "fund_type": data["fund_type"],
    #                     "actual_value": data["actual_value"],
    #                     "target_value": data["target_value"]
    #                 })

    #         # BUY allocation (ranking-aware)
    #         alloc = distribute_buy_to_funds(ac,bucket_funds,amt)

    #         # Build BUY records
    #         for fname, a in alloc.items():
    #             buy_recs.append({
    #                 "Level": "Fund",
    #                 "Asset Class": ac,
    #                 "Fund Type": row["funds"][fname]["fund_type"],
    #                 "Fund Name": fname,
    #                 "Amount": a
    #             })

    # # -----------------------------
    # # Mode B: Fund Type Rebalancing
    # # -----------------------------
    # elif mode.startswith("B"):
    #     df_over, df_under = compute_fund_type_drift(row)

    #     # SELL: distribute to funds inside fund type
    #     for _, r in df_over.iterrows():
    #         ac = r["Asset Class"]
    #         ft = r["Fund Type"]
    #         amt = r["Diff"]

    #         bucket_funds = []
    #         for fname, data in row["funds"].items():
    #             if data["asset_class"] == ac and data["fund_type"] == ft:
    #                 bucket_funds.append({
    #                     "name": fname,
    #                     "actual_value": data["actual_value"],
    #                     "target_value": data["target_value"]
    #                 })

    #         alloc = distribute_sell_to_funds(bucket_funds, amt)
    #         for fname, a in alloc.items():
    #             sell_recs.append({
    #                 "Level": "Fund",
    #                 "Asset Class": ac,
    #                 "Fund Type": ft,
    #                 "Fund Name": fname,
    #                 "Amount": a
    #             })

    #     # BUY: distribute to funds inside fund type
    #     for _, r in df_under.iterrows():
    #         ac = r["Asset Class"]
    #         ft = r["Fund Type"]
    #         amt = abs(r["Diff"])

    #         alloc = distribute_buy_to_funds(ac, ft, amt)
    #         for fname, a in alloc.items():
    #             buy_recs.append({
    #                 "Level": "Fund",
    #                 "Asset Class": ac,
    #                 "Fund Type": ft,
    #                 "Fund Name": fname,
    #                 "Amount": a
    #             })

    # # -----------------------------
    # # Mode C: Fund Level Rebalancing (Overweight + Ranking)
    # # -----------------------------
    # elif mode.startswith("C"):
    #     df_over, df_under = compute_fund_drift(row)

    #     # SELL overweight funds
    #     for _, r in df_over.iterrows():
    #         sell_recs.append({
    #             "Level": "Fund",
    #             "Asset Class": r["Asset Class"],
    #             "Fund Type": r["Fund Type"],
    #             "Fund Name": r["Fund Name"],
    #             "Amount": r["Diff"]
    #         })

    #     # BUY: redirect underweight to top 30% funds
    #     demand = {}
    #     for _, r in df_under.iterrows():
    #         key = (r["Asset Class"], r["Fund Type"])
    #         demand[key] = demand.get(key, 0) + abs(r["Diff"])

    #     for (ac, ft), amt in demand.items():
    #         alloc = distribute_buy_to_funds(ac, ft, amt)
    #         for fname, a in alloc.items():
    #             buy_recs.append({
    #                 "Level": "Fund",
    #                 "Asset Class": ac,
    #                 "Fund Type": ft,
    #                 "Fund Name": fname,
    #                 "Amount": a
    #             })

    # # -----------------------------
    # # Mode D: Ranking-Only Rebalancing
    # # -----------------------------
    # elif mode.startswith("D"):
    #     df_over, _ = compute_fund_drift(row)

    #     # SELL: only overweight portion of bottom-ranked funds
    #     for _, r in df_over.iterrows():
    #         fname = r["Fund Name"]
    #         ft = r["Fund Type"]
    #         ac = r["Asset Class"]
    #         diff = r["Diff"]

    #         bucket = map_fund_type_to_bucket(ft)
    #         if bucket and fname.lower() in bottom_half_by_type.get(bucket, set()):
    #             sell_recs.append({
    #                 "Level": "Fund",
    #                 "Asset Class": ac,
    #                 "Fund Type": ft,
    #                 "Fund Name": fname,
    #                 "Amount": diff
    #             })

    #     # BUY: reinvest into top 30% funds within same asset class
    #     sell_df = pd.DataFrame(sell_recs)
    #     if not sell_df.empty:
    #         grouped = sell_df.groupby(["Asset Class", "Fund Type"])["Amount"].sum().reset_index()
    #         for _, r in grouped.iterrows():
    #             ac = r["Asset Class"]
    #             ft = r["Fund Type"]
    #             amt = r["Amount"]

    #             alloc = distribute_buy_to_funds(ac, ft, amt)
    #             for fname, a in alloc.items():
    #                 buy_recs.append({
    #                     "Level": "Fund",
    #                     "Asset Class": ac,
    #                     "Fund Type": ft,
    #                     "Fund Name": fname,
    #                     "Amount": a
    #                 })

    # # --------------------------------------------------------
    # # Display Results
    # # --------------------------------------------------------
    # st.subheader("Sell Recommendations")
    # sell_df = pd.DataFrame(sell_recs)
    # if not sell_df.empty:
    #     st.dataframe(sell_df.style.format({"Amount": "₹{:,.0f}"}), use_container_width=True)
    # else:
    #     st.info("No sell recommendations.")

    # st.subheader("Buy Recommendations")
    # buy_df = pd.DataFrame(buy_recs)
    # if not buy_df.empty:
    #     st.dataframe(buy_df.style.format({"Amount": "₹{:,.0f}"}), use_container_width=True)
    # else:
    #     st.info("No buy recommendations.")

    # st.subheader("Net Cashflow Summary")
    # summary_df = build_net_cashflow_summary(sell_df, buy_df)
    # st.dataframe(summary_df.style.format({"Amount": "₹{:,.0f}"}), use_container_width=True)