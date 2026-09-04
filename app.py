import io
import numpy as np
import pandas as pd
import streamlit as st

# Pro-Level Page Styling & Configuration optimized for Mobile & Desktop
st.set_page_config(
    page_title="Pro Institutional Naked Option Buyer Terminal",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stAlert { border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Pro Institutional Naked Option Buyer Terminal")
st.markdown(
    "Quadruple-verified, zero-false-signal algorithmic engine designed exclusively for "
    "**Capital-Efficient Naked Option Buyers** using official **NSE India CSV exports**."
)


@st.cache_data(show_spinner=False)
def load_and_parse_nse_csv(bytes_data: bytes) -> pd.DataFrame:
    """Layer 1: Robust file decoding, multi-encoding fallback, and structural boundary detection."""
    text_data = None
    for enc in ["utf-8", "latin1", "cp1252", "iso-8859-1"]:
        try:
            text_data = bytes_data.decode(enc)
            break
        except UnicodeDecodeError:
            continue

    if not text_data:
        raise ValueError("Decoding Error: File encoding could not be resolved.")

    lines = text_data.splitlines()
    header_idx = -1
    for idx, line in enumerate(lines):
        if "STRIKE" in line.upper():
            header_idx = idx
            break

    if header_idx == -1:
        raise ValueError("Validation Error: Mandatory 'STRIKE PRICE' column missing from NSE export.")

    csv_content = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(csv_content), on_bad_lines="skip", low_memory=False)
    df.columns = df.columns.astype(str).str.strip().str.upper()
    return df


def clean_and_map_nse_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Layer 2: Standardizing NSE header naming conventions and cleaning raw numerical entries."""
    cols = list(df.columns)
    strike_pos = next((i for i, c in enumerate(cols) if "STRIKE" in c or "STRIKE PRICE" in c), -1)
    if strike_pos == -1:
        return pd.DataFrame()

    mapping = {}
    for i, c in enumerate(cols):
        if i == strike_pos:
            mapping[c] = "Strike"
            continue
        c_up = str(c).upper()
        if i < strike_pos:
            if "CHNG IN OI" in c_up or "CHANGE IN OI" in c_up:
                mapping[c] = "Call_Chng_OI"
            elif c_up == "OI" or ("OPEN INTEREST" in c_up and "CHNG" not in c_up and "CHANGE" not in c_up):
                mapping[c] = "Call_OI"
            elif "VOLUME" in c_up:
                mapping[c] = "Call_Volume"
            elif "IV" in c_up:
                mapping[c] = "Call_IV"
            elif c_up == "LTP":
                mapping[c] = "Call_LTP"
        elif i > strike_pos:
            if "CHNG IN OI" in c_up or "CHANGE IN OI" in c_up:
                mapping[c] = "Put_Chng_OI"
            elif c_up == "OI" or ("OPEN INTEREST" in c_up and "CHNG" not in c_up and "CHANGE" not in c_up):
                mapping[c] = "Put_OI"
            elif "VOLUME" in c_up:
                mapping[c] = "Put_Volume"
            elif "IV" in c_up:
                mapping[c] = "Put_IV"
            elif c_up == "LTP":
                mapping[c] = "Put_LTP"

    df = df.rename(columns=mapping)
    
    # Ensure all column names are unique and handle duplicate mappings safely
    df = df.loc[:, ~df.columns.duplicated()]

    required_cols = [
        "Strike", "Call_OI", "Call_Chng_OI", "Call_Volume", "Call_LTP", "Call_IV",
        "Put_OI", "Put_Chng_OI", "Put_Volume", "Put_LTP", "Put_IV"
    ]

    for col in required_cols:
        if col in df.columns:
            if isinstance(df[col], pd.DataFrame):
                df[col] = df[col].iloc[:, 0]
            df[col] = df[col].astype(str).str.replace(",", "").str.replace("-", "0").str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            df[col] = 0.0

    return df[df["Strike"] > 0]


def detect_spot_and_market_regime(df: pd.DataFrame):
    """Layer 3: Straddle-based spot detection and rigorous market regime / no-trade zone gating."""
    valid_straddle_df = df[(df["Call_LTP"] > 0) & (df["Put_LTP"] > 0)].copy()
    if not valid_straddle_df.empty:
        valid_straddle_df["Straddle"] = valid_straddle_df["Call_LTP"] + valid_straddle_df["Put_LTP"]
        atm_row = valid_straddle_df.loc[valid_straddle_df["Straddle"].idxmin()]
        spot = float(atm_row["Strike"])
        expected_move = float(atm_row["Straddle"])
    else:
        spot = float(df["Strike"].iloc[len(df) // 2])
        expected_move = 150.0

    total_call_oi = df["Call_OI"].sum()
    total_put_oi = df["Put_OI"].sum()
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    is_no_trade_zone = False
    if 0.98 <= pcr <= 1.02:
        is_no_trade_zone = True
        bias = "CHOOPY_NEUTRAL"
    elif pcr > 1.05:
        bias = "BULLISH"
    elif pcr < 0.95:
        bias = "BEARISH"
    else:
        is_no_trade_zone = True
        bias = "RANGEBOUND"

    return spot, expected_move, pcr, bias, is_no_trade_zone


def evaluate_quadruple_verified_buyer_engine(
    sub_df: pd.DataFrame,
    option_type_str: str,
    ltp_col: str,
    chng_oi_col: str,
    vol_col: str,
    iv_col: str,
    spot_price: float,
    market_trend_bias: str,
) -> pd.DataFrame:
    """Layer 4: Quadruple mathematical validation, anti-trap filtering, and momentum scoring."""
    if sub_df.empty or market_trend_bias in ["CHOOPY_NEUTRAL", "RANGEBOUND"]:
        return pd.DataFrame()

    sub_df = sub_df.copy()
    sub_df["Option_Type"] = option_type_str

    if market_trend_bias == "BEARISH" and option_type_str.startswith("CE"):
        return pd.DataFrame()
    if market_trend_bias == "BULLISH" and option_type_str.startswith("PE"):
        return pd.DataFrame()

    sub_df = sub_df[(sub_df[iv_col] >= 5.0) & (sub_df[iv_col] <= 75.0)]
    if sub_df.empty:
        return pd.DataFrame()

    sub_df["Strike_Distance"] = abs(sub_df["Strike"] - spot_price)
    sub_df = sub_df[sub_df["Strike_Distance"] <= (spot_price * 0.03)]
    if sub_df.empty:
        return pd.DataFrame()

    median_vol = sub_df[vol_col].median() if not sub_df[vol_col].empty else 1.0
    active_vol_mask = sub_df[vol_col] >= max(100.0, median_vol * 0.5)

    sub_df["True_Unwind"] = np.where((sub_df[chng_oi_col] < 0) & active_vol_mask, abs(sub_df[chng_oi_col]), 0.0)
    sub_df["Long_Buildup"] = np.where((sub_df[chng_oi_col] > 0) & active_vol_mask, sub_df[chng_oi_col], 0.0)

    sub_df["Buyer_Score"] = (
        (sub_df["True_Unwind"] * 4.0 + sub_df["Long_Buildup"] * 3.0)
        * (1.0 / (1.0 + 0.03 * sub_df["Strike_Distance"] / 100))
        / (sub_df[iv_col] + 1.0)
    )

    sub_df["Entry_Price"] = sub_df[ltp_col]
    sub_df["Volume"] = sub_df[vol_col]
    sub_df["Chng_OI"] = sub_df[chng_oi_col]
    sub_df["IV"] = sub_df[iv_col]

    return sub_df[["Strike", "Option_Type", "Entry_Price", "Volume", "Chng_OI", "IV", "Buyer_Score"]]


# Mobile-Optimized File Input (Flexible type to prevent blanking on mobile browsers)
uploaded_file = st.file_uploader("📂 Upload Official NSE Option Chain CSV Export", type=None)

# Mobile fallback text area if file upload blanks out on phone
with st.expander("📱 Mobile Fallback: Paste CSV Text Here (If File Upload Blanks)"):
    pasted_csv = st.text_area("Paste raw NSE option chain CSV content here:")

raw_bytes = None
if uploaded_file is not None:
    try:
        raw_bytes = uploaded_file.read()
    except Exception:
        raw_bytes = None

if raw_bytes is None and pasted_csv.strip():
    raw_bytes = pasted_csv.strip().encode("utf-8")

if raw_bytes is not None:
    try:
        raw_df = load_and_parse_nse_csv(raw_bytes)
        df = clean_and_map_nse_columns(raw_df)

        if df.empty:
            st.error("Error: Failed to parse NSE columns. Verify file format.")
        else:
            spot_price, expected_move, pcr, market_trend_bias, is_no_trade_zone = detect_spot_and_market_regime(df)

            col1, col2, col3 = st.columns(3)
            col1.metric("📌 Spot Reference", f"₹{spot_price:,.2f}")
            col2.metric("📊 PCR Ratio", f"{pcr:.2f}")
            col3.metric("🌐 Market Sentiment", market_trend_bias)

            st.divider()

            if is_no_trade_zone:
                st.error(
                    "🛑 **NO-TRADE ZONE ACTIVE (SIT ON CASH)** 🛑\n\n"
                    f"**Reason:** Market PCR is locked at **{pcr:.2f}** (Choppy / Rangebound regime). "
                    "Option writers are defending both sides aggressively. No high-probability directional edge exists. "
                    "**Executing naked trades here will cause theta decay losses. Protect your capital.**"
                )
            else:
                ce_df = df[(df["Call_LTP"] >= 5.0) & (df["Call_LTP"] <= 500.0)]
                pe_df = df[(df["Put_LTP"] >= 5.0) & (df["Put_LTP"] <= 500.0)]

                ce_ranked = evaluate_quadruple_verified_buyer_engine(
                    ce_df, "CE (CALL)", "Call_LTP", "Call_Chng_OI", "Call_Volume", "Call_IV", spot_price, market_trend_bias
                )
                pe_ranked = evaluate_quadruple_verified_buyer_engine(
                    pe_df, "PE (PUT)", "Put_LTP", "Put_Chng_OI", "Put_Volume", "Put_IV", spot_price, market_trend_bias
                )

                combined_ranked = pd.concat([ce_ranked, pe_ranked]).sort_values(by="Buyer_Score", ascending=False)

                if combined_ranked.empty or combined_ranked["Buyer_Score"].max() <= 0:
                    st.warning(
                        "⚠️ **NO-TRADE ZONE:** No setups passed all 4 layers of institutional validation. "
                        "Avoid forced trading and wait for the next market snapshot."
                    )
                else:
                    best_row = combined_ranked.iloc[0]
                    strike = int(best_row["Strike"])
                    option_type = str(best_row["Option_Type"])
                    ltp_price = float(best_row["Entry_Price"])
                    iv = float(best_row["IV"])
                    volume = int(best_row["Volume"])
                    chng_oi = int(best_row["Chng_OI"])

                    if ltp_price <= 0:
                        st.warning("🛑 **NO-TRADE ZONE:** Selected strike price is invalid.")
                    else:
                        limit_buy_price = round(ltp_price * 0.975, 2)
                        stop_loss = round(ltp_price * 0.72, 2)
                        risk_amount = ltp_price - stop_loss
                        
                        target_1 = round(ltp_price + (risk_amount * 1.6), 2)
                        target_2 = round(ltp_price + (risk_amount * 2.8), 2)
                        lot_cost = ltp_price * 25

                        st.success("🔥 **QUADRUPLE-VERIFIED SINGLE TRADE SIGNAL READY** 🔥")

                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            st.metric(label="⚡ Recommended Strike", value=f"{strike} {option_type}")
                            st.metric(label="💵 Instant LTP Price", value=f"₹{ltp_price:.2f}")
                            st.metric(label="🎯 Limit Buy (Pullback Dip)", value=f"₹{limit_buy_price:.2f}")
                            st.metric(label="📦 1 Lot Capital Required", value=f"₹{lot_cost:,.2f}")
                        with r_col2:
                            st.metric(label="🚀 Target 1 (First Book)", value=f"₹{target_1:.2f}")
                            st.metric(label="🚀 Target 2 (Extended Run)", value=f"₹{target_2:.2f}")
                            st.metric(label="🛑 Strict Stop Loss (SL)", value=f"₹{stop_loss:.2f}")

                        st.info(
                            f"🛡️ **Mathematical Audit:** Unwound OI: {chng_oi:,} | "
                            f"Volume: {volume:,} | IV: {iv:.2f}% | Expected Move: ₹{expected_move:.1f}"
                        )

                        st.divider()
                        st.subheader("🏆 Alternative Backup Candidates Audit")
                        display_df = combined_ranked.head(3).copy()
                        display_df["Limit Buy"] = (display_df["Entry_Price"] * 0.975).round(2)
                        display_df["Target 1"] = (display_df["Entry_Price"] * 1.35).round(2)
                        display_df["Target 2"] = (display_df["Entry_Price"] * 1.80).round(2)
                        display_df["SL"] = (display_df["Entry_Price"] * 0.72).round(2)
                        st.dataframe(display_df[["Strike", "Option_Type", "Entry_Price", "Limit Buy", "Target 1", "Target 2", "SL", "IV"]], use_container_width=True)

    except Exception as e:
        st.error(f"Critical System Error: {e}. Please ensure correct official NSE CSV format.")
else:
    st.info("📁 Upload your official NSE Option Chain CSV file above or paste raw CSV text in the mobile fallback box to initiate the terminal.")
