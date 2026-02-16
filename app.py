import io
import re
import unicodedata
from typing import List, Optional, Tuple, Dict

import pandas as pd
import streamlit as st
import pdfplumber

# PDF generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# ----------------------------
# Text normalization helpers
# ----------------------------

def nfkc(s: str) -> str:
    """Normalize Arabic presentation forms into base letters (very important for your PDFs)."""
    return unicodedata.normalize("NFKC", s or "")

def reversed_text(s: str) -> str:
    """Reverse text for cases where extracted Arabic appears reversed in logical order."""
    return (s or "")[::-1]


# ----------------------------
# Numeric parsing
# ----------------------------

NUM_RE = re.compile(r"-?[\d]{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")

def parse_number_from_line(line: str) -> Optional[float]:
    """
    Extract a number from a line (handles commas). Uses the last number in the line,
    which works for both:
      'النقدية 17,371,249'
      '17,371,249 النقدية'
    """
    matches = NUM_RE.findall(line)
    if not matches:
        return None
    token = matches[-1].replace(",", "")
    try:
        return float(token)
    except:
        return None


# ----------------------------
# PDF text extraction
# ----------------------------

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    texts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                texts.append(txt)
    return "\n".join(texts)


# ----------------------------
# Group name
# ----------------------------

def find_group_name(text: str, fallback: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].lower().startswith("group"):
        return lines[0]
    for l in reversed(lines):
        if l.lower().startswith("group"):
            return l
    m = re.search(r"\bGroup\s*\d+\b.*", text, flags=re.IGNORECASE)
    return (m.group(0).strip() if m else fallback)


# ----------------------------
# Cash & NAV detection (FIXED)
# ----------------------------

# We match *multiple* variants because your PDF extraction returns:
# 'ﺔﻳﺪﻘﻨﻟﺍ' -> nfkc -> 'ةيدقنلا'
# 'ﻆﻓﺎﺤﻤﻟﺍ ﻲﻟﺎﻤﺟﺍ' -> nfkc -> 'ظفاحملا يلامجا'
CASH_KEYS = {"النقدية", "ةيدقنلا"}  # include normalized extracted form
NAV_KEYS  = {("اجمالي", "المحافظ"), ("يلامجا", "ظفاحملا")}  # both normal & extracted-normalized

def line_has_cash(line: str) -> bool:
    n = nfkc(line)
    r = nfkc(reversed_text(line))
    return any(k in n for k in CASH_KEYS) or any(k in r for k in CASH_KEYS)

def line_has_nav(line: str) -> bool:
    n = nfkc(line)
    r = nfkc(reversed_text(line))
    for a, b in NAV_KEYS:
        if (a in n and b in n) or (a in r and b in r):
            return True
    return False

def find_cash_and_nav(text: str) -> Tuple[Optional[float], Optional[float]]:
    cash = None
    nav = None

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # 1) Find NAV line (prefer the last occurrence in the document)
    nav_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if line_has_nav(lines[i]):
            nav = parse_number_from_line(lines[i])
            nav_idx = i
            break

    # 2) CASH is ALWAYS above NAV => scan UPWARDS from nav_idx-1
    if nav_idx is not None:
        for i in range(nav_idx - 1, -1, -1):
            if line_has_cash(lines[i]):
                cash = parse_number_from_line(lines[i])
                break

    # 3) Fallbacks (in case NAV not found for some PDF)
    if nav is None:
        for i in range(len(lines) - 1, -1, -1):
            if line_has_nav(lines[i]):
                nav = parse_number_from_line(lines[i])
                break

    if cash is None:
        # As last resort, scan whole doc but prefer TOP-to-bottom so it doesn't pick the one below NAV
        for i in range(0, len(lines)):
            if line_has_cash(lines[i]):
                cash = parse_number_from_line(lines[i])
                break

    return cash, nav



# ----------------------------
# Holdings parsing (FIXED)
# ----------------------------

def normalize_ticker(token: str) -> str:
    """
    Keep Arabic as-is unless it's the known 'Momentum' label, then map to MOMENTUM.
    For English tickers/words: uppercase.
    Strip trailing punctuation.
    """
    raw = (token or "").strip().strip(" .,:;|")

    # Normalize Arabic presentation forms
    n = nfkc(raw)
    nr = nfkc(reversed_text(raw))

    # Map Momentum Arabic variants -> MOMENTUM (prevents black boxes in PDF)
    # Common extracted forms from your PDFs:
    # - 'ﻡﺗﻧﻣﻭﻣ' (presentation forms)
    # - 'مومنتم' (NFKC normalized)
    if any(x in n for x in ["مومنتم", "ﻡﺗﻧﻣﻭﻣ"]) or any(x in nr for x in ["مومنتم", "ﻡﺗﻧﻣﻭﻣ"]):
        return "MOMENTUM"

    # English / symbols -> uppercase
    if re.match(r"^[A-Za-z0-9_.]+$", raw):
        return raw.upper()

    return raw

def is_holdings_row(line: str) -> bool:
    """
    Any row in the table:
    - Has at least one '%' (usually weight% and change%)
    - Not the header (contains Arabic column words)
    - Has enough tokens
    """
    if "%" not in line:
        return False

    n = nfkc(line)
    # header words seen in your PDF header
    if any(h in n for h in ["تغير", "الوزن", "رويترز", "Row Labels"]):
        return False

    parts = line.split()
    return len(parts) >= 5

def parse_holdings(text: str) -> List[Tuple[str, float]]:
    """
    Correct method for your template:
    - ticker = FIRST token in the row (covers CCAP.CA, stream, and Arabic tickers)
    - weight = SECOND-LAST percent token (because the row ends: weight% change%)
    """
    holdings = []

    for line in text.splitlines():
        line = line.strip()
        if not line or not is_holdings_row(line):
            continue

        parts = line.split()
        ticker_raw = parts[0]
        ticker = normalize_ticker(ticker_raw)

        pct_tokens = [p for p in parts if p.endswith("%")]
        if not pct_tokens:
            continue

        # weight is second-last %, fallback to only one % if that’s all we have
        weight_token = pct_tokens[-2] if len(pct_tokens) >= 2 else pct_tokens[0]

        try:
            weight = float(weight_token.replace("%", ""))
        except:
            continue

        holdings.append((ticker, weight))

    return holdings


# ----------------------------
# Consolidation
# ----------------------------

def build_consolidated(groups: List[Dict]) -> pd.DataFrame:
    group_names = [g["group_name"] for g in groups]
    tickers = sorted({t for g in groups for (t, _) in g["holdings"]}, key=lambda x: str(x))

    index = ["NAV", "CASH"] + tickers
    out = pd.DataFrame(index=index, columns=group_names, dtype="float")

    for g in groups:
        name = g["group_name"]
        if g["nav"] is not None:
            out.loc["NAV", name] = g["nav"]
        if g["cash"] is not None:
            out.loc["CASH", name] = g["cash"]

        for ticker, w in g["holdings"]:
            out.loc[ticker, name] = w

    out["Presence"] = out[group_names].notna().sum(axis=1).astype(int)
    return out

def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="Consolidated", index=True)
    return bio.getvalue()


# ----------------------------
# PDF export (same structure)
# ----------------------------

def df_to_pdf_bytes(df: pd.DataFrame, title: str) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), leftMargin=18, rightMargin=18, topMargin=18, bottomMargin=18)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 10))

    group_cols = [c for c in df.columns if c != "Presence"]

    # TOTALS
    story.append(Paragraph("TOTALS", styles["Heading2"]))
    totals_df = df.loc[["NAV", "CASH"], group_cols].copy()
    totals_df.index = ["Total NAV", "Total Cash/PP"]

    totals_table_data = [["Metric"] + group_cols]
    for idx, row in totals_df.iterrows():
        totals_table_data.append([idx] + [
            "" if pd.isna(row[c]) else f"{row[c]:,.2f}" for c in group_cols
        ])

    t = Table(totals_table_data, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # HOLDINGS
    story.append(Paragraph("HOLDINGS", styles["Heading2"]))
    holdings_df = df.drop(index=["NAV", "CASH"], errors="ignore").copy()

    holdings_table_data = [["Ticker"] + group_cols + ["Presence"]]
    for idx, row in holdings_df.iterrows():
        holdings_table_data.append([str(idx)] + [
            "" if pd.isna(row[c]) else f"{row[c]:.2f}%" for c in group_cols
        ] + [f"{int(row['Presence'])}/{len(group_cols)}"])

    ht = Table(holdings_table_data, repeatRows=1)
    ht.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-2, -1), "RIGHT"),
        ("ALIGN", (-1, 1), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
    ]))
    story.append(ht)

    doc.build(story)
    return buf.getvalue()


# ----------------------------
# Streamlit UI
# ----------------------------

st.set_page_config(page_title="Portfolio Consolidation (PDF)", layout="wide")
st.title("📄 Portfolio Consolidation – Multi PDF")

files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

if files:
    groups = []
    failed = []

    with st.spinner("Extracting PDFs..."):
        for f in files:
            try:
                text = extract_text_from_pdf(f.getvalue())

                group_name = find_group_name(text, fallback=f.name.replace(".pdf", ""))
                cash, nav = find_cash_and_nav(text)
                holdings = parse_holdings(text)

                groups.append({
                    "group_name": group_name,
                    "cash": cash,
                    "nav": nav,
                    "holdings": holdings,
                })
            except Exception as e:
                failed.append((f.name, str(e)))

    if failed:
        st.error("Some PDFs failed:")
        for name, err in failed:
            st.write(f"- {name}: {err}")

    if groups:
        st.subheader("Extraction preview")
        for g in groups:
            st.markdown(f"### {g['group_name']}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cash", f"{g['cash']:,.2f}" if g["cash"] is not None else "—")
            c2.metric("Total NAV", f"{g['nav']:,.2f}" if g["nav"] is not None else "—")
            c3.metric("Holdings rows", f"{len(g['holdings'])}")

            if g["holdings"]:
                st.dataframe(
                    pd.DataFrame(g["holdings"], columns=["Ticker", "Weight_%"]).sort_values("Weight_%", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

        consolidated = build_consolidated(groups)
        st.subheader("✅ Consolidated Output")
        st.dataframe(consolidated, use_container_width=True)

        st.download_button(
            "⬇️ Download Excel",
            data=to_excel_bytes(consolidated),
            file_name="portfolio_consolidation_multi_pdf.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            "⬇️ Download PDF",
            data=df_to_pdf_bytes(consolidated, title="Master Allocation Comparison (Consolidated)"),
            file_name="portfolio_consolidation_multi_pdf.pdf",
            mime="application/pdf",
        )
else:
    st.info("Upload one or more PDFs to start.")
