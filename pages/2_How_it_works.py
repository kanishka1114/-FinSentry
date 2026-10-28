"""FinSentry-Mini dashboard: How it works.

Explains the data, inputs, time split and threshold using outputs/results.json. Nothing is
trained or scored here.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "results.json"


@st.cache_data
def load_results(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


st.set_page_config(page_title="FinSentry-Mini: How it works", layout="wide")
st.title("How it works")

if not RESULTS_PATH.exists():
    st.error(f"Saved results not found at {RESULTS_PATH}.")
    st.stop()

r = load_results(str(RESULTS_PATH))
data, split, model = r["data"], r["split"], r["model"]
rf_val, rf_test = r["random_forest"]["validation"], r["random_forest"]["test"]

st.header("1. Data")
st.markdown(
    f"PaySim is a **simulated** mobile-money dataset with {data['rows_in_file']:,} "
    f"transactions. We used a random sample of **{data['sample_rows']:,}**, of which only "
    f"**{data['sample_fraud']:,}** are fraud, about 1 in "
    f"{round(data['sample_rows'] / data['sample_fraud']):,}.")

st.header("2. What the model looks at")
st.markdown(
    "- **Amount** of the transaction\n"
    "- **Type**: CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER\n"
    "- **Hour of day**\n"
    "- **Sender's balance before** and **after** the transaction\n\n"
    "No names, account numbers or other personal details are used, and the fraud label is "
    "never an input.\n\n"
    "Because the balance **after** the transaction is only known once the money has moved, "
    "this is **post-transaction fraud review**: it checks completed transactions. It does not "
    "block or predict fraud before a transaction happens.")

st.header("3. Train, validation and test, in time order")
st.dataframe(pd.DataFrame([
    {"Period": name, "Hours": f"{s['first_step']}–{s['last_step']}",
     "Transactions": f"{s['rows']:,}", "Fraud": f"{s['fraud']:,}", "Used to": use}
    for name, s, use in (
        ("Train", split["train"], "teach the model"),
        ("Validation", split["validation"], "choose the alert threshold"),
        ("Test", split["test"], "measure the final results (used once)"),
    )]).set_index("Period"), width="stretch")
st.markdown(
    "The data is split by time, like real life: the model learns from earlier transactions "
    "and is tested on later ones it has never seen.")

st.header("4. The alert threshold")
st.markdown(
    "For each transaction the Random Forest gives a **score from 0 to 1**. If the score is at "
    f"or above the **threshold ({model['threshold']:.3f})**, it raises an alert; below it, no "
    "alert. The threshold was the one that gave the best F1 on the **validation** period, "
    "so the test period was not used to pick it.")
st.dataframe(pd.DataFrame([
    {"Period": period, "Fraud caught": f"{t['fraud_caught']:,}",
     "False alerts": f"{t['false_alerts']:,}", "Missed fraud": f"{t['missed_fraud']:,}",
     "F1": f"{t['f1']:.2f}"}
    for period, t in (("Validation (threshold chosen here)", rf_val),
                      ("Test (final result)", rf_test))
]).set_index("Period"), width="stretch")

st.header("5. Why we don't use accuracy")
st.markdown(
    "Fraud is so rare that a model which **never** raises an alert would still be over 99% "
    f"accurate (on the test period: {1 - split['test']['fraud_rate']:.2%}) while catching "
    "zero fraud. That is why we report **precision, recall and F1**, which only count "
    "how well fraud is found.")

st.info(
    "**Limitation:** PaySim is simulated, so these results do not show how the model would "
    "do on real bank data.")
