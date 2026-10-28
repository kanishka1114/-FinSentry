"""FinSentry-Mini dashboard: Results page.

Shows the saved results in outputs/results.json (written by train.py). Nothing is trained
or scored here.

    streamlit run Results.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

RESULTS_PATH = Path(__file__).resolve().parent / "outputs" / "results.json"


@st.cache_data
def load_results(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_row(name, t):
    return {"Approach": name, "Fraud caught": f"{t['fraud_caught']:,}",
            "False alerts": f"{t['false_alerts']:,}", "Missed fraud": f"{t['missed_fraud']:,}",
            "Precision": f"{t['precision']:.2f}", "Recall": f"{t['recall']:.2f}",
            "F1": f"{t['f1']:.2f}"}


def confusion_table(t):
    return pd.DataFrame(
        {"Alert": [f"{t['fraud_caught']:,}", f"{t['false_alerts']:,}"],
         "No alert": [f"{t['missed_fraud']:,}", f"{t['correctly_ignored']:,}"]},
        index=["Really fraud", "Really normal"])


st.set_page_config(page_title="FinSentry-Mini: Results", layout="wide")
st.title("FinSentry-Mini: Results")

if not RESULTS_PATH.exists():
    st.error(f"Saved results not found at {RESULTS_PATH}.")
    st.stop()

r = load_results(str(RESULTS_PATH))
test = r["split"]["test"]
rule, rf, before = r["rule"]["test"], r["random_forest"]["test"], r["previous_version"]

st.markdown(
    "**Goal:** flag fraudulent mobile-money transactions. We compare a **simple rule** with a "
    "**Random Forest** model on the **test period**: "
    f"{test['rows']:,} later transactions ({test['fraud']} of them fraud) that the model "
    "never saw while it was trained or while its threshold was chosen.")
st.info(
    "**Simulated data.** PaySim is made-up mobile-money data, not real bank records. "
    "These results show how the approaches did on this simulated data, not how they would "
    "do in the real world.")

st.header("Rule vs Random Forest (test period)")
st.dataframe(pd.DataFrame([result_row("Simple rule", rule), result_row("Random Forest", rf)])
             .set_index("Approach"), width="stretch")
st.markdown(
    f"- **Simple rule:** {r['rule']['description']}\n"
    f"- **Random Forest:** many decision trees vote on each transaction; it raises an alert "
    f"when its score is at least {r['model']['threshold']:.3f}.\n"
    "- **Precision:** of the alerts, how many were really fraud. "
    "**Recall:** of the fraud, how much was caught. **F1:** one number that balances the two "
    "(0 = useless, 1 = perfect).")

st.success(
    f"The Random Forest caught **{rf['fraud_caught']} of {test['fraud']}** frauds with only "
    f"**{rf['false_alerts']} false alerts**. The rule caught {rule['fraud_caught']}, but raised "
    f"{rule['false_alerts']:,} false alerts, so almost every alert it raised was wrong.")

st.header("What changed from the first version")
st.dataframe(pd.DataFrame([
    {"Version": "First version", "Model inputs": before["inputs"],
     "Fraud caught": f"{before['fraud_caught']:,}", "False alerts": f"{before['false_alerts']:,}",
     "Precision": f"{before['precision']:.2f}", "Recall": f"{before['recall']:.2f}",
     "F1": f"{before['f1']:.2f}"},
    {"Version": "This version", "Model inputs": before["inputs"] + ", sender's balance before "
     "and after", "Fraud caught": f"{rf['fraud_caught']:,}", "False alerts": f"{rf['false_alerts']:,}",
     "Precision": f"{rf['precision']:.2f}", "Recall": f"{rf['recall']:.2f}",
     "F1": f"{rf['f1']:.2f}"},
]).set_index("Version"), width="stretch")
st.markdown(
    "Adding the sender's account balance before and after the transaction raised F1 from "
    f"**{before['f1']:.2f} to {rf['f1']:.2f}**. In PaySim, fraud often empties the sender's "
    "account, and the amount alone cannot show that.")

st.header("Confusion matrices (test period)")
c1, c2 = st.columns(2)
for col, name, t in ((c1, "Simple rule", rule), (c2, "Random Forest", rf)):
    with col:
        st.markdown(f"**{name}**")
        st.dataframe(confusion_table(t), width="stretch")
st.caption("Rows show what each transaction really was; columns show whether it was alerted.")
