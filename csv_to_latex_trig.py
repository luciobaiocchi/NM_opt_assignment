import pandas as pd
from pathlib import Path

##############################################################################
# Configurazione
##############################################################################

SUMMARY_DIR = Path("./summary")
OUT_DIR = Path("latex_tables")
OUT_DIR.mkdir(exist_ok=True)

##############################################################################
# Metadati tabelle
##############################################################################

TABLE_META = {
    "summary_tn_trig_exact.csv": {
        "caption": "Truncated Newton on the Banded Trigonometric problem with exact gradient and exact Hessian.",
        "label": "tab:tn_trig_exact"
    },
    "summary_mn_trig_exact.csv": {
        "caption": "Modified Newton on the Banded Trigonometric problem with exact gradient and exact Hessian.",
        "label": "tab:mn_trig_exact"
    },
    "summary_tn_trig_ex_grad.csv": {
        "caption": "Truncated Newton on the Banded Trigonometric problem with exact gradient and approximate Hessian.",
        "label": "tab:tn_trig_ex_grad"
    },
    "summary_mn_trig_ex_grad.csv": {
        "caption": "Modified Newton on the Banded Trigonometric problem with exact gradient and approximate Hessian.",
        "label": "tab:mn_trig_ex_grad"
    },
    "summary_tn_trig_all_aprox.csv": {
        "caption": "Truncated Newton on the Banded Trigonometric problem with approximate gradient and approximate Hessian.",
        "label": "tab:tn_trig_all_aprox"
    },
    "summary_mn_trig_all_aprox.csv": {
        "caption": "Modified Newton on the Banded Trigonometric problem with approximate gradient and approximate Hessian.",
        "label": "tab:mn_trig_all_aprox"
    }
}

##############################################################################
# Rinomina colonne
##############################################################################

COLUMN_RENAME = {
    "n": r"$n$",
    "h": r"$h$",
    "is_dynamic": r"dyn",
    "mean_k": r"$\overline{k}$",
    "mean_time": r"$\overline{t}$ [s]",
    "mean_grad_norm": r"$\|\nabla F(x_k)\|$",
    "mean_fx": r"$F(x_k)$",
    "mean_conv_rate": r"$\overline{p}$",
    "runs": r"runs"
}

ORDER_EXACT = [
    "n", "mean_k", "mean_time",
    "mean_grad_norm", "mean_fx",
    "mean_conv_rate", "runs"
]

ORDER_WITH_H = [
    "n", "h", "is_dynamic",
    "mean_k", "mean_time",
    "mean_grad_norm", "mean_fx",
    "mean_conv_rate", "runs"
]

def reorder_columns(df):
    cols = df.columns.tolist()
    desired = ORDER_WITH_H if ("h" in cols or "is_dynamic" in cols) else ORDER_EXACT
    ordered = [c for c in desired if c in cols]
    extras = [c for c in cols if c not in ordered]
    return df[ordered + extras]

##############################################################################
# Generazione tabelle LaTeX (SAFE per scientific notation)
##############################################################################

for csv_name, meta in TABLE_META.items():
    csv_path = SUMMARY_DIR / csv_name
    if not csv_path.exists():
        print(f"⚠️ File mancante: {csv_name}")
        continue

    # 🔑 FORZA LETTURA COME STRINGHE
    df = pd.read_csv(csv_path, dtype=str)

    # Riordina colonne
    df = reorder_columns(df)

    # Rinomina colonne
    df = df.rename(columns={k: v for k, v in COLUMN_RENAME.items() if k in df.columns})

    col_format = "c" * len(df.columns)

    latex = df.to_latex(
        index=False,
        escape=False,          # NON escapare: lascia 1.23e-08 intatto
        column_format=col_format,
        caption=meta["caption"],
        label=meta["label"]
    )

    out_file = OUT_DIR / csv_name.replace(".csv", ".tex")
    out_file.write_text(latex, encoding="utf-8")

    print(f"✅ Tabella LaTeX generata: {out_file}")

print("\n✔ Notazione scientifica conservata (e-notation)")
print("✔ Aggiungi \\usepackage{booktabs} nel preambolo")
