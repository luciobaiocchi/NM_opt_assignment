import pandas as pd

##############################################################################
# Helper: formatta colonne numeriche con cifre significative (scientific)
##############################################################################

def format_sigfig(df, cols, sig=5):
    df_fmt = df.copy()
    for c in cols:
        df_fmt[c] = df_fmt[c].apply(
            lambda x: f"{x:.{sig}g}" if pd.notnull(x) else x
        )
    return df_fmt


cols_sigfig = [
    "mean_k",
    "mean_time",
    "mean_grad_norm",
    "mean_fx",
    "mean_conv_rate"
]

##############################################################################
############# TRUNCATED NEWTON BROYDEN ALL EXACT #############################
##############################################################################

df_tn_trig_exact = pd.read_csv("results/results_tn_trig_all_exact.csv")
df_success = df_tn_trig_exact[df_tn_trig_exact["success"] == "yes"]

summary_tn_trig_exact = (
    df_success
    .groupby(["n"])
    .agg(
        mean_k=("k", "mean"),
        mean_time=("time", "mean"),
        mean_grad_norm=("grad_norm_final", "mean"),
        mean_fx=("fx_final", "mean"),
        mean_conv_rate=("convergence_rate", "mean"),
        runs=("k", "count")
    )
    .reset_index()
)

summary_tn_trig_exact_fmt = format_sigfig(summary_tn_trig_exact, cols_sigfig)
summary_tn_trig_exact_fmt.to_csv("./summary/summary_tn_trig_exact.csv", index=False)

##############################################################################
############# MODIFIED NEWTON BROYDEN ALL EXACT #############################
##############################################################################

df_mn_trig_exact = pd.read_csv("results/results_mn_trig_all_exact.csv")
df_success = df_mn_trig_exact[df_mn_trig_exact["success"] == "yes"]

summary_mn_trig_exact = (
    df_success
    .groupby(["n"])
    .agg(
        mean_k=("k", "mean"),
        mean_time=("time", "mean"),
        mean_grad_norm=("grad_norm_final", "mean"),
        mean_fx=("fx_final", "mean"),
        mean_conv_rate=("convergence_rate", "mean"),
        runs=("k", "count")
    )
    .reset_index()
)

summary_mn_trig_exact_fmt = format_sigfig(summary_mn_trig_exact, cols_sigfig)
summary_mn_trig_exact_fmt.to_csv("./summary/summary_mn_trig_exact.csv", index=False)

##############################################################################
######## TRUNCATED NEWTON BROYDEN EXACT Gradient + Approx Hessian #############
##############################################################################

df_tn_trig_ex_grad = pd.read_csv("results/results_tn_trig_exact_gradient.csv")
df_success = df_tn_trig_ex_grad[df_tn_trig_ex_grad["success"] == "yes"]

summary_tn_trig_ex_grad = (
    df_success
    .groupby(["n", "h", "is_dynamic"])
    .agg(
        mean_k=("k", "mean"),
        mean_time=("time", "mean"),
        mean_grad_norm=("grad_norm_final", "mean"),
        mean_fx=("fx_final", "mean"),
        mean_conv_rate=("convergence_rate", "mean"),
        runs=("k", "count")
    )
    .reset_index()
    .sort_values(["n", "h", "is_dynamic"])
)

summary_tn_trig_ex_grad_fmt = format_sigfig(summary_tn_trig_ex_grad, cols_sigfig)
summary_tn_trig_ex_grad_fmt.to_csv("./summary/summary_tn_trig_ex_grad.csv", index=False)

##############################################################################
######## MODIFIED NEWTON BROYDEN EXACT Gradient + Approx Hessian ##############
##############################################################################

df_mn_trig_ex_grad = pd.read_csv("results/results_mn_trig_exact_gradient.csv")
df_success = df_mn_trig_ex_grad[df_mn_trig_ex_grad["success"] == "yes"]

summary_mn_trig_ex_grad = (
    df_success
    .groupby(["n", "h", "is_dynamic"])
    .agg(
        mean_k=("k", "mean"),
        mean_time=("time", "mean"),
        mean_grad_norm=("grad_norm_final", "mean"),
        mean_fx=("fx_final", "mean"),
        mean_conv_rate=("convergence_rate", "mean"),
        runs=("k", "count")
    )
    .reset_index()
    .sort_values(["n", "h", "is_dynamic"])
)

summary_mn_trig_ex_grad_fmt = format_sigfig(summary_mn_trig_ex_grad, cols_sigfig)
summary_mn_trig_ex_grad_fmt.to_csv("./summary/summary_mn_trig_ex_grad.csv", index=False)

##############################################################################
###### TRUNCATED NEWTON BROYDEN Approx Gradient + Approx Hessian ##############
##############################################################################

df_tn_trig_all_aprox = pd.read_csv("results/results_tn_trig_all_aprox.csv")
df_success = df_tn_trig_all_aprox[df_tn_trig_all_aprox["success"] == "yes"]

summary_tn_trig_all_aprox = (
    df_success
    .groupby(["n", "h", "is_dynamic"])
    .agg(
        mean_k=("k", "mean"),
        mean_time=("time", "mean"),
        mean_grad_norm=("grad_norm_final", "mean"),
        mean_fx=("fx_final", "mean"),
        mean_conv_rate=("convergence_rate", "mean"),
        runs=("k", "count")
    )
    .reset_index()
    .sort_values(["n", "h", "is_dynamic"])
)

summary_tn_trig_all_aprox_fmt = format_sigfig(summary_tn_trig_all_aprox, cols_sigfig)
summary_tn_trig_all_aprox_fmt.to_csv("./summary/summary_tn_trig_all_aprox.csv", index=False)

##############################################################################
###### MODIFIED NEWTON BROYDEN Approx Gradient + Approx Hessian ##############
##############################################################################

df_mn_trig_all_aprox = pd.read_csv("results/results_mn_trig_all_aprox.csv")
df_success = df_mn_trig_all_aprox[df_mn_trig_all_aprox["success"] == "yes"]

summary_mn_trig_all_aprox = (
    df_success
    .groupby(["n", "h", "is_dynamic"])
    .agg(
        mean_k=("k", "mean"),
        mean_time=("time", "mean"),
        mean_grad_norm=("grad_norm_final", "mean"),
        mean_fx=("fx_final", "mean"),
        mean_conv_rate=("convergence_rate", "mean"),
        runs=("k", "count")
    )
    .reset_index()
    .sort_values(["n", "h", "is_dynamic"])
)

summary_mn_trig_all_aprox_fmt = format_sigfig(summary_mn_trig_all_aprox, cols_sigfig)
summary_mn_trig_all_aprox_fmt.to_csv("./summary/summary_mn_trig_all_aprox.csv", index=False)
