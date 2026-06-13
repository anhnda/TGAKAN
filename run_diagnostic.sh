#!/usr/bin/env bash
# TGA-KAN LunarLander diagnostic: reproduce K=1, then test whether the
# single-regime collapse is real (Reading A) or a suppressed gate (Reading B).
#
# Run from the repo root that contains scripts/ and runs/lunarlander/.
#   bash run_diagnostic.sh
#
# Assumes Torch is installed and runs/lunarlander/{model.zip,meta.json} exist.
set -euo pipefail

RUN=runs/lunarlander
mkdir -p "$RUN/sweep"
SUMMARY="$RUN/sweep/summary.tsv"
echo -e "tag\tlam_g\tseed\tK_active\tpointwise_MSE\treturn_gap\tsuccess" > "$SUMMARY"

# helper: pull nested numeric fields out of the eval json
# eval_surrogate writes: {full_eval:{pointwise_mse,return_gap}, surrogate_success:{success_rate}}
evalfields() {
  python3 - "$1" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
fe=d.get("full_eval",{})
mse=fe.get("pointwise_mse","")
gap=fe.get("return_gap","")
succ=d.get("surrogate_success",{}).get("success_rate","")
print(f"{mse}\t{gap}\t{succ}")
PY
}

run_one () {
  local tag="$1" lam_g="$2" seed="$3"; shift 3
  local extra="$*"
  local ckpt="$RUN/sweep/tga_${tag}.pt"
  local evalout="$RUN/sweep/eval_${tag}.json"

  echo "================  $tag  (lam_g=$lam_g seed=$seed $extra)  ================"
  python scripts/run_surrogate.py --run "$RUN" --surrogate tga \
      --dagger-iters 3 --seed "$seed" --lam-g "$lam_g" $extra \
      --save "$ckpt"

  python scripts/eval_surrogate.py --run "$RUN" --load "$ckpt" \
      --surrogate tga --episodes 100 --out "$evalout"

  # K_active is reported by extract_rules; capture it from the rules header
  python scripts/extract_rules.py --run "$RUN" --load "$ckpt" \
      --out "$RUN/sweep/rules_${tag}" > "$RUN/sweep/extract_${tag}.log" 2>&1 || true
  local kact
  kact=$(grep -oE "active regimes: [0-9]+/[0-9]+" "$RUN/sweep/extract_${tag}.log" | head -1 | grep -oE "^active regimes: [0-9]+" | grep -oE "[0-9]+$" || echo "?")

  local mse gap succ
  IFS=$'\t' read -r mse gap succ < <(evalfields "$evalout")
  echo -e "${tag}\t${lam_g}\t${seed}\t${kact}\t${mse}\t${gap}\t${succ}" >> "$SUMMARY"
}

# ---------------------------------------------------------------------------
# STEP 0 — reproduce the published K=1 baseline exactly (your three commands)
# ---------------------------------------------------------------------------
echo "########  STEP 0: baseline reproduction  ########"
python scripts/run_surrogate.py --run "$RUN" --surrogate tga \
    --dagger-iters 3 --save "$RUN/tga.pt"
python scripts/eval_surrogate.py --run "$RUN" \
    --load "$RUN/tga.pt" --surrogate tga --episodes 100
python scripts/extract_rules.py --run "$RUN" \
    --load "$RUN/tga.pt" --out "$RUN/rules" --plots

# ---------------------------------------------------------------------------
# STEP 1 — lam_g sweep at fixed seed (does MDL pressure cause the collapse?)
# ---------------------------------------------------------------------------
echo "########  STEP 1: lam_g sweep (seed=0)  ########"
run_one  lamg_0p00   0.0    0
run_one  lamg_0p003  0.003  0
run_one  lamg_0p01   0.01   0
run_one  lamg_0p03   0.03   0   # = published default, sanity re-run

# ---------------------------------------------------------------------------
# STEP 2 — seed sweep at lam_g=0 (is the collapse initialization-driven?)
#          MDL fully off here, so any remaining K=1 is the gate init / softmax.
# ---------------------------------------------------------------------------
echo "########  STEP 2: gate-init sweep (lam_g=0)  ########"
run_one  seed1_lamg0  0.0  1
run_one  seed2_lamg0  0.0  2
run_one  seed3_lamg0  0.0  3

# ---------------------------------------------------------------------------
echo ""
echo "########  SUMMARY  ########"
column -t "$SUMMARY"
echo ""
echo "Interpretation:"
echo "  * K stays 1 across ALL rows, MSE flat ~0.022  -> Reading A: K=1 is real."
echo "  * K rises >1 in any row AND its MSE/return_gap drops materially"
echo "    below the baseline -> Reading B: the gate was being suppressed."
echo "  * K rises but MSE does NOT improve -> extra regimes redundant; K=1 stands."