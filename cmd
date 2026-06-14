# 1) train một lần, lưu lại
python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
    --dagger-iters 3 --save runs/lunarlander/tga.pt 

# 2) load ra đánh giá (success rate PPO vs TGA-KAN + return gap), không train lại
python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
    --dagger-iters 3 --save runs/lunarlander/tga.pt 
python scripts/eval_surrogate.py --run runs/lunarlander \
    --load runs/lunarlander/tga.pt --surrogate tga --episodes 100

python scripts/train_ppo.py --env Pendulum-v1 --steps 500000 --out runs/pendulum 

python scripts/run_surrogate.py --run runs/pendulum --surrogate tga --dagger-iters 3 --save runs/pendulum/tga.pt --K 1
python scripts/eval_surrogate.py --run runs/pendulum \
    --load runs/pendulum/tga.pt --surrogate tga --episodes 100

python scripts/extract_rules.py \
    --run runs/pendulum \
    --load runs/pendulum/tga.pt \
    --out  runs/pendulum/rules --plots
python scripts/train_ppo.py --env LunarLanderContinuous-v3 --steps 1000000 --out runs/lunarlander

python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga --dagger-iters 3 --save runs/lunarlander/tga.pt --

python scripts/eval_surrogate.py --run runs/lunarlander \
    --load runs/lunarlander/tga.pt --surrogate tga --episodes 100

python scripts/extract_rules.py \
    --run runs/lunarlander \
    --load runs/lunarlander/tga.pt \
    --out  runs/lunarlander/rules --plots


python scripts/train_ppo.py --env Pendulum-v1 --steps 500000 --out runs/pendulum 
python scripts/run_surrogate.py --run runs/pendulum --surrogate tga --dagger-iters 3 --save runs/pendulum/tga.pt --K 1
python scripts/eval_surrogate.py --run runs/pendulum \
    --load runs/pendulum/tga.pt --surrogate tga --episodes 100


python scripts/eval_surrogate.py --run runs/pendulum \
    --load runs/pendulum/tga_l2.pt --surrogate tga --episodes 100


python scripts/train_ppo.py --env LunarLanderContinuous-v3 --steps 1000000 --out runs/lunarlander
# train (now with k-means warm-start + non-saturating MDL)
python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
    --dagger-iters 3 --seed 0 --lam-g 0.0 --save runs/lunarlander/tga.pt

# extract rules (now sentinel-free + support-aware ranking)
python scripts/extract_rules.py --run runs/lunarlander \
    --load runs/lunarlander/tga.pt --out runs/lunarlander/rules --plots
python scripts/eval_surrogate.py --run runs/lunarlander \
    --load runs/lunarlander/tga.pt --surrogate tga --episodes 100

python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
    --axis-only --K 3 --lam-g 0.05 --n-basis 6 --seed 0 \
    --save runs/lunarlander/tga_axis.pt
python scripts/extract_rules.py --run runs/lunarlander \
    --load runs/lunarlander/tga_axis.pt --out runs/lunarlander/rules --plots
python scripts/eval_surrogate.py --run runs/lunarlander \
    --load runs/lunarlander/tga_axis.pt --surrogate tga --episodes 100

python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
    --axis-only --K 3 --lam-g 0.05 --seed 0 \
    --save runs/lunarlander/tga_fo.pt

python scripts/extract_rules.py --run runs/lunarlander \
    --load runs/lunarlander/tga_fo.pt --out runs/lunarlander/rules_fo
python scripts/eval_surrogate.py --run runs/lunarlander \
    --load runs/lunarlander/tga_fo.pt --surrogate tga --episodes 100


python scripts/train_ppo.py --env Pendulum-v1 --steps 500000 --out runs/pendulum

python scripts/run_surrogate.py --run runs/pendulum --surrogate tga \
    --axis-only --K 1 --lam-g 0.05 --seed 0 \
    --save runs/pendulum/tga_fo.pt
python scripts/extract_rules.py --run runs/pendulum \
    --load runs/pendulum/tga_fo.pt --out runs/pendulum/rules_fo
python scripts/eval_surrogate.py --run runs/pendulum \
    --load runs/pendulum/tga_fo.pt --surrogate tga --episodes 100

