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
