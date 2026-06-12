# 1) train một lần, lưu lại
python scripts/run_surrogate.py --run runs/lunarlander --surrogate tga \
    --dagger-iters 3 --save runs/lunarlander/tga.pt

# 2) load ra đánh giá (success rate PPO vs TGA-KAN + return gap), không train lại
python scripts/eval_surrogate.py --run runs/lunarlander \
    --load runs/lunarlander/tga.pt --surrogate tga --episodes 100