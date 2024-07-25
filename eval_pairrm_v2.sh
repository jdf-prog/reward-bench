CUDA_VISIBLE_DEVICES=0 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-400" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=1 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-800" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=2 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-1200" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=3 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-1600" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=4 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-2000" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=5 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-2400" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=6 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-2800" --trust_remote_code --batch_size=4 &
CUDA_VISIBLE_DEVICES=7 python scripts/run_rm.py --model "DongfuJiang/PairRM-V2-phi3-3-mini-checkpoint-2882" --trust_remote_code --batch_size=4 &
wait