
# python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 1000 --n_augs 02 --n_latents 3 --coef_clip 1 --coef_lpips 0 --seed 1 --device cuda:0 --use_wandb &
# python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 1000 --n_augs 04 --n_latents 3 --coef_clip 1 --coef_lpips 0 --seed 1 --device cuda:1 --use_wandb &
# python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 1000 --n_augs 08 --n_latents 3 --coef_clip 1 --coef_lpips 0 --seed 1 --device cuda:2 --use_wandb &
# python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 1000 --n_augs 16 --n_latents 3 --coef_clip 1 --coef_lpips 0 --seed 1 --device cuda:3 --use_wandb &
# wait
# python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 1000 --n_augs 08 --n_latents 3 --coef_clip 0 --coef_lpips 1 --seed 1 --device cuda:0 --use_wandb &
# python clip_lpips.py --lr 1e-2 --weight_decay 0 --n_steps 1000 --n_augs 08 --n_latents 3 --coef_clip 0 --coef_lpips 1 --seed 1 --device cuda:1 --use_wandb &
# python clip_lpips.py --lr 1e-0 --weight_decay 0 --n_steps 1000 --n_augs 08 --n_latents 3 --coef_clip 0 --coef_lpips 1 --seed 1 --device cuda:2 --use_wandb &
# python clip_lpips.py --lr 1e+2 --weight_decay 0 --n_steps 1000 --n_augs 08 --n_latents 3 --coef_clip 0 --coef_lpips 1 --seed 1 --device cuda:3 --use_wandb &
# wait



python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 0e-9 --seed 9 --device cuda:0 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 1e-3 --seed 9 --device cuda:1 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 1e-2 --seed 9 --device cuda:2 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 1e-1 --seed 9 --device cuda:3 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 3e-1 --seed 9 --device cuda:0 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 1e-0 --seed 9 --device cuda:1 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 1e+1 --seed 9 --device cuda:2 --use_wandb &
python clip_lpips.py --lr 1e-4 --weight_decay 0 --n_steps 10000 --n_augs 08 --n_latents 4 --coef_clip 1 --coef_lpips 1e+2 --seed 9 --device cuda:3 --use_wandb &
wait




