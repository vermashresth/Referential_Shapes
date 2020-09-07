python gen_shapes.py --dataset_type 0 --noise_strength 0

python test.py --dataset_type 0 --seed 0 --K 1 --noise_strength 0 --should_train_visual 1
python test.py --dataset_type 0 --seed 0 --K 1 --noise_strength 0 --should_train_visual 0
python test.py --dataset_type 0 --seed 0 --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1
