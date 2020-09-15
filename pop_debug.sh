apt-get install python3-cairo
pip install pybullet

mkdir data
mkdir data/shapes
dataset_type=0
pop_size=1
while [ $pop_size -le 10 ]
do
 seed=0
 python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0
 while [ $seed -le 1 ]
 do
  python pop_test.py --seed $seed --K 1 --pop_size $pop_size --use_distractors_in_sender
  # python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0
  # python random_model_gen.py
  # python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1
  ((seed++))
 done
 ((pop_size++))
 ((pop_size++))
done

dataset_type=0
pop_size=1
python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet 1
while [ $pop_size -le 10 ]
do
 seed=0
 while [ $seed -le 1 ]
 do
  python pop_test.py --seed $seed --K 1 --pop_size $pop_size --use_bullet --use_distractors_in_sender
  # python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0
  # python random_model_gen.py
  # python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1
  ((seed++))
 done
 ((pop_size++))
 ((pop_size++))
done



echo alldone
