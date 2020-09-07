apt-get install python3-cairo
python shapes/gen_shapes.py --dataset_type 0 --noise_strength 0

mkdir data
mkdir data/shapes

dataset_type=0
while [ $dataset_type -le 3 ]
do
 seed=0
 while [ $seed -le 3 ]
 do
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0
  python random_model_gen.py
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1
  ((seed++))
 done
 ((dataset_type++))
done

echo alldone
