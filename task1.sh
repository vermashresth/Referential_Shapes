apt-get install python3-cairo
pip install pybullet

epochs=30
total_seeds=1
# use_distractors_in_sender=0
# while [ $use_distractors_in_sender -le 1 ]
#   do
#     use_bullet=0
#     while [ $use_bullet -le 1 ]
#     do
#       dataset_type=0
#       while [ $dataset_type -le 1 ]
#       do
#        seed=0
#        while [ $seed -le $total_seeds ]
#        do
#         mkdir data
#         mkdir data/shapes
#         python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet
#
#         python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
#         python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
#         python random_model_gen.py
#         python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
#
#         rm -r data
#         rm -r dumps
#         ((seed++))
#        done
#        ((dataset_type++))
#       done
#       ((use_bullet++))
#       done
#       ((use_distractors_in_sender++))
#     done
#
# echo alldone

use_distractors_in_sender=0
use_bullet=0

dataset_type=0
while [ $dataset_type -le 1 ]
do
 seed=0
 while [ $seed -le $total_seeds ]
 do
  mkdir data
  mkdir data/shapes
  python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet
  python shapes/gen_shapes.py --dataset_type 4 --noise_strength 0 --use_bullet $use_bullet

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python random_model_gen.py
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --use_random_model 1 --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  python shapes/delete_script.py --dataset_type 4
  ((seed++))
 done
 ((dataset_type++))
done




use_distractors_in_sender=0
use_bullet=1

dataset_type=0
while [ $dataset_type -le 1 ]
do
 seed=0
 while [ $seed -le $total_seeds ]
 do
  mkdir data
  mkdir data/shapes
  python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet
  python shapes/gen_shapes.py --dataset_type 4 --noise_strength 0 --use_bullet $use_bullet

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python random_model_gen.py
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --use_random_model 1 --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  python shapes/delete_script.py --dataset_type 4
  ((seed++))
 done
 ((dataset_type++))
done




use_distractors_in_sender=1
use_bullet=0

dataset_type=0
while [ $dataset_type -le 1 ]
do
 seed=0
 while [ $seed -le $total_seeds ]
 do
  mkdir data
  mkdir data/shapes
  python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet
  python shapes/gen_shapes.py --dataset_type 4 --noise_strength 0 --use_bullet $use_bullet

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python random_model_gen.py
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --use_random_model 1 --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  python shapes/delete_script.py --dataset_type 4
  ((seed++))
 done
 ((dataset_type++))
done



use_distractors_in_sender=1
use_bullet=1

dataset_type=0
while [ $dataset_type -le 1 ]
do
 seed=0
 while [ $seed -le $total_seeds ]
 do
  mkdir data
  mkdir data/shapes
  python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet
  python shapes/gen_shapes.py --dataset_type 4 --noise_strength 0 --use_bullet $use_bullet

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python random_model_gen.py
  python test.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 0 --use_random_model 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
  python zero_shot_predict.py --dataset_type 4 --pretrain_dataset_type $dataset_type --use_random_model 1 --seed $seed --K 1 --noise_strength 0  --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  python shapes/delete_script.py --dataset_type 4
  ((seed++))
 done
 ((dataset_type++))
done
