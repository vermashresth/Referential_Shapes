apt-get install python3-cairo
pip install pybullet

epochs=5
total_seeds=3


use_bullet=0

dataset_type=0
while [ $dataset_type -le 1 ]
do
   mkdir data
   mkdir data/shapes
   python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet

  use_distractors_in_sender=0
  while [ $use_distractors_in_sender -le 1 ]
  do
   seed=0
   while [ $seed -le $total_seeds ]
   do
    python train_classifier.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
    python visual_drift.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
    python train_classifier.py --only_eval 1 --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

    ((seed++))
   done
   ((use_distractors_in_sender++))
  done
  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  ((dataset_type++))
done


use_bullet=1

dataset_type=0
while [ $dataset_type -le 1 ]
do
   mkdir data
   mkdir data/shapes
   python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet

  use_distractors_in_sender=0
  while [ $use_distractors_in_sender -le 1 ]
  do
   seed=0
   while [ $seed -le $total_seeds ]
   do
     python train_classifier.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
     python visual_drift.py --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs
     python train_classifier.py --only_eval 1 --dataset_type $dataset_type --seed $seed --K 1 --noise_strength 0 --should_train_visual 1 --use_bullet $use_bullet --use_distractors_in_sender $use_distractors_in_sender --epochs $epochs

    ((seed++))
   done
   ((use_distractors_in_sender++))
  done
  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  ((dataset_type++))
done
