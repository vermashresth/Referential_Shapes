apt-get install python3-cairo
pip install pybullet

epochs=5
total_seeds=0


use_bullet=0

dataset_type=0
while [ $dataset_type -le 0 ]
do
   mkdir data
   mkdir data/shapes
   python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet

   pop_size=1
   while [ $pop_size -le 10 ]
  do
   seed=0
   while [ $seed -le $total_seeds ]
   do
     python pop_test.py --seed $seed --K 1 --pop_size $pop_size --use_distractors_in_sender 1 --epochs $epochs --use_bullet $use_bullet

    ((seed++))
   done
   ((pop_size++))
   ((pop_size++))
  done
  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  ((dataset_type++))
done


use_bullet=1

dataset_type=0
while [ $dataset_type -le 0 ]
do
   mkdir data
   mkdir data/shapes
   python shapes/gen_shapes.py --dataset_type $dataset_type --noise_strength 0 --use_bullet $use_bullet

   pop_size=1
   while [ $pop_size -le 10 ]
  do
   seed=0
   while [ $seed -le $total_seeds ]
   do
     python pop_test.py --seed $seed --K 1 --pop_size $pop_size --use_distractors_in_sender 1 --epochs $epochs --use_bullet $use_bullet

    ((seed++))
   done
   ((pop_size++))
   ((pop_size++))
  done
  rm -r data
  rm -r dumps
  python shapes/delete_script.py --dataset_type $dataset_type
  ((dataset_type++))
done
