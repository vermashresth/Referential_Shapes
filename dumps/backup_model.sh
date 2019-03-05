#!/bin/bash
model_id=$1
mkdir ${model_id}
sshpass -f cred scp dianar@lisa.surfsara.nl:~/game/dumps/${model_id}/*test* ${model_id}/

files=$(ls ${model_id}/*test*)

if [[ ${files} =~ _[0-9]+_test ]];
then
    cap=${BASH_REMATCH[0]};
else
    echo 'Error with test files';
fi

best_epoch=${cap:1:(${#cap}-6)}

sshpass -f cred scp dianar@lisa.surfsara.nl:~/game/dumps/${model_id}/*_${best_epoch}_* ${model_id}/

sshpass -f cred scp dianar@lisa.surfsara.nl:~/game/dumps/${model_id}/*.png ${model_id}/

files=$(ls ${model_id}/*.png)

if [[ ${files} =~ _[0-9]+.png ]];
then
    cap=${BASH_REMATCH[0]};
else
    echo 'Error with png files';
fi

last_epoch=${cap:1:(${#cap}-5)}

sshpass -f cred scp dianar@lisa.surfsara.nl:~/game/dumps/${model_id}/*_${last_epoch}_* ${model_id}/

