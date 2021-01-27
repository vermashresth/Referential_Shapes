# Repo for analysing visual learning in referential games

## How to use

There are four bash scripts to run experiments with multiple seeds.
task1.sh : For generating dataset and getting communication accuracy, noise accuracy, top sim, generalization accuracy (zeroshot) for all game combinations.

task2.sh : For generating heatmaps. These are saved in figs/. Use handle_viz.py script to visualize these heatmaps.

task3.sh : For running visual drift experiment.

task4.sh : Population based experiments

These bash scripts also give information on allowed command line parameters.

All experiments should run on google colab with GPU enabled. Also reccommended to install and use weights and biases (https://www.wandb.com/) to log results.
