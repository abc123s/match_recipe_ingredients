# match_recipe_ingredients
Peter Zhang's (SUNetId: pzhanggg) CS 229 Final Project

## Model Scripts
The following scripts below are all located in the `model` directory.

### Training a model (offline triplets)
You can train a model using the command `python train.py` after setting the desired hyperparameters and dataset to train on in the file. This will train the model, and generate a new sub-directory within the `experiments` directory with the params used, performance of the model, the final model weights, and tensorboard logs. This training procedure uses the ~300k fixed triplet training dataset described in section 4.3.1 of my final project report.

### Training a model (online triplet-mining)
You can continue to train a model the command `python retrain_online.py` after specifying the experiment directory containing the model to retrain as well the desired hyperparameters for the retraining in the file. Like `python train.py` all experiment outputs will be stored in a new sub-directory of the `experiments` directory. This training procedure uses dynamically generated batches of hard triplets described in section 4.3.2 of my final project report.

### Run qualitative analysis
The simple qualitative analysis mentioned in the Experiments and Discussion section of the final report can be reproduced by running `python qualitative_analysis.py`. All you need to do is specify the experiment directory containing the model you would like to run the qualitative analysis on in the file.

## Baseline Scripts
The following scripts below are all located in the `baseline` directory.

`python naive_bayes.py` runs the naive bayes baseline discussed in the final report

The heuristic-based baseline was implemented in javascript and is located in the scripts directory of https://github.com/abc123s/example_tagger.
