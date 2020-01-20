# ACTRCE-Experiments
These are the experiments done to try to reproduce the results in the [ACTRCE paper](https://arxiv.org/abs/1902.04546). Documentations will be added later.

## How to run the experiments
1. Clone this repository.
2. Each folder contains a single experiment, and the file train.ipynb in each folder is used to run the experiment.
3. In the hyperparameter section of train.ipynb, set the variable log_dir, which is the path to the folder containing tensorboard log files.

## Environment 
The environment used in the experiments is [Krazyworld](https://github.com/bstadie/krazyworld). The environment file from the original repository is modified according to the environment specification in the ACTRCE paper. Each experiment contains its own copy of the environment file, so different state encodings can be tested.

## Selected experiment summary and analysis
Some experiment graphs and perhaps more detailed analysis can be found in the folder of each experiment.

1. Experiment name: fixed_agent_board, fixed_board, fixed_board_agent-advice_all
 
   These experiments show that the algorithm work on the simplest environment settings. The experiments use the singleton task settings for KrazyWorld. THe model architecture used is very similar to the architecture described in the paper. As the names of the experiments suggest, the experiemts fix the board of game and in some cases, allow the agent's position to reset. In all three experiments, the success rate is able to increase quickly as shown in the graphs in the folder of the experiments. The experiment fixed_board_agent-advice_all uses the goals "Avoid any goal" and "Avoid any lava" while the other two experiments do not because the reward function for the two goals has to be carefully engineered.

2. Experiment name: compare-teacher

    This experiment aims to compare the performance of the algorithm when using different teachers. More specifically, the experiment compares using optimistic and discouraging teachers to using knowledgeable and discouraging teachers. Other experiment settings are similar to the experiment settings in (1). So, the board is still fixed for this experiment. However, unlike the experiment (1), stochastic gradient descent is used instead of mini batch. The result, shown in the graph in the experiment folder, shows that the agent trained using knowledgeable and discouraging teachers performed much better. The ACTRCE paper showed a similar result.

3. Experiment name: transfer_learning

   The goal of this experiment is to transfer the knowledge that the agent learned from achieving undsirable state. The agent is first trained by two pessimistic teachers. Next, the same agent is trained on the same board with an optimistic teacher and a discouraging teacher. This is compared to an agent that is trained by an optimistic teacher and a discouraging teacher. As shown by the graphs in the folder, pretraining the network using undesirable goals does improve the training speed.

4. Experiment name: fixed_agent, not_fixed

    The experiments settings are similar to that of experiment (1) except the board is not fixed. Regardless of whether the agent is fixed or not, the experiments showed that the agent did not learn from playing the game. Various attemps have been made to fix this problem, but they are all not successful. More details about the attempts are shown in experiment (5).

5. Experiment name: delayed_board_update, expanded_grid-not_fixed, compact_state_encoding

    These are various attempts to fix (4), but none of them are successful. However, some of the methods do show potential. In delayed_board_update, the agent spends some time to learn to play on a fixed board. Next, the agent learns to play on the same board for a fixed amount of time before learning to play on a new board. The experiment shows the agent is able to learn to play a game on a new board much more quickly when it has already learn to play on another board. Yet, the agent is unable to play on a new board without learning.  

6. Experiment name: multi-goal-fixed

   The aim of this experiment is for the agent to learn compositional task. This is still work in progress.
