# BGTplanner-AAAI

This repository contains the code for **BGTplanner** based on Gaussian Process Regression and CMAB medthods. We leveage a federated learning framework with graph neural networks (GNNs). The code uses several command-line arguments to control various hyperparameters and settings. Below is a detailed explanation of each argument.

## Arguments

### `--embed_size` (int, default: 8)
Specifies the size of the embedding vectors used in the GNN model. A larger embedding size might capture more detailed features but also increases computational complexity.

### `--lr` (float, default: 0.05)
Learning rate for the optimization algorithm. This parameter controls how large the update step should be during each iteration of the training process.

### `--data` (required)
Specifies the dataset to be used for training and evaluation. You must provide one of the following:
- `'ml-1m-605_online'`
- `'ml-100k_online'`
- `'filmtrust'`

### `--user_batch` (int, required)
The batch size for processing user data during training. The batch size depends on the dataset selected:
- For `ml-1m-605_online`: 605
- For `ml-100k_online`: 943
- For `filmtrust`: 874

### `--clip` (float, default: 0.1)
Gradient clipping threshold to prevent the exploding gradient problem. This ensures that gradients during backpropagation are within a reasonable range.

### `--total_budget` (float, required)
The total budget used for differential privacy in terms of privacy cost. It could be a value between `1` and `20`.

### `--negative_sample` (int, required)
Specifies the number of fake samples used in the training process. The value should depend on the dataset:
- For `ml-1m`: 400
- For `ml-100k`: 50
- For `filmtrust`: 50

### `--valid_step` (int, default: 1)
Indicates how frequently (in terms of steps) validation should be performed during the training process.

### `--T` (int, required)
The total number of training iterations (or communication rounds) for the federated learning process. The default is 100.

### `--selected_item` (str, default: `'full'`)
Specifies which items are selected for training. The default option is `'full'`, meaning all items are used.

### `--dp_mechnisim` (str, required)
Specifies the differential privacy mechanism to be used. You must select one of the following:
- `'Gaussian'`
- `'laplace'`

### `--allocation` (str, required, default: `'BGTplanner'`)
Defines the allocation strategy for distributing resources or tasks during federated learning. The default strategy is `'BGTplanner'`.

### `--min_training_rounds_frac` (float, default: 4/5)
Minimum fraction of total rounds that should be used for training. This is a value between `0` and `1` indicating the proportion of the total number of rounds dedicated to training. It could be used to determine the action space of BGTplanner.

### `--total_run_times` (int, default: 5)
Specifies how many times the training process should be run, allowing for multiple trials and performance averaging.

### `--dp_delta` (float, default: 1e-5)
The privacy parameter delta used in differential privacy, which represents the probability of privacy being broken. A smaller value implies stronger privacy protection.

### `--rdp_alpha` (float, default: 2)
The alpha parameter for Renyi Differential Privacy (RDP). Controls the trade-off between privacy and utility.

## Example Usage

```bash
python3 main.py --embed_size 16 --lr 0.01 --data 'ml-100k_online' --user_batch 943 --clip 0.1 \
                 --total_budget 10 --negative_sample 50 --valid_step 1 --T 100 \
                 --dp_mechnisim 'Gaussian' --allocation 'BGTplanner' \
                 --min_training_rounds_frac 0.8 --total_run_times 5 --dp_delta 1e-5 --rdp_alpha 2