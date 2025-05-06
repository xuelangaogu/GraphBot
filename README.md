# GraphBot: A Unified Graph-based Framework for Agent Systems

GraphBot is a novel framework that leverages graph structures for unified modeling of multimodal semantic alignment and full-chain collaborative decision-making. This project implements IMDB dataset processing and classification using the GraphBot framework.

## Framework Architecture

GraphBot consists of five core components:

1. **Environment Perception Graph (EPG)**: Dynamically captures implicit relationships between entities through multimodal contrastive learning.
2. **Decision Making Graph (DMG)**: Integrates large language models with knowledge graphs for detailed task planning.
3. **Action Execution Graph (AEG)**: Bridges the semantic gap between digital instructions and physical actions through cross-modal tool chains.
4. **Feedback Memory Graph (FMG)**: Optimizes policy transfer based on spatiotemporal association mechanisms.
5. **Constraint Guidance Graph (CGG)**: Ensures safe and controlled behavior through causal-enhanced regularization.

## IMDB Dataset Processing

This implementation demonstrates using GraphBot for IMDB dataset processing and movie genre classification. The model is trained to classify movies into three categories: Action, Comedy, and Drama, using graph representations of movie data.

### Directory Structure

```
GraphBot/
├── config/                # Configuration files
│   └── data_config.yaml   # Dataset configuration
├── dataloaders/           # Dataset loaders
│   ├── imdb_dataloader.py # IMDB dataset class
│   └── imdb_utils.py      # Utility functions
├── model/                 # Model implementations
│   └── graphbot_imdb_model.py # GraphBot IMDB model
├── scripts/               # Training and evaluation scripts
│   ├── train_imdb_stage1.sh  # Stage 1 training script
│   ├── train_imdb_stage2.sh  # Stage 2 training script
│   └── evaluate_imdb.sh      # Evaluation script
├── train_imdb.py          # Basic training script
├── train_imdb_graphbot.py # Enhanced training with all GraphBot components
└── evaluate_imdb.py       # Evaluation script
```

## Installation Requirements

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.38.2
pip install torch-geometric==2.4.0
pip install scikit-learn==1.3.2
pip install tqdm pyyaml
```

## Training the Model

The training process is divided into two stages for better performance:

### Stage 1: Initial Training

```bash
bash GraphBot/scripts/train_imdb_stage1.sh
```

This script trains the model with basic parameters to establish initial representations of movie graph data.

### Stage 2: Advanced Training

```bash
bash GraphBot/scripts/train_imdb_stage2.sh
```

This script builds upon stage 1 training with enhanced parameters to refine the model's understanding of movie graphs.

## Evaluating the Model

```bash
bash GraphBot/scripts/evaluate_imdb.sh
```

This script evaluates the trained model on the IMDB validation dataset and saves evaluation metrics.

## Dataset Format

The IMDB dataset should be structured as follows:

- Each sample includes graph data representing movie entities (movies, actors, directors)
- The graph is represented as a heterogeneous graph with node features
- Node types include "movie", "director", "actor"
- The dataset supports both single graph and dual graph (higpt + skg) formats

## Model Specifics

GraphBot for IMDB processing integrates all five components:

1. **EPG**: Extracts semantic knowledge from movie graph data
2. **DMG**: Processes graph structure for decision making
3. **AEG**: Plans and executes classification actions
4. **FMG**: Stores and retrieves relevant movie pattern memories
5. **CGG**: Ensures robust classification by applying constraints

## Performance

On the IMDB dataset, GraphBot demonstrates superior performance compared to traditional methods:

- Higher F1 score than GNN-only approaches like GraphSAGE and GAT
- Better accuracy than heterogeneous graph approaches like HAN and HGT
- More robust classification compared to HiGPT and GraphAgent

## Citation

If you use GraphBot in your research, please cite:

```
@article{GraphBot2024,
  title={GraphBot: A Unified Graph-based Framework for Cross-modal Semantic Alignment and Collaborative Decision-making},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` 

python GraphBot/train_imdb_graphbot.py --model_name_or_path facebook/opt-1.3b --dataset_name IMDB_fewshot_train --train_data_path Datasets/IMDB/IMDB_fewshot_train --val_data_path Datasets/IMDB/IMDB_fewshot_train --hidden_size 768 --num_layers 4 --memory_size 128 --batch_size 4 --eval_batch_size 16 --num_epochs 5 --learning_rate 3e-5 --warmup_ratio 0.1 --weight_decay 0.01 --lr_scheduler_type cosine --save_path models/graphbot-imdb-stage1/model.pt --save_every_epochs 1 --num_workers 4
