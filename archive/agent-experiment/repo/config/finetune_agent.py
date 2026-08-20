import time

# AI Agent Expert Finetuning Config (RTX 4060 Laptop Optimized)
out_dir = 'out-agent-ft'
eval_interval = 20
eval_iters = 40
wandb_log = False
wandb_project = 'agent-ft'
wandb_run_name = 'ft-agent-' + str(time.time())

dataset = 'agent' # Our newly created dataset
init_from = 'gpt2' # 124M model

# Hardware Settings
device = 'cuda'
dtype = 'bfloat16'
compile = False

# Training Settings
batch_size = 4
gradient_accumulation_steps = 8
max_iters = 500 # About 1 hour experiment

# Finetuning specific LR
learning_rate = 3e-5
decay_lr = False
always_save_checkpoint = True
