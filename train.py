# train.py
import torch
import torch.nn.functional as F

from model import NanoFormer

# Check if Apple Silicon GPU (MPS) is available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training on device: {device}")

# Hyperparameters
batch_size = 32  # How many sequences we process in parallel
block_size = 128  # Max sequence length (context window)
embedding_dims = 128  # Size of our "semantic space"
num_heads = 4  # Number of parallel attention heads
num_layers = 4  # Number of transformer blocks stacked together
learning_rate = 1e-3  # How quickly the model updates its weights
max_iters = 5000  # Total training steps
eval_interval = 500  # How often to check validation loss

# Prepare Data
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Build the vocabulary and the encoder/decoder
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

# Split into Train and Validation
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


# Data Loading Function
def get_batch(split):
    """
    Randomly selects a batch of sequences from the data.
    x = Input (tokens 0 to block_size-1)
    y = Target (tokens 1 to block_size, the prediction for the next character)
    """
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


# Initialize Model and Optimizer
model = NanoFormer(vocab_size, embedding_dims, block_size, num_heads, num_layers)
model = model.to(device)

model.token_embedding_table.weight.data = model.token_embedding_table.weight.data.to(
    device
)
model.position_embedding_table.weight.data = (
    model.position_embedding_table.weight.data.to(device)
)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 5. Training Loop
print("Starting training...")
for step in range(max_iters):
    # Fetch a batch of training data
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    # Forward pass: get model predictions (logits)
    logits = model(xb)

    # Calculate Loss: Cross Entropy compares predictions to actual next characters
    B, T, C = logits.shape
    loss = F.cross_entropy(logits.view(B * T, C), yb.view(B * T))

    # Backward pass: calculate gradients (the 'blame' for the error)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Optimizer step: update weights
    optimizer.step()

    # Reporting
    if step % eval_interval == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

print("Training finished!")

# Simple Generation Test
# Let's see if the model has learned to output anything coherent
context = torch.zeros((1, 1), dtype=torch.long).to(device)
print("\nGenerated Text:")
# We generate a simple sequence to test the model's 'voice'
for _ in range(200):
    logits = model(context[:, -block_size:])
    probs = F.softmax(logits[:, -1, :], dim=-1)
    next_char = torch.multinomial(probs, num_samples=1)
    context = torch.cat((context, next_char), dim=1)
    print(itos[next_char.item()], end="")
