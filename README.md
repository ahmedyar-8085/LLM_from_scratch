# Building a Small LLM from Scratch 🤖

> A from-scratch implementation of a small GPT-style Language Model using PyTorch, SentencePiece, and a Google Colab T4 GPU.

This project was built as part of the **"Building LLMs from Scratch" Udemy course by Javier Ideami**, with the goal of going beyond simply using an existing LLM and actually understanding how the core components of a Transformer-based language model work.

Rather than treating an LLM as a black box, I implemented and explored the complete pipeline:

**Raw text → Tokenization → Token IDs → Embeddings → Positional Embeddings → Self-Attention → Multi-Head Attention → Feed-Forward Networks → Transformer Blocks → Logits → Cross-Entropy Loss → Backpropagation → Optimization → Text Generation**

---

## 🚀 Project Overview

The model is a small GPT-style autoregressive language model trained on a Wikipedia-based text dataset.

The objective is simple:

> Given a sequence of tokens, predict the next token.

For example:

```text
Input:
The cat sat on the

Target:
mat
```

During training, the model repeatedly predicts the next token, calculates the error using cross-entropy loss, backpropagates the error through the network, and updates its parameters.

At inference time, the model takes a prompt and generates new tokens one at a time.

---

## 🧠 What I Built

The project implements the major components of a decoder-only Transformer:

* SentencePiece tokenizer
* Token embeddings
* Positional embeddings
* Causal self-attention
* Multi-head attention
* Query, Key, and Value projections
* Causal attention masking
* Feed-forward neural networks
* Residual/skip connections
* Layer normalization
* Dropout
* Linear vocabulary projection
* Cross-entropy loss
* AdamW optimizer
* Cosine learning-rate scheduling
* Gradient clipping
* Checkpoint saving/loading
* Autoregressive text generation
* Training and validation loss evaluation

---

## 🏗️ Model Architecture

The model uses the following configuration:

| Hyperparameter      |                  Value |
| ------------------- | ---------------------: |
| Vocabulary Size     |                  4,096 |
| Embedding Dimension |                    384 |
| Context Length      |                    512 |
| Transformer Layers  |                      7 |
| Attention Heads     |                      7 |
| Parameters          |                ~19.84M |
| Batch Size          |                      8 |
| Dropout             |                   0.05 |
| Learning Rate       |                   3e-4 |
| Weight Decay        |                   0.01 |
| Gradient Clipping   |                    1.0 |
| Training Iterations |     100,000 configured |
| Data Type           |               bfloat16 |
| GPU                 | Google Colab NVIDIA T4 |

The model contains approximately **19.84 million trainable parameters**.

---

## 🔤 Tokenization

The model does not directly process raw text.

The text is first converted into tokens using **SentencePiece**.

The tokenizer contains:

```text
Vocabulary = 4,096 tokens
```

For example:

```text
"once upon a time"
        ↓
[2686, 698, 265, 261, 684]
```

The model then works with these numerical token IDs.

The reverse operation is also implemented:

```text
Token IDs
   ↓
SentencePiece decoder
   ↓
Human-readable text
```

---

## 📚 Dataset

The model was trained using a Wikipedia-based text file:

```text
wiki.txt
```

The tokenized dataset contains approximately:

```text
59.21 million tokens
```

The data was split into:

```text
90% → Training
10% → Validation
```

Resulting in approximately:

```text
53.29M training tokens
5.92M validation tokens
```

---

## 📦 Creating Training Batches

Training examples are created using a context window of 512 tokens.

With:

```text
Batch size = 8
Context length = 512
```

each batch has the shape:

```text
X → (8, 512)
Y → (8, 512)
```

The target sequence is shifted by one token:

```text
Input:
A B C D E

Target:
B C D E F
```

This is the fundamental training objective of an autoregressive language model:

```text
Predict token t+1 from tokens ≤ t
```

---

# 🔢 Embedding Layer

The model uses:

```python
nn.Embedding(vocab_size, embed_size)
```

with:

```text
4096 × 384
```

Each token ID is mapped to a learned 384-dimensional vector.

The model also contains a positional embedding:

```text
512 × 384
```

The token and positional embeddings are added together:

```text
Token Embedding
       +
Position Embedding
       ↓
Transformer input
```

This allows the model to represent both:

* what a token is
* where the token occurs in the sequence

---

# 👀 Multi-Head Self-Attention

Each Transformer block contains multi-head self-attention.

The embedding dimension is:

```text
384
```

and the model uses:

```text
7 attention heads
```

Each head operates on:

```text
384 // 7 = 54 dimensions
```

The attention heads independently compute relationships between tokens.

Each head creates:

```text
Q = Query
K = Key
V = Value
```

The attention scores are calculated using:

```text
QKᵀ / √d
```

which produces an attention matrix describing how strongly tokens relate to one another.

---

## 🔒 Causal Masking

Because this is an autoregressive language model, a token must not be allowed to look into the future.

For example:

```text
Token 1 → can see Token 1

Token 2 → can see Tokens 1-2

Token 3 → can see Tokens 1-3

Token 4 → can see Tokens 1-4
```

but:

```text
Token 1 ✗ cannot see Token 2
Token 2 ✗ cannot see Token 3
Token 3 ✗ cannot see Token 4
```

A lower-triangular causal mask is therefore applied to the attention matrix.

---

# 🧩 Transformer Block

Each Transformer block contains:

```text
LayerNorm
   ↓
Multi-Head Self-Attention
   ↓
Residual Connection
   ↓
LayerNorm
   ↓
Feed-Forward Network
   ↓
Residual Connection
```

The model stacks:

```text
7 Transformer Blocks
```

---

# ⚙️ Feed-Forward Network

The feed-forward component expands the representation:

```text
384
 ↓
2304
 ↓
ReLU
 ↓
384
```

because the intermediate dimension is:

```text
6 × 384 = 2304
```

Dropout is applied afterward for regularization.

---

# 🎯 Output Layer

After the Transformer blocks and final LayerNorm:

```text
(Batch, Context, 384)
```

is projected into the vocabulary space:

```text
384 → 4096
```

Therefore the final logits have the shape:

```text
(Batch, Context, Vocabulary)
```

or:

```text
(8, 512, 4096)
```

For every position, the model produces a score for every possible token in the vocabulary.

---

# 📉 Cross-Entropy Loss

The logits are reshaped so that every token prediction can be treated as an individual classification problem.

Conceptually:

```text
(8, 512, 4096)
        ↓
(4096, 4096)
```

and the targets are reshaped accordingly.

Cross-entropy then measures how much probability the model assigned to the correct next token.

The objective is:

```text
Minimize:

- log P(correct next token)
```

A lower loss means the model is becoming better at predicting the next token.

---

# 🔄 Training Pipeline

Each training iteration follows:

```text
Get batch
   ↓
Forward pass
   ↓
Generate logits
   ↓
Calculate cross-entropy loss
   ↓
Zero gradients
   ↓
Backpropagation
   ↓
Gradient clipping
   ↓
AdamW parameter update
   ↓
Learning-rate scheduler
   ↓
Next batch
```

This process is repeated throughout training.

---

# 🧮 Optimization

The model uses **AdamW** with:

```text
Learning Rate = 3e-4
Weight Decay = 0.01
Betas = (0.9, 0.99)
```

Weight decay is applied differently to parameter groups.

Gradient clipping is also used:

```text
max_norm = 1.0
```

to help prevent unstable updates caused by excessively large gradients.

---

# 📈 Learning-Rate Scheduling

A cosine annealing scheduler is used.

The learning rate gradually decreases during training:

```text
Initial LR
   ↓
   ↓
   ↓
   ↓
Minimum LR
```

The configured minimum learning rate is:

```text
3e-5
```

which is one tenth of the initial learning rate.

---

# 💾 Checkpointing

The training loop automatically saves a checkpoint whenever the validation loss improves.

The checkpoint contains:

```python
{
    "model_state_dict": ...,
    "optimizer_state_dict": ...,
    "loss": ...,
    "iteration": ...
}
```

This makes it possible to stop training and later continue from the saved state.

---

# ✍️ Text Generation

The model also includes autoregressive generation.

Given:

```text
"The mountain in my city is"
```

the model:

1. Tokenizes the prompt
2. Runs the Transformer
3. Takes the logits from the final position
4. Applies softmax
5. Samples the next token
6. Adds the new token to the sequence
7. Runs the model again
8. Repeats

Conceptually:

```text
Prompt
  ↓
Transformer
  ↓
Next-token probabilities
  ↓
Sample token
  ↓
Append token
  ↓
Transformer again
  ↓
Next token
  ↓
...
```

The generation function keeps the most recent 512 tokens so that generation remains within the model's context window.

---

# 📊 Training Results

The model started with a loss of approximately:

```text
Train loss: 8.375
Validation loss: 8.375
```

As training progressed, the loss decreased substantially.

At iteration 4,650, the recorded result was:

```text
Train loss:      3.5094
Validation loss: 3.4719
```

This represents a significant improvement from the initial loss.

The training run was configured for 100,000 iterations, but the recorded checkpoint/inference state in the notebook is at iteration 4,650.

---

# 💬 Generation Results

The model was tested with prompts such as:

```text
once upon a time
```

and:

```text
the capital of united states of america is
```

Because this is a relatively small model trained for a limited number of iterations, the generated text is not comparable to a production-scale LLM.

However, the generation experiments demonstrate that the model learned statistical patterns from the training corpus and could produce multi-token continuations.

This project is primarily focused on **understanding the architecture and training process**, rather than achieving state-of-the-art language generation.

---

# 💻 Hardware

Training was performed in:

**Google Colab**

using a:

```text
NVIDIA T4 GPU
```

The model was trained using CUDA when available and configured to use:

```python
dtype = torch.bfloat16
```

The notebook also enables TF32 operations for CUDA matrix multiplication and cuDNN where supported.

---

# 🛠️ Technologies Used

* Python
* PyTorch
* SentencePiece
* CUDA
* Google Colab
* NVIDIA T4 GPU
* tqdm
* Weights & Biases support
* Git / GitHub

---

# 📁 Project Structure

A simplified project structure:

```text
small-llm-from-scratch/
│
├── notebook.ipynb
├── wiki.txt
├── wiki_tokenizer.model
├── encoded_data.pt
│
├── models/
│   └── latest.pt
│
└── README.md
```

---

# 🎓 What I Learned

This project helped me understand the Transformer architecture from the inside rather than only using high-level libraries.

Key concepts I worked with:

### 1. Tokenization

How human-readable text becomes numerical token IDs.

### 2. Embeddings

How token IDs are converted into learned vectors.

### 3. Positional Information

How the Transformer receives information about token positions.

### 4. Self-Attention

How tokens communicate with previous tokens to build contextual representations.

### 5. Query, Key, Value

How attention determines which parts of the context are important.

### 6. Multi-Head Attention

How different attention heads can learn different relationships within the sequence.

### 7. Causal Masking

How GPT-style models prevent future-token information from leaking into predictions.

### 8. Transformer Blocks

How attention, feed-forward networks, normalization, and residual connections work together.

### 9. Cross-Entropy

How the model's next-token probability distribution is compared with the correct token.

### 10. Backpropagation

How the loss produces gradients for the model's parameters.

### 11. Optimization

How AdamW uses those gradients to update the model.

### 12. Autoregressive Generation

How a language model generates text one token at a time.

---

# 🔬 Why Build an LLM From Scratch?

Using an existing LLM is useful, but implementing a small one from the ground up provides a much deeper understanding of what happens underneath the API.

Instead of simply thinking:

```text
Prompt → LLM → Answer
```

I can now understand the process more deeply:

```text
Text
 ↓
Tokenizer
 ↓
Token IDs
 ↓
Embeddings
 ↓
Contextual representations
 ↓
Q/K/V attention
 ↓
Causal masking
 ↓
Multi-head attention
 ↓
Feed-forward computation
 ↓
Residual connections
 ↓
Layer normalization
 ↓
Vocabulary logits
 ↓
Cross-entropy
 ↓
Gradients
 ↓
Parameter updates
 ↓
Learned language model
 ↓
Next-token generation
```

---

# 🚧 Limitations

This is intentionally a **small educational LLM**.

It should not be compared directly with large production language models.

Limitations include:

* ~19.84M parameters
* Limited training duration in the recorded run
* Limited training corpus/domain coverage
* 512-token context window
* 4,096-token vocabulary
* Small batch size
* No instruction tuning
* No RLHF
* No preference optimization
* No retrieval system
* No production inference optimization

The goal of this project was **learning the mechanics of LLMs**, not building a production-grade chatbot.

---

# 🙏 Acknowledgment

This project was developed while following the **Building LLMs from Scratch** Udemy course by **Javier Ideami**.

The course provided the learning framework, while implementing, debugging, experimenting with, and studying the components helped me understand how the pieces of a GPT-style language model fit together.

---

# 📌 Future Improvements

Possible next steps:

* Train for substantially more iterations
* Increase model size
* Experiment with larger datasets
* Improve tokenizer configuration
* Experiment with different context lengths
* Add temperature and top-k/top-p sampling
* Compare different optimizers and learning-rate schedules
* Track training curves with Weights & Biases
* Fine-tune the pretrained model for a specific task
* Build an interactive inference interface
* Experiment with larger GPU configurations

---

## ⭐ Takeaway

Building this model gave me a much stronger understanding of the statement:

> **An LLM is fundamentally a neural network trained to predict the next token — but the Transformer architecture gives it the ability to build rich contextual representations through attention.**

This project was my step toward understanding LLMs from the **mathematics and code level**, rather than treating them as black boxes.
