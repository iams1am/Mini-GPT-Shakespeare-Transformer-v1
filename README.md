\# Mini-GPT Shakespeare Transformer (PyTorch from Scratch)



A decoder-only Transformer Language Model (GPT-style architecture) built completely from scratch in PyTorch, featuring custom Causal Multi-Head Attention, Positional Embeddings, and GELU Feed-Forward Networks.



\## Architecture Highlights

\* \*\*Model Type:\*\* Decoder-Only Transformer (GPT Architecture)

\* \*\*Parameters:\*\* \~825,000 trainable parameters

\* \*\*Embedding Dimension:\*\* 128

\* \*\*Attention:\*\* 4 Causal Multi-Head Attention blocks with strict upper-triangular masking (`float('-inf')`)

\* \*\*Context Length (Block Size):\*\* 128 tokens

\* \*\*Layers:\*\* 4 stacked Transformer Blocks with Pre-Layer Normalization and Residual Connections



\## Key Innovations vs. RNN/LSTM

\* \*\*Parallel Computation:\*\* Replaced sequential time-step loops with parallel matrix multiplications (`Q @ K.T`), drastically improving GPU utilization on an NVIDIA GeForce MX330.

\* \*\*Positional Encoding:\*\* Incorporated learned spatial position vectors to preserve sequence order without recurrence.

\* \*\*Causal Masking:\*\* Implemented lower-triangular attention masks to prevent target token lookahead during autoregressive pretraining.



\## Performance Metrics

\* \*\*Final Validation Loss:\*\* 1.58

\* \*\*Character-Level Perplexity:\*\* \~4.86

\* \*\*Peak VRAM Usage:\*\* 380 MB

\* \*\*Training Time:\*\* \~6.5 minutes on local CUDA GPU (2,000 iterations)



\## How to Run

1\. Execute `python step5\_train\_gpt.py` to download `shakespeare.txt` and train the model from scratch.

2\. Run `python step6\_generate\_gpt.py` to generate novel text using temperature-scaled multinomial sampling.

