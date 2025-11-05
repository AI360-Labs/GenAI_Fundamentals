# LLM Data Training and Foundations

## How LLMs Work

LLMs function by predicting the next token in a sequence based on massive previously seen text datasets.

### Tokenization

Input text is divided into smaller units called **tokens** (words or subwords) via methods like Byte Pair Encoding (BPE). This allows flexible handling of vocabulary and rare words.

### Embedding Layer

Tokens are transformed into continuous numerical vectors (embeddings) which capture syntactic and semantic relationships between words, enabling contextual representation.

### Transformer Architecture

The core model structure involves stacks of transformer blocks, each including:

**Attention Mechanisms:** Particularly self-attention lets the model weigh relationships between all tokens in the input, regardless of their position, enabling understanding of long-range dependencies and context.

**Feedforward Neural Networks:** These layers process attended information to create refined token representations.

**Normalization Layers:** Stabilize training and improve convergence.

### Encoder-Decoder Framework

- The **encoder** converts input text into a rich vector representation. Models like BERT use encoders only for understanding text.
- The **decoder** generates text based on encoded vectors; models like GPT use decoders only for generation.
- Combined encoder-decoder models (e.g., T5) are used for sequence-to-sequence tasks like translation.

## Core Behavior of LLMs

- LLMs act as probabilistic "autocompleters," sampling the most likely next tokens based on learned distributions.
- Because outputs are probabilistic, the same prompt can generate different outputs on each run, leading to inconsistency.
- **Hallucinations** occur when models produce plausible-sounding but factually incorrect outputs.
- Rare or unseen tokens have lower probabilities and thus appear less frequently.

## Generation Parameters

### Temperature

Controls the randomness of generated text by scaling the probability distribution before sampling.

- **Low values** (e.g., 0.2–0.5) lead to conservative, coherent, and deterministic text.
- **High values** (e.g., 0.8–1.2) increase randomness and creativity but risk incoherence.

### Top-K Sampling

Limits the candidate next tokens to the top K most probable ones to pick from.

### Top-P (Nucleus) Sampling

Dynamically selects tokens whose cumulative probability covers threshold P (e.g., 0.9), balancing diversity and coherence.

### Repetition Penalty

Reduces the chance of repeated token generation to improve output diversity.

## Prompt Engineering

Crafting effective prompts is essential to guide LLMs to desired outputs.

### Approaches

- **Zero-shot:** No examples, just instructions.
- **Few-shot:** Providing a handful of examples in the prompt.
- **Chain-of-thought:** Encouraging step-by-step reasoning to improve response quality.

### Best Practices

- Specifying roles and contexts in prompts enhances model comprehension and output relevance.
- Using special tokens and structured templates prevents ambiguity.

## Agents and Agentic Systems

**Agents** are autonomous AI entities that can make decisions, choose actions, and adapt workflows dynamically.

### Key Characteristics

- They orchestrate tasks by routing queries, parallelizing subtasks, evaluating outputs, and optimizing iteratively.
- **Reactive agents** manage limited-scope tasks efficiently.
- Agents are suited for complex, variable tasks; workflows for predictable tasks.
- **Hybrid systems** combine both for improved efficiency and adaptability.

## Embeddings

Embeddings are dense vector representations encoding the meaning of text or data.

### Applications

- Enable similarity search, recommendations, clustering, and more.
- Common similarity metrics include **cosine similarity** and **dot product**.
- Transform discrete tokens into continuous vector spaces where semantically similar items lie closer.

## Retrieval-Augmented Generation (RAG)

Combines information retrieval from external knowledge bases with LLM generation.

### How It Works

- **Chunking** breaks large documents into manageable pieces indexed by vector databases.
- Retrieval grounds generated responses in factual data, reducing hallucinations.

### Use Cases

- Customer support bots
- Summarizing multimedia content
- Fraud detection

## Testing and Evaluation

- LLMs are benchmarked on accuracy, fluency, and robustness.
- Techniques include adversarial "red-team" testing and regression testing for model updates.
- Evaluation metrics combine quantitative scores and qualitative human feedback.
- Continuous evaluation improves safety, reliability, and user trust.

---

## Resources

### Datasets & Tools

- **FineWeb Dataset:** [HuggingFace FineWeb](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)
- **Tokenizer from OpenAI:** [Tiktoken Visualizer](https://tiktokenizer.vercel.app/?model=gpt-3.5-turbo)
- **Transformer Explainer:** [Interactive Transformer](https://poloclub.github.io/transformer-explainer/)
- **Prompt Tuning from OpenAI:** [OpenAI Chat Editor](https://platform.openai.com/chat/edit?models=gpt-5&optimize=true)

### Evaluation & Development

- **LLM as Judge and Evaluations:** [Evidently AI Blog](https://www.evidentlyai.com/blog)
- **Developing Agents:** [Kaggle Agent Whitepaper](https://www.kaggle.com/whitepaper-agent-companion)

### Benchmarks & Comparisons

- **Vector DB Comparison:** [Superlinked Comparison](https://superlinked.com/vector-db-comparison)
- **LM Arena:** [lmarena.ai](https://lmarena.ai) - Find the best AI for you
- **Vellum AI** [LLM Leaderboard](https://www.vellum.ai/llm-leaderboard)

### Learning Resources

- **Andrej Karpathy:** [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI)

### Reddit Communities

- [r/LocalLLaMA](https://www.reddit.com/r/LocalLLaMA/)
- [r/LLMDevs](https://www.reddit.com/r/LLMDevs/)
- [r/Rag](https://www.reddit.com/r/Rag/)