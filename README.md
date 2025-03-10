# Agent Librarian - Generative AI Multimedia Transcriber & Vector Search #

## Purpose
**Inspiration:** I designed an agent that I can use in my daily studies as a Masters in AI student. This agent acts like a librarian, it not only locates information related to my queries within my lecture videos, but it also provides supporting research papers to enhance my learning.
 
## Description
The project provides a robust pipeline for:

-**Audio Transcription:** Converting audio files into JSON with timestamped text segments using OpenAI's Whisper model.

-**Text Vectorization:** Transforming transcript data and given files into vector embeddings.

-**Semantic Search:** Uses FAISS indexing for fast similarity searches, enabling applications such as chatbots, search engines, and retrieval-augmented generation (RAG) systems.



## Architecture & Design 
The flow is described below:
- **Audio Transcription (audio_transcriber.py):**
  Transcribes audio files into JSON with timestamped text segments using OpenAI's Whisper model.
- **Transcript Vectorization (vectorize_transcript.py):**
  Processes transcript JSON files, converts text segments into vector embeddings using OpenAI embeddings, and builds a FAISS index for fast similarity searches.
- **Knowledge Base Conversion (vectorize_knowledge_base.py):**
  Converts a JSON file of academic papers (combining title, link, and significance) into a FAISS vector database with metadata.
- **Interactive Chatbot (semantic_audio_search.py):**
  Provides an interactive chatbot to search transcript and knowledge base indexes.

## Data Flow 
- **Input:** Audio files and JSON knowledge base.
- **Processing:** 
    - Transcription of audio with timestamps.
    - Text Vectorization (OpenAI embeddings).
    - FAISS Indexing (for fast retrieval).
- **Output:** Vector search results, chatbot responses.



# Getting Started #
## Install Dependencies 
```bash
pip install -r requirments.txt
```

## Installation 
1. **Clone the Repository:**
```bash
git clone https://gitlab.com/ml2774906/agent_librarian.git
cd agent_librarian
``` 

## Set Enviornment Variables 
``` bash
export OPENAI_API_KEY="your_openai_api_key"
```

# Usage # 

**1. Transcribe Audio**
Run the transcription script to convert an audio file into JSON with timestamped text segments:
```bash
python src/audio_transcriber.py "path/to/audio.mp3" --output "transcription.json" --metadata_output "transcript_metadata.json"
```

**2. Vectorize Transcript**
Convert a transcript JSON into vector embeddings and build a FAISS index:
```bash
python src/vectorize_transcript.py --transcript_json "path/to/transcription.json" --output_index "transcript.index"
```

**3. Convert Given Data to Vector Database**
Process a JSON file of academic papers into a FAISS vector database:
```bash
python src/vectorize_knowledge_base.py --json_path "path/to/knowledge_base.json" --output_index "knowledge_base.index" --output_metadata "knowledge_base_metadata.json"
```

**4. Run the Chatbot**
Start the interactive chatbot to query both the transcript and knowledge base:
```bash
python src/semantic_audio_search.py \
    --transcript_index "path/to/transcript.index" \
    --transcript_metadata "path/to/transcript_metadata.json" \
    --kb_index "path/to/knowledge_base.index" \
    --kb_metadata "path/to/knowledge_base_metadata.json" \
    --top_k 3
```


# Example Queries & Demonstration:

Running -
```bash
python src/semantic_audio_search.py \
    --transcript_index "path/to/transcript.index" \
    --transcript_metadata "path/to/transcript_metadata.json" \
    --kb_index "path/to/knowledge_base.index" \
    --kb_metadata "path/to/knowledge_base_metadata.json" \
    --top_k 3
```
```
Chatbot (type 'exit' to quit)

User: Where does this video talk about latent space?

Transcript Results:
1. Score: 0.877 | Timestamp: 00:04:22
   Text: Latent space operation....

2. Score: 0.826 | Timestamp: 00:12:11
   Text: The query comes from the noisy latent images spatial features....

3. Score: 0.823 | Timestamp: 00:01:56
   Text: The model works in the latent space to make the process faster and more memory efficient....

Knowledge Base Results:
1. Score: 0.751 | Title: Auto-Encoding Variational Bayes
   Link: https://arxiv.org/abs/1312.6114
   Significance: This paper introduced the variational autoencoder (VAE), a powerful framework for probabilistic generative modeling. It combined variational inference...

2. Score: 0.751 | Title: Scaling Laws for Neural Language Models
   Link: https://arxiv.org/abs/2001.08361
   Significance: This paper investigated how the performance of neural language models scales with increasing model size, data, and compute. It provided empirical evid...

3. Score: 0.743 | Title: GPT-3: Language Models are Few-Shot Learners
   Link: https://arxiv.org/abs/2005.14165
   Significance: This paper presented GPT-3, a massive language model that achieved remarkable performance with minimal task-specific training. It demonstrated that sc...


User: What is stable diffusion?

Transcript Results:
1. Score: 0.896 | Timestamp: 00:00:16
   Text: how cross-attention works in stable diffusion....

2. Score: 0.888 | Timestamp: 00:03:03
   Text: Stable diffusion uses a process called latent diffusion to generate images....

3. Score: 0.885 | Timestamp: 00:04:24
   Text: Unlike traditional diffusion models, stable diffusion operates in latent space....

Knowledge Base Results:
1. Score: 0.729 | Title: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
   Link: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
   Significance: This paper introduced dropout as a novel regularization technique to prevent overfitting in neural networks. It works by randomly deactivating a subse...

2. Score: 0.728 | Title: Adam: A Method for Stochastic Optimization
   Link: https://arxiv.org/abs/1412.6980
   Significance: This paper introduced the Adam optimizer, which adapts learning rates for each parameter based on estimates of first and second moments. It combined i...

3. Score: 0.726 | Title: Deep Residual Learning for Image Recognition
   Link: https://arxiv.org/abs/1512.03385
   Significance: This paper introduced residual connections, which allowed for the effective training of extremely deep neural networks. It addressed the vanishing gra...


User: Where does the video reference the encoder and decoder?

Transcript Results:
1. Score: 0.865 | Timestamp: 00:08:57
   Text: The encoder captures context from the image, while the decoder reconstructs the image....

2. Score: 0.861 | Timestamp: 00:00:39
   Text: One, encoder....

3. Score: 0.858 | Timestamp: 00:01:44
   Text: One, encoder decoder structure....

Knowledge Base Results:
1. Score: 0.775 | Title: Auto-Encoding Variational Bayes
   Link: https://arxiv.org/abs/1312.6114
   Significance: This paper introduced the variational autoencoder (VAE), a powerful framework for probabilistic generative modeling. It combined variational inference...

2. Score: 0.749 | Title: Sequence to Sequence Learning with Neural Networks
   Link: https://arxiv.org/abs/1409.3215
   Significance: This paper introduced the sequence-to-sequence (seq2seq) framework, enabling end-to-end training for tasks like machine translation. It demonstrated t...

3. Score: 0.746 | Title: Attention Is All You Need
   Link: https://arxiv.org/abs/1706.03762
   Significance: This work introduced the Transformer architecture, which relies solely on attention mechanisms. It eliminated the need for recurrent and convolutional...

User: exit
Exiting.
```
# Changing the top_k = 5. (This will now return the top 5 scores for both the transcript and knowledge base indexes.)
Running -
```bash
python src/semantic_audio_search.py \
    --transcript_index "path/to/transcript.index" \
    --transcript_metadata "path/to/transcript_metadata.json" \
    --kb_index "path/to/knowledge_base.index" \
    --kb_metadata "path/to/knowledge_base_metadata.json" \
    --top_k 5
```
```
Chatbot (type 'exit' to quit)

User: Why is noise needed?

Transcript Results:
1. Score: 0.866 | Timestamp: 00:06:19
   Text: Noising....

2. Score: 0.848 | Timestamp: 00:10:27
   Text: Three. Noise subtraction....

3. Score: 0.840 | Timestamp: 00:09:58
   Text: One. Start with noise. The process begins with random noise, latent noise....

4. Score: 0.838 | Timestamp: 00:08:14
   Text: One. Noise prediction....

5. Score: 0.834 | Timestamp: 00:06:31
   Text: This is the noising part of the process, and the model learns how to reverse this process....

Knowledge Base Results:
1. Score: 0.748 | Title: Dropout: A Simple Way to Prevent Neural Networks from Overfitting
   Link: https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf
   Significance: This paper introduced dropout as a novel regularization technique to prevent overfitting in neural networks. It works by randomly deactivating a subse...

2. Score: 0.737 | Title: Attention Is All You Need
   Link: https://arxiv.org/abs/1706.03762
   Significance: This work introduced the Transformer architecture, which relies solely on attention mechanisms. It eliminated the need for recurrent and convolutional...

3. Score: 0.730 | Title: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks
   Link: https://arxiv.org/abs/1803.03635
   Significance: This work proposed that within large, over-parameterized neural networks exist smaller subnetworks, or 'winning tickets', that can be trained effectiv...

4. Score: 0.720 | Title: Generative Adversarial Nets
   Link: https://arxiv.org/abs/1406.2661
   Significance: This paper proposed the generative adversarial network (GAN) framework, which pits two neural networks against each other in a game-theoretic scenario...

5. Score: 0.718 | Title: Neural Machine Translation by Jointly Learning to Align and Translate
   Link: https://arxiv.org/abs/1409.0473
   Significance: This paper extended the seq2seq model by incorporating an attention mechanism to improve translation quality. The attention mechanism allowed the mode...

User: exit
Exiting.

```

# Future Enhancements # 
- I am currently enrolled in NVIDIA Riva training (Feb 25), where I will explore real-time ASR (Automatic Speech Recognition) and speech AI pipelines.

- Future work will include replacing OpenAI Whisper with Riva's optimized speech-to-text engine to enhance real-time processing.


## Authors

Aaron Rodriguez

## Version History

* 0.1
    * Initial Release

