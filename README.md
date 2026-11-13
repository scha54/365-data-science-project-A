# YouTube transcript chatbot

A LangChain retrieval-augmented chatbot for https://www.youtube.com/watch?v=6Fa91SY9Gnw.

## Run

Requires Python 3.12 (the tested locked environment) and an OpenAI API key with access to `text-embedding-3-small` and `gpt-4o-mini` (API usage incurs charges).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock.txt
streamlit run app.py --server.address 0.0.0.0
```

Enter your key in the sidebar, or set `OPENAI_API_KEY` in your environment. Click **Load video transcript**, then ask questions. Ask standalone questions: displayed chat history is not used to rewrite follow-up questions.

## Pipeline

1. `youtube-transcript-api` fetches English captions, or translates available captions to English. It requires captions and YouTube access; hosted IPs may be blocked by YouTube.
2. Neighboring captions are grouped with overlap, then split into chunks of at most 1,200 characters, with 200-character overlap. Each chunk keeps a link to the beginning of its caption group (approximate timestamp).
3. OpenAI creates embeddings. LangChain's `InMemoryVectorStore` performs cosine similarity search and persists the embeddings and text as JSON in `.cache/`. Cache keys include the transcript, embedding model and chunking settings. Delete `.cache/` to rebuild.
4. A LangChain runnable chain retrieves the four closest chunks and passes their text as context to the chat model. Answers include numbered citations, with timestamp links displayed below.

Transcript content is sent to OpenAI when indexed. Questions and retrieved passages are sent when answering. Local caches contain transcript text and vectors; do not share caches containing private content. The model is instructed to abstain when evidence is missing, but generated answers should still be checked against the video.

If YouTube access is unavailable, upload a JSON transcript of this same video through the sidebar:

```json
[{"text": "Actual caption text", "start": 0.0, "duration": 3.2}]
```

Do not use example or invented captions as the video's transcript.

In a restricted cloud environment, permit `www.youtube.com`, `youtube.com`, `*.youtube.com`, `api.openai.com`, and `openaipublic.blob.core.windows.net` (tokenizer assets). The current cloud instance blocks YouTube; live captions and OpenAI responses have not been verified here.

## Validate

```bash
pip install pytest
python -m pytest -q
```

Tests exercise chunking, embedding persistence, similarity retrieval, context assembly and the LangChain response pipeline using local deterministic doubles. They do not validate OpenAI or live YouTube access.
