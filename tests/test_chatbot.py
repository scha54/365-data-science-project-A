from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from chatbot import build_store, create_chain, transcript_documents


class TopicEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        return [1.0, 0.0] if 'python' in text.lower() else [0.0, 1.0]


def test_index_persistence_retrieval_and_chain(tmp_path):
    segments = [{"text": "Python is used for data analysis.", "start": 12.0}] * 20
    segments += [{"text": "Gardening needs sunlight.", "start": 90.0}] * 24
    store = build_store(segments, tmp_path, TopicEmbeddings())
    restored = build_store(segments, tmp_path, TopicEmbeddings())
    assert len(list(tmp_path.glob('*.json'))) == 1
    assert 'Python' in restored.similarity_search('Python', k=1)[0].page_content
    result = create_chain(store, FakeListChatModel(responses=['Python supports data analysis [1].'])).invoke('What is Python used for?')
    assert result['answer'] == 'Python supports data analysis [1].'
    assert 'Python' in result['context']
    assert result['documents'][0].metadata['source'].endswith('&t=12s')


def test_empty_transcript_rejected():
    import pytest
    with pytest.raises(ValueError):
        transcript_documents([])


def test_chunk_size_and_timestamp():
    chunks = transcript_documents([{'text': 'A long caption. ' * 500, 'start': 42}])
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 1200 for c in chunks)
    assert all(c.metadata['start'] == 42 for c in chunks)


def test_fetch_uses_selected_video(monkeypatch):
    from unittest.mock import Mock
    import chatbot
    transcript = Mock()
    transcript.fetch.return_value.to_raw_data.return_value = [{'text': 'Caption', 'start': 0}]
    api = Mock()
    api.list.return_value.find_transcript.return_value = transcript
    monkeypatch.setattr(chatbot, 'YouTubeTranscriptApi', lambda: api)
    assert chatbot.fetch_transcript() == [{'text': 'Caption', 'start': 0}]
    api.list.assert_called_once_with('6Fa91SY9Gnw')
