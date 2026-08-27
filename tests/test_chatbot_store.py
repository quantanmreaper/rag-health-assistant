"""
Property tests for ConversationStore (Properties 1-4, 10-12, 26-28, 30-32).
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from rag_health_assistant.chatbot.conversation_store import ConversationStore
from rag_health_assistant.chatbot.json_io import sanitize_id
from rag_health_assistant.chatbot.models import Message, MessageRole, utc_now


@pytest.fixture
def store(tmp_path: Path) -> ConversationStore:
    return ConversationStore(storage_dir=tmp_path / "conversations")


def _msg(content: str, role: MessageRole = MessageRole.USER, mid: str | None = None) -> Message:
    return Message(
        message_id=mid or f"msg_{abs(hash(content)) % 10**8}",
        role=role,
        content=content,
        timestamp=utc_now(),
    )


@given(content=st.text(min_size=0, max_size=200))
@settings(max_examples=25, deadline=None)
def test_property_1_message_persistence_round_trip(tmp_path_factory, content):
    store = ConversationStore(storage_dir=tmp_path_factory.mktemp("c") / "conv")
    conv = store.create_conversation("patient_a", is_anonymous=True)
    msg = _msg(content, mid="msg_rt_1")
    store.append_message(conv.metadata.conversation_id, "patient_a", msg)
    loaded = store.load_conversation(conv.metadata.conversation_id, "patient_a")
    assert loaded is not None
    assert loaded.messages[-1].content == content
    assert loaded.messages[-1].role == MessageRole.USER


def test_property_2_conversation_structure_completeness(store: ConversationStore):
    conv = store.create_conversation("patient_b", title="BP questions")
    data = store.load_conversation(conv.metadata.conversation_id, "patient_b")
    assert data is not None
    meta = data.metadata
    assert meta.conversation_id
    assert meta.patient_id == "patient_b"
    assert meta.created_at
    assert meta.updated_at
    assert isinstance(data.messages, list)


def test_property_28_utf8_encoding_preservation(store: ConversationStore):
    conv = store.create_conversation("patient_utf")
    text = "Glucose ≤ 70 mg/dL — 血糖 😊 café naïve"
    store.append_message(
        conv.metadata.conversation_id,
        "patient_utf",
        _msg(text, mid="msg_utf"),
    )
    loaded = store.load_conversation(conv.metadata.conversation_id, "patient_utf")
    assert loaded.messages[-1].content == text


def test_property_3_patient_association(store: ConversationStore):
    conv = store.create_conversation("patient_x")
    store.append_message(conv.metadata.conversation_id, "patient_x", _msg("hello", mid="m1"))
    loaded = store.load_conversation(conv.metadata.conversation_id, "patient_x")
    assert loaded.metadata.patient_id == "patient_x"
    assert store.load_conversation(conv.metadata.conversation_id, "other") is None


def test_property_4_timestamp_monotonicity(store: ConversationStore):
    conv = store.create_conversation("patient_t")
    cid = conv.metadata.conversation_id
    t0 = conv.metadata.updated_at
    store.append_message(cid, "patient_t", _msg("one", mid="m1"))
    mid = store.load_conversation(cid, "patient_t").metadata.updated_at
    store.append_message(cid, "patient_t", _msg("two", mid="m2"))
    t2 = store.load_conversation(cid, "patient_t").metadata.updated_at
    assert mid >= t0
    assert t2 >= mid


def test_property_31_file_naming_convention(store: ConversationStore):
    conv = store.create_conversation("patient_file")
    path = store._filepath("patient_file", conv.metadata.conversation_id)
    assert path.name == f"{sanitize_id('patient_file')}_{sanitize_id(conv.metadata.conversation_id)}.json"
    assert path.exists()


def test_property_30_malformed_json_read(store: ConversationStore):
    conv = store.create_conversation("patient_bad")
    path = store._filepath("patient_bad", conv.metadata.conversation_id)
    path.write_text("{not valid json", encoding="utf-8")
    assert store.load_conversation(conv.metadata.conversation_id, "patient_bad") is None


def test_property_27_filesystem_error_handling(store: ConversationStore, monkeypatch):
    conv = store.create_conversation("patient_fs")

    def boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(
        "rag_health_assistant.chatbot.conversation_store.atomic_write_json", boom
    )
    with pytest.raises(OSError):
        store.save_conversation(conv)


def test_property_26_concurrent_writes(store: ConversationStore):
    conv = store.create_conversation("patient_conc")
    cid = conv.metadata.conversation_id

    def write_one(i: int):
        store.append_message(cid, "patient_conc", _msg(f"msg-{i}", mid=f"msg_{i}"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_one, range(20)))

    loaded = store.load_conversation(cid, "patient_conc")
    assert loaded is not None
    assert loaded.metadata.message_count == len(loaded.messages)
    assert len(loaded.messages) == 20


def test_property_10_11_12_sliding_window(store: ConversationStore):
    conv = store.create_conversation("patient_win")
    cid = conv.metadata.conversation_id
    now = datetime.now(timezone.utc)

    for i in range(50):
        msg = Message(
            message_id=f"msg_{i}",
            role=MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT,
            content=f"content {i}",
            timestamp=now - timedelta(days=10 if i < 10 else 0, minutes=50 - i),
        )
        store.append_message(cid, "patient_win", msg)

    window = store.get_conversation_window(cid, "patient_win", window_size=20, window_days=7)
    assert len(window) <= 20
    # All within 7 days
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    for m in window:
        ts = m.timestamp if m.timestamp.tzinfo else m.timestamp.replace(tzinfo=timezone.utc)
        assert ts >= cutoff
    # Chronological
    timestamps = [m.timestamp for m in window]
    assert timestamps == sorted(timestamps)

    full = store.load_conversation(cid, "patient_win")
    assert len(full.messages) == 50  # Property 12: storage independent of window


def test_property_32_json_human_readable(store: ConversationStore):
    conv = store.create_conversation("patient_pretty")
    path = store._filepath("patient_pretty", conv.metadata.conversation_id)
    text = path.read_text(encoding="utf-8")
    assert "\n" in text
    assert '  "' in text or '{\n' in text


def test_auto_title_from_first_message(store: ConversationStore):
    conv = store.create_conversation("patient_title")
    store.append_message(
        conv.metadata.conversation_id,
        "patient_title",
        _msg("What is my target blood pressure?", mid="m1"),
    )
    loaded = store.load_conversation(conv.metadata.conversation_id, "patient_title")
    assert loaded.metadata.title.startswith("What is my target")
