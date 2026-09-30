"""Synthetic component integration for #92611 and PR #92620.

Real SessionDB, SessionStore, BasePlatformAdapter admission, resolver and durable
claims. Only external transport and the result-consuming model handler are fake.
No gateway service, model call, or real user database is involved.
"""

import asyncio
import queue
from collections import OrderedDict
import threading
import time
from unittest.mock import patch

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter
from gateway.run import GatewayRunner
from gateway.session import AsyncSessionStore, SessionSource, SessionStore
from hermes_state import AsyncSessionDB
from tools import async_delegation as ad


class CaptureAdapter(BasePlatformAdapter):
    async def connect(self, *, is_reconnect=False):
        pass

    async def disconnect(self):
        pass

    async def send(self, chat_id, text, **kwargs):
        return None

    async def get_chat_info(self, chat_id):
        return {}


@pytest.fixture
def context(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    ad._reset_for_tests()
    config = GatewayConfig()
    store = SessionStore(tmp_path / "sessions", config)
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="synthetic-chat",
        chat_type="dm",
        user_id="synthetic-user",
    )
    parent = store.get_or_create_session(source)
    db = store._db
    adapter = CaptureAdapter(PlatformConfig(enabled=True), Platform.TELEGRAM)
    runner = object.__new__(GatewayRunner)
    runner.config = config
    runner.session_store = store
    runner._async_session_store = AsyncSessionStore(store)
    runner._session_db = AsyncSessionDB(db)
    runner.adapters = {Platform.TELEGRAM: adapter}
    runner._session_source_cache = {}
    runner._completion_delivery_lock = threading.Lock()
    runner._completion_deliveries_inflight = set()
    runner._completion_deliveries_delivered = OrderedDict()
    runner._completion_delivery_retention = 2048
    runner._background_tasks = set()
    consumed = []
    handled = asyncio.Event()

    async def handler(event):
        try:
            current = store.get_or_create_session(source)
            resolved = await runner._resolve_async_delegation_session(
                current, event.metadata["gateway_session_id"]
            )
            if resolved is not None:
                consumed.append(resolved.session_id)
            return None
        finally:
            handled.set()

    adapter.set_message_handler(handler)
    yield runner, store, db, parent, source, consumed, handled
    ad._reset_for_tests()
    db.close()


async def deliver(context, provenance, event_type="async_delegation"):
    runner, store, db, parent, source, consumed, handled = context
    child = "synthetic-child"
    target = parent.session_id
    if provenance == "missing":
        target = "missing-parent"
    elif provenance == "cycle":
        target = child
    elif provenance == "missing_marker":
        target = None
    elif provenance in {
        "invalid_false",
        "invalid_zero",
        "invalid_list",
        "invalid_dict",
        "invalid_empty",
    }:
        target = {
            "invalid_false": False,
            "invalid_zero": 0,
            "invalid_list": [],
            "invalid_dict": {},
            "invalid_empty": "",
        }[provenance]
    elif provenance == "reset":
        store.reset_session(parent.session_key)
    if provenance == "nested":
        db.create_session(
            "synthetic-middle",
            source="subagent",
            parent_session_id=parent.session_id,
            model_config={"_delegate_from": parent.session_id},
        )
        target = "synthetic-middle"
    db.create_session(
        child,
        source="subagent",
        parent_session_id=parent.session_id,
        model_config={}
        if provenance == "missing_marker"
        else {"_delegate_from": target},
    )
    if provenance == "compressed":
        tip = parent.session_id + "-tip"
        db.publish_compression_child(
            parent_session_id=parent.session_id,
            child_session_id=tip,
            source="telegram",
            require_compression_lease=False,
            model="offline",
            model_config={},
            system_prompt="offline",
            messages=[{"role": "user", "content": "retained"}],
        )
        assert store.advance_compression_session(
            parent.session_key, parent.session_id, tip
        )
    event = {
        "type": event_type,
        "delegation_id": "synthetic-result",
        "session_id": "synthetic-process",
        "started_at": time.time(),
        "exit_code": 0,
        "session_key": parent.session_key,
        "parent_session_id": child,
        "goal": "synthetic routing test",
        "status": "completed",
        "summary": "synthetic result",
        "dispatched_at": time.time(),
        "completed_at": time.time(),
    }
    if event_type == "async_delegation":
        ad._persist_dispatch(event)
        ad._persist_completion(
            event, {"status": "completed", "summary": event["summary"]}
        )
    result = await runner._deliver_completion_notification(
        "synthetic completion", dict(event)
    )
    if result is True:
        await asyncio.wait_for(handled.wait(), timeout=10)
    row = ad.get_durable_delegation(event["delegation_id"])
    assert (row is not None) == (event_type == "async_delegation")
    route = store.get_or_create_session(source).session_id
    runner._completion_deliveries_delivered.clear()
    restored = ad.restore_undelivered_completions(queue.Queue())
    return result, row, route, restored


@pytest.mark.asyncio
@pytest.mark.parametrize("provenance", ["valid", "nested", "compressed"])
@pytest.mark.parametrize("event_type", ["async_delegation", "completion"])
async def test_child_completion_keeps_parent_and_prevents_replay(
    context, provenance, event_type
):
    original_parent_id = context[3].session_id
    result, row, route, restored = await deliver(context, provenance, event_type)
    _, _, db, parent, _, consumed, _ = context
    # SessionStore advances this entry in place on compression.
    expected = parent.session_id
    assert route == expected, f"route hijacked: {route}; consumed={consumed}"
    assert consumed == [expected]
    assert db.get_session(expected)["ended_at"] is None
    if provenance == "compressed":
        assert db.get_session(original_parent_id)["end_reason"] == "compression"
    assert result is True
    if event_type == "async_delegation":
        assert row["delivery_state"] == "delivered"
    assert restored == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provenance",
    [
        "missing",
        "cycle",
        "reset",
        "unavailable",
        "unavailable_after_claim",
        "invalid_false",
        "invalid_zero",
        "invalid_list",
        "invalid_dict",
        "invalid_empty",
        "missing_marker",
    ],
)
async def test_invalid_or_unavailable_provenance_has_honest_disposition(
    context, provenance
):
    if provenance in {"unavailable", "unavailable_after_claim"}:
        runner = context[0]
        original = runner._session_db.get_session
        claim = ad.claim_completion_delivery
        claimed = False

        def claim_then_fail(delegation_id, claim_id):
            nonlocal claimed
            result = claim(delegation_id, claim_id)
            claimed = result
            return result

        async def temporarily_unavailable(session_id):
            if session_id == context[3].session_id and (
                provenance == "unavailable" or claimed
            ):
                raise RuntimeError("synthetic parent DB unavailable")
            return await original(session_id)

        with (
            patch.object(runner._session_db, "get_session", temporarily_unavailable),
            patch.object(ad, "claim_completion_delivery", claim_then_fail),
        ):
            result, row, route, restored = await deliver(context, provenance)
        assert result is False
        assert (row["delivery_state"], row["delivery_attempts"], restored) == (
            "pending",
            0,
            1,
        )
        assert not context[5]
        assert route == context[3].session_id
        pending = queue.Queue()
        assert ad.restore_undelivered_completions(pending) == 1
        recovered_event = pending.get_nowait()
        assert (
            await runner._deliver_completion_notification(
                "recovered result", recovered_event
            )
            is True
        )
        await asyncio.wait_for(context[6].wait(), timeout=10)
        recovered = ad.get_durable_delegation(recovered_event["delegation_id"])
        assert recovered is not None
        assert (recovered["delivery_state"], recovered["delivery_attempts"]) == (
            "delivered",
            1,
        )
        assert context[5] == [context[3].session_id]
        return
    result, row, route, restored = await deliver(context, provenance)
    _, _, db, parent, _, consumed, _ = context
    assert not consumed, f"unverified result consumed in {consumed}"
    if provenance != "reset":
        assert route == parent.session_id
        assert db.get_session(parent.session_id)["ended_at"] is None
    else:
        assert route != parent.session_id
    assert row["delivery_state"] == "dropped", (
        f"no handler consumed result, but durable state={row['delivery_state']}; admission={result}; replay={restored}"
    )
    assert restored == 0
