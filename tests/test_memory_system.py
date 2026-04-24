from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_memory_system(monkeypatch):
    @dataclass
    class Memory:
        id: str
        content: str
        user_id: str
        importance: float = 0.5
        metadata: dict = field(default_factory=dict)

    @dataclass
    class MemorySearchResult:
        memory: Memory
        score: float

        def to_dict(self):
            return {"id": self.memory.id, "content": self.memory.content, "score": self.score}

    models_module = types.ModuleType("headroom.memory.models")
    models_module.Memory = Memory
    ports_module = types.ModuleType("headroom.memory.ports")
    ports_module.MemorySearchResult = MemorySearchResult
    tools_module = types.ModuleType("headroom.memory.tools")
    tools_module.MEMORY_TOOLS = [{"name": "basic"}]
    tools_module.MEMORY_TOOLS_OPTIMIZED = [{"name": "optimized"}]

    monkeypatch.setitem(sys.modules, "headroom.memory.models", models_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.ports", ports_module)
    monkeypatch.setitem(sys.modules, "headroom.memory.tools", tools_module)

    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "headroom" / "memory" / "system.py"
    monkeypatch.delitem(sys.modules, "headroom.memory.system", raising=False)
    spec = importlib.util.spec_from_file_location("headroom.memory.system", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["headroom.memory.system"] = module
    spec.loader.exec_module(module)
    return module, Memory, MemorySearchResult


class FakeBackend:
    def __init__(self, Memory, MemorySearchResult):
        self.Memory = Memory
        self.MemorySearchResult = MemorySearchResult
        self.supports_graph = True
        self.supports_vector_search = False
        self.saved_kwargs = None
        self.save_result = Memory(id="m1", content="saved", user_id="u1", importance=0.8)
        self.search_results = []
        self.memories = {}
        self.delete_result = True
        self.task_status = {"status": "completed"}
        self.pending_tasks = ["t1"]
        self.flush_result = {"completed": 1, "failed": 0, "pending": 0}

    async def save_memory(self, **kwargs):
        self.saved_kwargs = kwargs
        if kwargs.get("content") == "type-error" and ("facts" in kwargs or "background" in kwargs):
            raise TypeError("unsupported kwargs")
        return self.save_result

    async def search_memories(self, **kwargs):
        self.search_kwargs = kwargs
        return self.search_results

    async def update_memory(self, **kwargs):
        self.update_kwargs = kwargs
        if kwargs["memory_id"] == "bad-update":
            raise ValueError("cannot update")
        return self.Memory(
            id=kwargs["memory_id"],
            content=kwargs["new_content"],
            user_id=kwargs.get("user_id") or "u1",
        )

    async def delete_memory(self, **kwargs):
        self.delete_kwargs = kwargs
        return self.delete_result

    async def get_memory(self, memory_id):
        return self.memories.get(memory_id)

    async def close(self):
        self.closed = True

    def get_task_status(self, task_id):
        return self.task_status | {"task_id": task_id}

    def get_pending_tasks(self):
        return list(self.pending_tasks)

    async def wait_for_task(self, task_id, timeout):
        return {"status": "completed", "task_id": task_id, "timeout": timeout}

    async def flush_pending(self, timeout):
        return self.flush_result | {"timeout": timeout}


@pytest.mark.asyncio
async def test_memory_system_dispatch_and_handlers(monkeypatch) -> None:
    system_module, Memory, MemorySearchResult = _load_memory_system(monkeypatch)
    backend = FakeBackend(Memory, MemorySearchResult)
    system = system_module.MemorySystem(backend, user_id="u1", session_id="s1")

    assert system.get_tools() == [{"name": "basic"}]
    assert system.get_tools(optimized=True) == [{"name": "optimized"}]

    unknown = await system.process_tool_call("bad_tool", {})
    assert unknown["success"] is False

    invalid_importance = await system.handle_memory_save("x", 1.5)
    assert invalid_importance["success"] is False

    backend.save_result = Memory(id="m1", content="saved", user_id="u1", importance=0.8)
    saved = await system.handle_memory_save(
        "saved",
        0.8,
        entities=["Python"],
        relationships=[{"source": "a", "relationship": "likes", "destination": "b"}],
        metadata={"k": 1},
        facts=["fact1", "fact2"],
        extracted_entities=[{"entity": "Python", "entity_type": "language"}],
        extracted_relationships=[{"source": "a", "relationship": "likes", "destination": "b"}],
        background=True,
    )
    assert saved["optimized"] is True
    assert saved["fact_count"] == 2
    assert backend.saved_kwargs["background"] is True

    backend.save_result = Memory(
        id="m2",
        content="queued",
        user_id="u1",
        importance=0.7,
        metadata={"_async": True, "_task_id": "task-1", "_status": "processing"},
    )
    queued = await system.handle_memory_save("queued", 0.7)
    assert queued["async"] is True
    assert queued["task_id"] == "task-1"

    typed_fallback = await system.handle_memory_save("type-error", 0.5, metadata={"z": 1})
    assert typed_fallback["memory_id"] == "m2"

    no_results = await system.handle_memory_search("none", top_k=0)
    assert no_results["count"] == 0
    assert backend.search_kwargs["top_k"] == 1

    backend.search_results = [
        MemorySearchResult(memory=Memory(id="m3", content="python", user_id="u1"), score=0.9)
    ]
    found = await system.handle_memory_search(
        "python", entities=["Python"], include_related=True, top_k=99
    )
    assert found["count"] == 1
    assert backend.search_kwargs["top_k"] == 50

    backend.memories["missing"] = None
    update_missing = await system.handle_memory_update("missing", "new", "because")
    assert update_missing["success"] is False

    backend.memories["other"] = Memory(id="other", content="x", user_id="u2")
    update_denied = await system.handle_memory_update("other", "new", "because")
    assert update_denied["error"] == "Permission denied"

    backend.memories["bad-update"] = Memory(id="bad-update", content="old", user_id="u1")
    update_error = await system.handle_memory_update("bad-update", "new", "because")
    assert update_error["success"] is False

    backend.memories["mine"] = Memory(id="mine", content="old", user_id="u1")
    updated = await system.handle_memory_update("mine", "new", "because")
    assert updated["old_content"] == "old"
    assert updated["new_content"] == "new"

    delete_missing = await system.handle_memory_delete("missing", "cleanup")
    assert delete_missing["success"] is False
    delete_denied = await system.handle_memory_delete("other", "cleanup")
    assert delete_denied["error"] == "Permission denied"
    backend.memories["mine-delete"] = Memory(id="mine-delete", content="gone", user_id="u1")
    backend.delete_result = False
    delete_failed = await system.handle_memory_delete("mine-delete", "cleanup")
    assert delete_failed["success"] is False
    backend.delete_result = True
    deleted = await system.handle_memory_delete("mine-delete", "cleanup")
    assert deleted["deleted_content"] == "gone"

    dispatch_save = await system._handle_save({"content": "saved", "importance": 0.2})
    assert dispatch_save["success"] is True
    dispatch_search = await system._handle_search({"query": "python"})
    assert dispatch_search["success"] is True
    dispatch_update = await system._handle_update({"memory_id": "mine", "new_content": "newer"})
    assert dispatch_update["reason"] == "Updated by user"
    dispatch_delete = await system._handle_delete({"memory_id": "mine-delete"})
    assert dispatch_delete["reason"] == "Deleted by user"

    boom_system = system_module.MemorySystem(backend, user_id="u1")

    async def boom(arguments):
        raise RuntimeError("boom")

    monkeypatch.setattr(boom_system, "_handle_search", boom)
    failed = await boom_system.process_tool_call("memory_search", {"query": "x"})
    assert failed["success"] is False


@pytest.mark.asyncio
async def test_memory_system_properties_and_async_task_helpers(monkeypatch) -> None:
    system_module, Memory, MemorySearchResult = _load_memory_system(monkeypatch)
    backend = FakeBackend(Memory, MemorySearchResult)
    system = system_module.MemorySystem(backend, user_id="u1", session_id="s1")

    assert system.user_id == "u1"
    assert system.session_id == "s1"
    assert system.backend is backend
    assert system.supports_graph is True
    assert system.supports_vector_search is False
    assert system.get_task_status("task")["task_id"] == "task"
    assert system.get_pending_tasks() == ["t1"]
    assert (await system.wait_for_task("task", timeout=2.5))["timeout"] == 2.5
    assert (await system.flush_pending(timeout=5))["timeout"] == 5

    bare_backend = SimpleNamespace(supports_graph=False, supports_vector_search=True)
    bare = system_module.MemorySystem(bare_backend, user_id="u2")
    assert bare.get_task_status("task")["status"] == "not_supported"
    assert bare.get_pending_tasks() == []
    assert (await bare.wait_for_task("task"))["status"] == "not_supported"
    assert (await bare.flush_pending())["completed"] == 0
