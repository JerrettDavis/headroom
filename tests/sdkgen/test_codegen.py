"""Offline unit/contract tests. Run with unittest, independent of root conftest.py."""
from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from tools.sdkgen.compiler import Compiler, ContractError, canonical, load_sources
from tools.sdkgen.emit import render
from tools.sdkgen.__main__ import compare, main

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads((ROOT / "sdk/codegen/config.json").read_text())


def compiler() -> Compiler:
    sources = CONFIG["sources"] if (ROOT / "headroom/proxy/server.py").exists() else CONFIG["fixture_sources"]
    return Compiler(load_sources(ROOT, sources))


def load_generated(root: Path):
    package = root / "python"
    spec = importlib.util.spec_from_file_location("generated_test", package / "__init__.py", submodule_search_locations=[str(package)])
    module = importlib.util.module_from_spec(spec)
    for key in list(sys.modules):
        if key == "generated_test" or key.startswith("generated_test."):
            del sys.modules[key]
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, sys.modules["generated_test.runtime"]


BASE = '''
from typing import TypedDict, NotRequired, Required, Literal
class Input(TypedDict):
    hash: str
class Output(TypedDict):
    value: str | None
@app.post("/v1/example")
@sdk_operation(operation_id="example", request=Input, response=Output, access="loopback")
async def endpoint(request: Request):
    pass
'''


class SourceCompilerTests(unittest.TestCase):
    def test_real_pilot_paths_and_types(self):
        doc = compiler().compile()
        self.assertEqual(set(doc["paths"]), {"/v1/retrieve", "/v1/retrieve/{hash_key}"})
        model = doc["components"]["schemas"]["RetrieveResponse"]
        self.assertIn("tool_name", model["required"])
        self.assertEqual(model["properties"]["tool_name"]["anyOf"], [{"type": "string"}, {"type": "null"}])
        self.assertEqual(set(model["properties"]), {"hash", "original_content", "original_tokens", "original_item_count", "compressed_item_count", "tool_name", "retrieval_count"})

    def test_imports_and_bodies_never_execute(self):
        doc = Compiler({"x.py": 'raise Exception("must not run")\n' + BASE}).compile()
        self.assertIn("/v1/example", doc["paths"])

    def test_shuffled_source_files_are_byte_identical(self):
        source = load_sources(ROOT, CONFIG["fixture_sources"])
        first = render(Compiler(source).compile())
        second = render(Compiler(dict(reversed(list(source.items())))).compile())
        self.assertEqual(first, second)

    def test_comments_and_lines_do_not_change_contract(self):
        a = Compiler({"x.py": BASE}).compile()
        b = Compiler({"renamed/file.py": "# comment\n\n" + BASE}).compile()
        self.assertEqual(canonical(a), canonical(b))

    def test_field_order_does_not_change_contract(self):
        a = BASE.replace("hash: str", "hash: str\n    note: NotRequired[str]")
        b = BASE.replace("hash: str", "note: NotRequired[str]\n    hash: str")
        self.assertEqual(canonical(Compiler({"a.py": a}).compile()), canonical(Compiler({"b.py": b}).compile()))

    def test_optional_nullable_distinguished(self):
        source = BASE.replace("hash: str", "hash: str\n    note: NotRequired[str | None]\n    enabled: NotRequired[bool]")
        doc = Compiler({"x.py": source}).compile()
        model = doc["components"]["schemas"]["Input"]
        self.assertEqual(model["required"], ["hash"])
        self.assertIn("anyOf", model["properties"]["note"])
        self.assertIn(b"Note Optional[string]", render(doc)["go/models.go"])
        self.assertIn(b"Enabled *bool", render(doc)["go/models.go"])

    def test_total_false_and_required(self):
        source = BASE.replace("class Input(TypedDict):", "class Input(TypedDict, total=False):").replace("hash: str", "hash: Required[str]\n    note: str")
        self.assertEqual(Compiler({"x.py": source}).compile()["components"]["schemas"]["Input"]["required"], ["hash"])

    def test_qualified_types_are_not_guessed(self):
        for annotation in ("other.str", "typing.Optional[str]", "other.RetrieveRequest"):
            with self.subTest(annotation=annotation), self.assertRaises(ContractError):
                Compiler({"x.py": BASE.replace("hash: str", "hash: " + annotation)}).compile()

    def test_unsafe_integer_enum_rejected(self):
        with self.assertRaisesRegex(ContractError, "safe range"):
            Compiler({"x.py": BASE.replace("hash: str", "hash: Literal[9007199254740993]")}).compile()

    def test_any_rejected(self):
        with self.assertRaisesRegex(ContractError, "Any"):
            Compiler({"x.py": BASE.replace("hash: str", "hash: Any")}).compile()

    def test_intentional_json_value_supported(self):
        doc = Compiler({"x.py": BASE.replace("hash: str", "hash: JsonValue")}).compile()
        self.assertIn("x-headroom-opaque", doc["components"]["schemas"]["Input"]["properties"]["hash"])

    def test_unknown_model_rejected(self):
        with self.assertRaises(ContractError):
            Compiler({"x.py": BASE.replace("hash: str", "hash: Unresolved")}).compile()

    def test_arbitrary_union_rejected(self):
        with self.assertRaises(ContractError):
            Compiler({"x.py": BASE.replace("hash: str", "hash: str | int")}).compile()

    def test_recursion_rejected(self):
        with self.assertRaisesRegex(ContractError, "Recursive"):
            Compiler({"x.py": BASE.replace("hash: str", "hash: Input")}).compile()

    def test_executable_default_rejected(self):
        with self.assertRaises(ContractError):
            Compiler({"x.py": BASE.replace("hash: str", "hash: str = dangerous()")}).compile()

    def test_duplicate_operation_rejected(self):
        extra = '\n@app.get("/other")\n@sdk_operation(operation_id="example", response=Output, access="loopback")\ndef other(): pass\n'
        with self.assertRaisesRegex(ContractError, "Duplicate operation"):
            Compiler({"x.py": BASE + extra}).compile()

    def test_dynamic_route_rejected_and_inventoried(self):
        c = Compiler({"x.py": BASE.replace('"/v1/example"', 'PREFIX + "/example"')})
        self.assertTrue(c.inventory()["unresolved"])
        with self.assertRaises(ContractError):
            c.compile()

    def test_hidden_route_candidate_is_not_silently_exported(self):
        c = Compiler({"x.py": BASE + '\n@app.get("/debug/secret", include_in_schema=False)\ndef hidden(): pass\n'})
        report = c.inventory()
        self.assertEqual(len(report["routes"]), 2)
        self.assertEqual(sum(r["exported"] for r in report["routes"]), 1)
        self.assertNotIn("/debug/secret", c.compile()["paths"])

    def test_dynamic_registration_reported(self):
        report = Compiler({"x.py": BASE + '\napp.include_router(router, prefix="/v2")\napp.mount("/other", other)\n'}).inventory()
        self.assertEqual(len(report["unresolved"]), 2)

    def test_unannotated_query_parameter_rejected(self):
        with self.assertRaisesRegex(ContractError, "adapter"):
            Compiler({"x.py": BASE.replace("request: Request", "request: Request, limit: int = 10")}).compile()

    def test_stream_response_rejected(self):
        with self.assertRaisesRegex(ContractError, "streaming"):
            Compiler({"x.py": BASE.replace('@app.post("/v1/example")', '@app.post("/v1/example", response_class=StreamingResponse)')}).compile()

    def test_handler_return_conflict_rejected(self):
        with self.assertRaisesRegex(ContractError, "return annotation"):
            Compiler({"x.py": BASE.replace("async def endpoint(request: Request):", "async def endpoint(request: Request) -> Input:")}).compile()

    def test_reserved_generated_names_rejected(self):
        with self.assertRaisesRegex(ContractError, "Reserved"):
            render(Compiler({"x.py": BASE.replace('operation_id="example"', 'operation_id="constructor"')}).compile())

    def test_cross_language_property_collision_rejected(self):
        with self.assertRaisesRegex(ContractError, "collision"):
            render(Compiler({"x.py": BASE.replace("hash: str", "hash: str\n    extra_key: str\n    extraKey: str")}).compile())

    def test_nonfinite_canonical_json_rejected(self):
        with self.assertRaises(ValueError):
            canonical({"value": float("nan")})

    def test_marker_preserves_handler_identity(self):
        source = ROOT / "headroom/proxy/sdk_contracts.py"
        spec = importlib.util.spec_from_file_location("contract_marker", source)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        def fn(): return 123
        marked = module.sdk_operation(operation_id="example", response=dict, access="loopback")(fn)
        self.assertIs(fn, marked)
        self.assertEqual(marked(), 123)
        self.assertFalse(hasattr(marked, "__wrapped__"))

    def test_manifest_hashes_are_correct(self):
        output = render(compiler().compile())
        manifest = json.loads(output["manifest.json"])
        for file, sha in manifest["files"].items():
            self.assertEqual(hashlib.sha256(output[file]).hexdigest(), sha)
        self.assertNotIn("timestamp", manifest)
        self.assertNotIn("source_commit", manifest)

    def test_diff_is_conservative(self):
        before = compiler().compile(); after = copy.deepcopy(before)
        self.assertEqual(compare(before, after), [])
        after["paths"].pop("/v1/retrieve")
        self.assertEqual(compare(before, after)[0]["severity"], "breaking")
        after = copy.deepcopy(before)
        after["components"]["schemas"]["RetrieveRequest"]["properties"]["optional_field"] = {"type": "string"}
        self.assertEqual(compare(before, after)[0]["severity"], "review")
        after = copy.deepcopy(before); after["info"]["description"] = "docs only"
        self.assertEqual(compare(before, after), [])

    def test_drift_and_stale_owned_file_detection(self):
        with tempfile.TemporaryDirectory() as td:
            args = ["--root", str(ROOT), "--fixture", "--out", td]
            self.assertEqual(main(["generate", *args]), 0)
            self.assertEqual(main(["check", *args]), 0)
            target = Path(td) / "python/client.py"; target.write_text(target.read_text() + "# drift\n")
            self.assertEqual(main(["check", *args]), 1)
            self.assertEqual(main(["generate", *args]), 0)
            (Path(td) / "unowned.txt").write_text("keep")
            self.assertEqual(main(["generate", *args]), 2)
            self.assertEqual((Path(td) / "unowned.txt").read_text(), "keep")

    def test_nonempty_unowned_output_refused(self):
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "openapi.json").write_text("do not clobber")
            self.assertEqual(main(["generate", "--root", str(ROOT), "--fixture", "--out", td]), 2)
            self.assertEqual((Path(td) / "openapi.json").read_text(), "do not clobber")

    def test_subprocess_hash_seed_reproducibility(self):
        with tempfile.TemporaryDirectory() as td:
            roots = [Path(td) / "a", Path(td) / "b"]
            for seed, out in zip(("1", "999"), roots):
                env = {**os.environ, "PYTHONHASHSEED": seed, "TZ": "Pacific/Honolulu" if seed == "1" else "UTC"}
                subprocess.run([sys.executable, "-m", "tools.sdkgen", "generate", "--root", str(ROOT), "--fixture", "--out", str(out)], cwd=ROOT, env=env, check=True, capture_output=True)
            a = {p.relative_to(roots[0]).as_posix(): p.read_bytes() for p in roots[0].rglob("*") if p.is_file()}
            b = {p.relative_to(roots[1]).as_posix(): p.read_bytes() for p in roots[1].rglob("*") if p.is_file()}
            self.assertEqual(a, b)


class HandlerContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        out = Path(cls.temp.name)
        for file, content in render(compiler().compile()).items():
            path = out / file; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        cls.generated, cls.runtime = load_generated(out)
        real = ROOT / "headroom/proxy/server.py"
        cls.source_path = real if real.exists() else ROOT / "tests/sdkgen/fixtures/retrieval_routes.py"
        cls.tree = ast.parse(cls.source_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def handler(self, name, available=True, retrieval=True, tool_name=None):
        original = next(n for n in ast.walk(self.tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
        node = copy.deepcopy(original); node.decorator_list = []
        entry = types.SimpleNamespace(original_content='{"snake_case":"世界"}', original_tokens=100,
                                      original_item_count=10, compressed_item_count=2,
                                      tool_name=tool_name, retrieval_count=1)
        store = types.SimpleNamespace(get_entry_status=lambda *a, **k: {"status": "available" if available else "expired"},
                                      retrieve=lambda *a: entry if retrieval else None)
        class HTTPException(Exception):
            def __init__(self, status_code, detail): self.status_code = status_code; self.detail = detail
        namespace = {"Request": object, "HTTPException": HTTPException,
                     "get_compression_store": lambda: store, "format_retrieval_miss_detail": lambda _: "Entry missing"}
        module = ast.Module(body=[node], type_ignores=[]); ast.fix_missing_locations(module)
        exec(compile(module, str(self.source_path), "exec"), namespace)
        return namespace[name], HTTPException

    def request(self, value):
        async def body(): return value
        return types.SimpleNamespace(json=body)

    def test_successful_actual_handler_body_matches_contract(self):
        for name in ("ccr_retrieve", "ccr_retrieve_get"):
            for tool in (None, "", "search_tool"):
                with self.subTest(name=name, tool=tool):
                    fn, _ = self.handler(name, tool_name=tool)
                    arg = self.request({"hash": "abc"}) if name == "ccr_retrieve" else "abc"
                    result = asyncio.run(fn(arg))
                    self.runtime.validate(result, self.runtime.SCHEMAS["RetrieveResponse"])
                    self.assertEqual(result["tool_name"], tool)
                    self.assertEqual(result["original_content"], '{"snake_case":"世界"}')

    def test_missing_hash_error_matches_declared_model(self):
        fn, error = self.handler("ccr_retrieve")
        with self.assertRaises(error) as cm:
            asyncio.run(fn(self.request({})))
        self.assertEqual(cm.exception.status_code, 400)
        self.runtime.validate({"detail": cm.exception.detail}, self.runtime.SCHEMAS["RetrievalError"])

    def test_expired_and_race_misses_match_declared_model(self):
        for name in ("ccr_retrieve", "ccr_retrieve_get"):
            for available, retrieval in ((False, False), (True, False)):
                fn, error = self.handler(name, available, retrieval)
                arg = self.request({"hash": "abc"}) if name == "ccr_retrieve" else "abc"
                with self.assertRaises(error) as cm:
                    asyncio.run(fn(arg))
                self.assertEqual(cm.exception.status_code, 404)
                self.runtime.validate({"detail": cm.exception.detail}, self.runtime.SCHEMAS["RetrievalError"])

    def test_missing_required_nullable_not_silently_accepted(self):
        fn, _ = self.handler("ccr_retrieve_get")
        result = asyncio.run(fn("abc")); del result["tool_name"]
        with self.assertRaises(self.runtime.ProtocolError):
            self.runtime.validate(result, self.runtime.SCHEMAS["RetrieveResponse"])

    def test_wrong_types_and_bool_as_integer_rejected(self):
        fn, _ = self.handler("ccr_retrieve_get")
        result = asyncio.run(fn("abc"))
        for bad in ("100", True, None, float("nan")):
            value = {**result, "original_tokens": bad}
            with self.assertRaises(self.runtime.ProtocolError):
                self.runtime.validate(value, self.runtime.SCHEMAS["RetrieveResponse"])

    def test_python_preserves_arbitrary_integer_and_extra_keys(self):
        fn, _ = self.handler("ccr_retrieve_get")
        result = asyncio.run(fn("abc")); result["original_tokens"] = 9007199254740993
        result["future_extension"] = {"snake_case": "unchanged"}
        self.runtime.validate(result, self.runtime.SCHEMAS["RetrieveResponse"])
        self.assertEqual(result["original_tokens"], 9007199254740993)
        self.assertEqual(result["future_extension"]["snake_case"], "unchanged")


if __name__ == "__main__":
    unittest.main()
