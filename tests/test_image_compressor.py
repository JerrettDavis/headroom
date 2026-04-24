from __future__ import annotations

import base64
import sys
import types

import headroom.image.compressor as image_compressor
from headroom.image.compressor import (
    CompressionResult,
    ImageCompressor,
    Technique,
    compress_images,
    get_compressor,
)


def _data_url(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode()}"


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


def test_image_detection_query_extraction_and_singleton_helpers(monkeypatch) -> None:
    compressor = ImageCompressor()

    messages = [
        {"role": "system", "content": "ignore"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                " this",
                {"type": "image_url", "image_url": {"url": _data_url(b"img")}},
            ],
        },
    ]
    assert compressor.has_images(messages) is True
    assert compressor.has_images([{"role": "user", "content": "plain text"}]) is False
    assert compressor._extract_query(messages) == "describe  this"
    assert compressor._extract_query([{"role": "assistant", "content": "no user"}]) == ""

    assert compressor._extract_image_data(messages) == b"img"
    assert (
        compressor._extract_image_data(
            [
                {
                    "content": [
                        {"type": "image", "source": {"type": "base64", "data": _b64(b"anthropic")}}
                    ]
                }
            ]
        )
        == b"anthropic"
    )
    assert (
        compressor._extract_image_data([{"content": [{"inlineData": {"data": _b64(b"google")}}]}])
        == b"google"
    )
    assert compressor._extract_image_data([{"content": "plain"}]) is None

    compressor.last_result = CompressionResult(
        technique=Technique.FULL_LOW,
        original_tokens=200,
        compressed_tokens=50,
        confidence=0.9,
    )
    assert compressor.last_savings == 75.0
    assert (
        CompressionResult(
            technique=Technique.PRESERVE,
            original_tokens=0,
            compressed_tokens=0,
            confidence=1.0,
        ).savings_percent
        == 0.0
    )

    monkeypatch.setattr(image_compressor, "_default_compressor", None)
    singleton = get_compressor()
    assert singleton is get_compressor()
    monkeypatch.setattr(
        image_compressor,
        "_default_compressor",
        types.SimpleNamespace(compress=lambda messages, provider: [provider]),
    )
    assert compress_images(messages, "google") == ["google"]


def test_count_result_tokens_covers_ocr_and_provider_specific_paths(monkeypatch) -> None:
    compressor = ImageCompressor()
    token_values = {b"openai": 12, b"anthropic": 15, b"google": 18, b"original": 21}

    monkeypatch.setattr(
        compressor,
        "_estimate_tokens",
        lambda data, detail="high": token_values[data],
    )

    messages = [
        {
            "content": [
                {"type": "text", "text": "[OCR from image]\nhello world"},
                {"type": "image_url", "image_url": {"url": _data_url(b"openai"), "detail": "high"}},
                {"type": "image_url", "image_url": {"url": _data_url(b"skip"), "detail": "low"}},
                {"type": "image", "source": {"type": "base64", "data": _b64(b"anthropic")}},
                {"inlineData": {"data": _b64(b"google")}},
            ]
        }
    ]

    total = compressor._count_result_tokens(messages, b"original", "openai")
    assert total == max(1, len("[OCR from image]\nhello world") // 4) + 12 + 85 + 15 + 18

    assert (
        compressor._count_result_tokens([{"content": ["no dict blocks"]}], b"original", "openai")
        == 21
    )


def test_apply_compression_handles_preserve_transcode_and_resize_paths(monkeypatch) -> None:
    compressor = ImageCompressor()
    openai_item = {"type": "image_url", "image_url": {"url": _data_url(b"openai")}}
    anthropic_item = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": _b64(b"anthropic")},
    }
    google_item = {"inlineData": {"mimeType": "image/png", "data": _b64(b"google")}}

    messages = [{"role": "user", "content": [openai_item, anthropic_item, google_item, "keep-me"]}]
    assert compressor._apply_compression(messages, Technique.PRESERVE, "openai") == messages

    monkeypatch.setattr(compressor, "_ocr_extract", lambda image_data: "decoded text")
    transcoded = compressor._apply_compression(
        [{"content": [openai_item]}], Technique.TRANSCODE, "openai"
    )
    assert transcoded[0]["content"] == [{"type": "text", "text": "[OCR from image]\ndecoded text"}]

    monkeypatch.setattr(compressor, "_ocr_extract", lambda image_data: None)
    openai_low = compressor._apply_compression(
        [{"content": [openai_item]}], Technique.TRANSCODE, "openai"
    )
    assert openai_low[0]["content"][0]["image_url"]["detail"] == "low"

    monkeypatch.setattr(
        compressor, "_resize_image", lambda image_data, max_dimension=512: (b"small", "image/jpeg")
    )
    anthropic_low = compressor._apply_compression(
        [{"content": [anthropic_item]}], Technique.FULL_LOW, "anthropic"
    )
    assert anthropic_low[0]["content"][0]["source"]["media_type"] == "image/jpeg"
    assert base64.b64decode(anthropic_low[0]["content"][0]["source"]["data"]) == b"small"

    google_low = compressor._apply_compression(
        [{"content": [google_item]}], Technique.CROP, "google"
    )
    assert google_low[0]["content"][0]["inlineData"]["mimeType"] == "image/jpeg"
    assert base64.b64decode(google_low[0]["content"][0]["inlineData"]["data"]) == b"small"


def test_ocr_extract_handles_success_low_confidence_and_errors(monkeypatch) -> None:
    compressor = ImageCompressor()

    class _RapidOCR:
        def __init__(self, result):
            self.result = result

        def __call__(self, image_data: bytes):
            return self.result, None

    monkeypatch.setitem(
        sys.modules,
        "rapidocr_onnxruntime",
        types.SimpleNamespace(
            RapidOCR=lambda: _RapidOCR(
                [
                    (None, "first", 0.95),
                    (None, "second", 0.85),
                ]
            )
        ),
    )
    assert compressor._ocr_extract(b"img") == "first\nsecond"

    compressor = ImageCompressor()
    monkeypatch.setitem(
        sys.modules,
        "rapidocr_onnxruntime",
        types.SimpleNamespace(RapidOCR=lambda: _RapidOCR([(None, "low", 0.2)])),
    )
    assert compressor._ocr_extract(b"img", min_confidence=0.7) is None

    compressor = ImageCompressor()
    monkeypatch.setitem(
        sys.modules,
        "rapidocr_onnxruntime",
        types.SimpleNamespace(RapidOCR=lambda: _RapidOCR([])),
    )
    assert compressor._ocr_extract(b"img") is None

    compressor = ImageCompressor()

    class _ExplodingOCR:
        def __call__(self, image_data: bytes):
            raise RuntimeError("boom")

    monkeypatch.setitem(
        sys.modules,
        "rapidocr_onnxruntime",
        types.SimpleNamespace(RapidOCR=lambda: _ExplodingOCR()),
    )
    assert compressor._ocr_extract(b"img") is None


def test_compress_orchestrates_tile_savings_router_paths_and_fallbacks(monkeypatch) -> None:
    compressor = ImageCompressor()
    image_messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _data_url(b"img")}}],
        }
    ]

    assert compressor.compress([{"role": "user", "content": "plain"}], "openai") == [
        {"role": "user", "content": "plain"}
    ]

    monkeypatch.setitem(
        sys.modules,
        "headroom.image.tile_optimizer",
        types.SimpleNamespace(
            optimize_images_in_messages=lambda messages, provider: (
                messages,
                [types.SimpleNamespace(tokens_saved=7)],
            )
        ),
    )
    monkeypatch.setattr(compressor, "_extract_query", lambda messages: "")
    monkeypatch.setattr(compressor, "_extract_image_data", lambda messages: b"img")
    assert compressor.compress(image_messages, "openai") == image_messages
    assert compressor.last_result.technique == Technique.PRESERVE
    assert compressor.last_result.original_tokens == 7
    assert compressor.last_result.compressed_tokens == 0

    compressor = ImageCompressor()
    monkeypatch.setitem(
        sys.modules,
        "headroom.image.tile_optimizer",
        types.SimpleNamespace(
            optimize_images_in_messages=lambda messages, provider: (
                messages,
                [types.SimpleNamespace(tokens_saved=10)],
            )
        ),
    )
    monkeypatch.setattr(compressor, "_extract_query", lambda messages: "what is shown?")
    monkeypatch.setattr(compressor, "_extract_image_data", lambda messages: b"img")
    monkeypatch.setattr(compressor, "_estimate_tokens", lambda image_data, detail="high": 200)
    monkeypatch.setattr(
        compressor,
        "_apply_compression",
        lambda messages, technique, provider: [{"role": "assistant", "content": technique.value}],
    )
    monkeypatch.setattr(
        compressor, "_count_result_tokens", lambda messages, original_image_data, provider: 50
    )

    class _OnnxRouter:
        def __init__(self, use_siglip: bool = True) -> None:
            self.use_siglip = use_siglip

        def classify(self, image_data: bytes, query: str):
            return types.SimpleNamespace(technique=Technique.FULL_LOW, confidence=0.8)

    monkeypatch.setitem(
        sys.modules,
        "headroom.image.onnx_router",
        types.SimpleNamespace(OnnxTechniqueRouter=_OnnxRouter),
    )
    routed = compressor.compress(image_messages, "openai")
    assert routed == [{"role": "assistant", "content": "full_low"}]
    assert compressor.last_result.technique == Technique.FULL_LOW
    assert compressor.last_result.original_tokens == 210
    assert compressor.last_result.compressed_tokens == 50
    assert compressor.last_result.confidence == 0.8

    compressor = ImageCompressor()
    monkeypatch.setitem(
        sys.modules,
        "headroom.image.tile_optimizer",
        types.SimpleNamespace(
            optimize_images_in_messages=lambda messages, provider: (messages, [])
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.image.onnx_router",
        types.SimpleNamespace(
            OnnxTechniqueRouter=type(
                "BrokenOnnxRouter",
                (),
                {
                    "__init__": lambda self, use_siglip=True: None,
                    "classify": lambda self, image_data, query: (_ for _ in ()).throw(
                        RuntimeError("onnx fail")
                    ),
                },
            )
        ),
    )
    monkeypatch.setattr(compressor, "_extract_query", lambda messages: "fallback")
    monkeypatch.setattr(compressor, "_extract_image_data", lambda messages: b"img")
    monkeypatch.setattr(
        compressor, "_get_router", lambda: (_ for _ in ()).throw(RuntimeError("pt fail"))
    )
    monkeypatch.setattr(compressor, "_estimate_tokens", lambda image_data, detail="high": 30)
    monkeypatch.setattr(
        compressor, "_count_result_tokens", lambda messages, original_image_data, provider: 30
    )

    preserved = compressor.compress(image_messages, "openai")
    assert preserved == image_messages
    assert compressor.last_result.technique == Technique.PRESERVE
    assert compressor.last_result.confidence == 0.0


def test_router_resize_and_token_estimation_paths(monkeypatch) -> None:
    compressor = ImageCompressor(model_id="router-id", use_siglip=False, device="cpu")
    created: list[tuple[str | None, bool, str | None]] = []

    class _Router:
        def __init__(self, model_path=None, use_siglip=True, device=None) -> None:
            created.append((model_path, use_siglip, device))

    monkeypatch.setitem(
        sys.modules,
        "headroom.image.trained_router",
        types.SimpleNamespace(Technique=Technique, TrainedRouter=_Router),
    )
    assert compressor._get_router().__class__ is _Router
    assert compressor._get_router().__class__ is _Router
    assert created == [("router-id", False, "cpu")]

    class _FakeImage:
        def __init__(self, size: tuple[int, int], *, fmt: str = "PNG", mode: str = "RGBA") -> None:
            self.size = size
            self.format = fmt
            self.mode = mode
            self.saved = False

        def resize(self, size: tuple[int, int], resample) -> _FakeImage:
            return _FakeImage(size, fmt=self.format, mode=self.mode)

        def convert(self, mode: str) -> _FakeImage:
            return _FakeImage(self.size, fmt=self.format, mode=mode)

        def save(self, buf, format: str, quality: int, optimize: bool) -> None:
            buf.write(f"{format}:{quality}:{optimize}:{self.size}".encode())

    pil_module = types.SimpleNamespace(
        Image=types.SimpleNamespace(
            open=lambda stream: _FakeImage((1024, 256)),
            Resampling=types.SimpleNamespace(LANCZOS="lanczos"),
        )
    )
    monkeypatch.setitem(sys.modules, "PIL", pil_module)
    resized, media_type = compressor._resize_image(b"img", max_dimension=512, quality=77)
    assert media_type == "image/jpeg"
    assert b"JPEG:77:True:(512, 128)" in resized

    pil_module.Image.open = lambda stream: _FakeImage((100, 100), fmt="PNG", mode="RGB")
    original, media_type = compressor._resize_image(b"small", max_dimension=512)
    assert original == b"small"
    assert media_type == "image/png"

    pil_module.Image.open = lambda stream: _FakeImage((1025, 513), fmt="PNG", mode="RGB")
    assert compressor._estimate_tokens(b"img", "high") == 680
    assert compressor._estimate_tokens(b"img", "low") == 85

    def raise_open(stream):
        raise RuntimeError("bad image")

    pil_module.Image.open = raise_open
    assert compressor._estimate_tokens(b"img", "high") == 765


def test_apply_compression_and_compress_cover_remaining_fallback_paths(monkeypatch) -> None:
    compressor = ImageCompressor()
    weird_technique = types.SimpleNamespace(value="weird")

    openai_remote = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    anthropic_bad = {"type": "image", "source": {"type": "url", "data": "ignored"}}
    google_item = {"inlineData": {"mimeType": "image/png", "data": _b64(b"google")}}

    result = compressor._apply_compression(
        [
            {"content": "plain"},
            {
                "content": [
                    "keep",
                    {"type": "text", "text": "no image"},
                    openai_remote,
                    anthropic_bad,
                    google_item,
                ]
            },
        ],
        weird_technique,
        "openai",
    )
    assert result[0]["content"] == "plain"
    assert result[1]["content"][0] == "keep"
    assert result[1]["content"][1] == {"type": "text", "text": "no image"}
    assert result[1]["content"][2] == openai_remote
    assert result[1]["content"][3] == anthropic_bad
    assert result[1]["content"][4] == google_item

    monkeypatch.setattr(
        compressor,
        "_resize_image",
        lambda image_data, max_dimension=512: (_ for _ in ()).throw(RuntimeError("resize failed")),
    )
    failed_google = compressor._apply_compression(
        [{"content": [google_item]}], Technique.CROP, "google"
    )
    assert failed_google[0]["content"][0] == google_item

    failed_anthropic = compressor._apply_compression(
        [
            {
                "content": [
                    {"type": "image", "source": {"type": "base64", "data": _b64(b"anthropic")}}
                ]
            }
        ],
        Technique.FULL_LOW,
        "anthropic",
    )
    assert failed_anthropic[0]["content"][0]["type"] == "image"

    compressor = ImageCompressor()
    image_messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": _data_url(b"img")}}],
        }
    ]
    monkeypatch.setitem(
        sys.modules,
        "headroom.image.tile_optimizer",
        types.SimpleNamespace(
            optimize_images_in_messages=lambda messages, provider: (_ for _ in ()).throw(
                RuntimeError("tile fail")
            )
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "headroom.image.onnx_router",
        types.SimpleNamespace(
            OnnxTechniqueRouter=type(
                "BrokenOnnxRouter",
                (),
                {
                    "__init__": lambda self, use_siglip=True: None,
                    "classify": lambda self, image_data, query: (_ for _ in ()).throw(
                        RuntimeError("onnx fail")
                    ),
                },
            )
        ),
    )
    monkeypatch.setattr(compressor, "_extract_query", lambda messages: "fallback")
    monkeypatch.setattr(compressor, "_extract_image_data", lambda messages: b"img")
    monkeypatch.setattr(
        compressor,
        "_get_router",
        lambda: types.SimpleNamespace(
            classify=lambda image_data, query: types.SimpleNamespace(
                technique=Technique.CROP,
                confidence=0.6,
            )
        ),
    )
    monkeypatch.setattr(compressor, "_estimate_tokens", lambda image_data, detail="high": 50)
    monkeypatch.setattr(
        compressor,
        "_apply_compression",
        lambda messages, technique, provider: [{"role": "assistant", "content": technique.value}],
    )
    monkeypatch.setattr(
        compressor, "_count_result_tokens", lambda messages, original_image_data, provider: 15
    )
    routed = compressor.compress(image_messages, "google")
    assert routed == [{"role": "assistant", "content": "crop"}]
    assert compressor.last_result.technique == Technique.CROP
    assert compressor.last_result.original_tokens == 50
    assert compressor.last_result.compressed_tokens == 15
