"""Offline adapter contracts and content-free request feature classification."""

from typing import Any


def implemented_features(
    protocol: str, transport: str, native: bool, target: str | None = None
) -> frozenset[str]:
    if transport == "websocket" and (protocol != "openai-responses" or not native):
        return frozenset()
    if protocol == "bedrock-invoke" and transport != "http-json":
        return frozenset()
    if not native:
        return (
            frozenset({"text"})
            if transport == "http-json"
            or (protocol, target, transport) == ("openai-chat", "anthropic-messages", "http-stream")
            else frozenset()
        )
    return frozenset(
        {"text", "tools", "parallel_tools", "inline_images", "structured_output", "signed_state"}
    )


def requested_features(payload: dict[str, Any]) -> frozenset[str]:
    features = {"text"}
    if payload.get("tools") or payload.get("tool_choice") or payload.get("toolConfig"):
        features.add("tools")
    if payload.get("parallel_tool_calls"):
        features.add("parallel_tools")
    text_options = payload.get("text")
    if (
        payload.get("response_format")
        or isinstance(text_options, dict)
        and text_options.get("format")
    ):
        features.add("structured_output")

    def visit(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            kind = value.get("type")
            if (
                kind in {"image", "image_url", "input_image"}
                or "inlineData" in value
                or "fileData" in value
            ):
                features.add("inline_images")
            if (
                kind in {"thinking", "redacted_thinking", "reasoning"}
                or "signature" in value
                or "encrypted_content" in value
            ):
                features.add("signed_state")
            if isinstance(kind, str) and kind.startswith(
                ("web_search", "code_interpreter", "file_search", "computer")
            ):
                features.add("hosted_tools")
            if kind in {"input_audio", "audio", "video", "file", "input_file"}:
                features.add("unsupported_media")
            for item in value.values():
                visit(item)

    visit(payload)
    return frozenset(features)
