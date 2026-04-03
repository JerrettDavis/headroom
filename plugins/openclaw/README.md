# @headroom-ai/openclaw

Context compression plugin for [OpenClaw](https://github.com/openclaw/openclaw). Compresses tool outputs, code, logs, and structured data — 70-90% token savings with zero LLM calls.

## Install

```bash
pip install "headroom-ai[proxy]"
openclaw plugins install headroom-ai/openclaw
```

## Configure

```json
{
  "plugins": {
    "entries": {
      "headroom": {
        "enabled": true,
        "config": {
          "proxyUrl": "http://127.0.0.1:8787"
        }
      }
    },
    "slots": {
      "contextEngine": "headroom"
    }
  }
}
```

`proxyUrl` is required and must be localhost (`127.0.0.1` or `localhost`). The plugin never starts processes and only connects to the configured local proxy.

## Required Proxy Setup

Run Headroom proxy yourself before launching OpenClaw.

Python install:

```bash
pip install "headroom-ai[proxy]"
headroom proxy --host 127.0.0.1 --port 8787
```

NPM install:

```bash
npm install -g headroom-ai
headroom proxy --host 127.0.0.1 --port 8787
```

## How It Works

Every time OpenClaw assembles context for the model, the plugin compresses tool outputs and large messages:

- **JSON arrays** (tool outputs, search results) — statistical selection keeps anomalies, errors, boundaries
- **Code** — AST-aware compression via tree-sitter
- **Logs** — pattern deduplication, keeps errors and boundaries
- **Text** — ML-based token compression

Compression is lossless via CCR (Compress-Cache-Retrieve): originals are stored and the agent gets a `headroom_retrieve` tool to access full details when needed.

## Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `proxyUrl` | required | URL of an already running Headroom proxy (`http://127.0.0.1:<port>` or `http://localhost:<port>`) |

## Comparison with lossless-claw

| | lossless-claw | headroom |
|---|---|---|
| Compaction method | LLM summarization (DAG) | Content-aware compression (zero LLM) |
| Cost of compaction | Tokens (LLM calls) | Zero |
| Best for | Long conversations | Tool-heavy agents with large outputs |
| Retrieval | `lcm_grep`, `lcm_expand` | `headroom_retrieve` (instant) |

## License

Apache-2.0
