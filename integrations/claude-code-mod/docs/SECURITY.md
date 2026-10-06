# Security and privacy boundaries

The UI never changes prompts, tool results, compression settings, cache policy,
logging preferences, provider credentials, or Headroom statistics. The launcher's
sole attribution addition is `X-Headroom-Mod-Session`, not the cache-identity header.
It sets the explicitly selected local client base URL and refuses a conflicting
inherited upstream. Existing credential-bearing custom headers remain only in the
child environment and normal provider request path, not in sidebar state or API
responses. The companion does not expose arbitrary log tags or raw errors.

All companion routes reuse Headroom's actual loopback peer/Host guard and same-origin
guard. Responses are `Cache-Control: no-store`. There is no wildcard CORS change,
proxy-token bypass, separate admin mutation endpoint, secret retrieval endpoint,
relay, or unauthenticated remote deployment. The client accepts literal loopback
HTTP origins only and sends GETs without an auth handle or body. The launcher
disables environment proxies and redirects for its identity probe. The mod uses
the Claude host HTTP API, which does not expose equivalent redirect/size/abort
controls; endpoint/response checks do not turn a malicious local process into a
trusted one. The companion and the local runtime must themselves be trusted.

Session IDs are routing/attribution selectors, **not credentials**. Other processes
on the same trusted host are within Headroom's existing local trust boundary.
This implementation is not multi-user tenant isolation. Do not put the endpoints
behind a public reverse proxy and assume UUIDs provide authorization.

Metadata polling never copies message bodies. Inspection filters ownership first,
then serializes only the selected view under node, depth, string and output limits.
Text is plain, with control and bidi characters replaced. It is not rendered as
Markdown or HTML, linked to executable actions, or submitted to a model. Content
is discarded from mod state on close, tab departure, mismatched session, restart,
or session end. Capture remains Headroom's explicit `--log-messages` opt-in; its
existing logs may still contain sensitive data after this UI closes/uninstalls.

The private `_logs` adapter is a compatibility risk, not a stable Headroom API.
It verifies a bounded deque of dataclass records with tags, fails closed on drift,
and has a real-package contract gate. Production rollout also needs the full
Headroom suite/security review. Offline tests use explicit guard doubles to check
routing; they are not a substitute for executing the real guards in Headroom.
