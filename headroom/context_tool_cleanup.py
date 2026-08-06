"""Undo the retired CLI context tools on machines that already have them.

Headroom used to download ``rtk`` (Rust Token Killer) and ``lean-ctx``, register
their Claude Code ``PreToolUse`` hooks, symlink the managed binaries onto
``PATH``, and inject a marker-fenced instruction block ("always prefix shell
commands with ``rtk``" / "prefer ``ctx_read`` over ``Read``") into a dozen agent
hint files. Both integrations have been removed from Headroom.

Deleting the code is not enough: everything above is *durable state on the
user's disk*. Left alone, the Claude hooks keep rewriting every Bash command
through binaries Headroom no longer manages, and the injected guidance keeps
telling agents to use tools that may not resolve. So ``headroom wrap`` /
``headroom unwrap`` call :func:`purge_context_tool_artifacts` once per run to
remove what earlier versions installed.

Everything here is idempotent, best-effort and deliberately conservative:

* only files Headroom installed (or caused a context tool to install) are
  deleted — an MCP entry's ``command`` or a hook script's body counts as
  Headroom's only when it names a path inside :func:`paths.bin_dir`,
  compared on normalized forms (:func:`_references_managed_bin`): the raw
  text is scanned for the managed directory (never pre-split on whitespace,
  so a path containing a space — e.g. a ``$HOME`` with one — cannot be
  severed), and a hit counts only at a real path boundary — end-of-string, a
  non-path character (quote, ``=``, ``:``, ``;``, ``,``, ``(``, ``)``, ``|``,
  or whitespace), or a ``/`` whose lexically-normalized (``.``/``..``
  collapsed) continuation still starts with the managed directory. A sibling
  like ``bin-backup`` and a ``bin/../evil`` traversal never match;
* a hook entry (:func:`_references_context_tool`) passes a marker prefilter
  first, then counts as Headroom's either because its own ``command`` names
  the managed directory directly, or — the common case, since a hook command
  usually names a script rather than the binary — because its ``command`` is,
  by full path (not basename — a same-named script of the user's own
  elsewhere must never inherit another script's verdict), one of the
  ``_HOOK_SCRIPTS`` files under ``~/.claude/hooks`` the classification map
  already marked ``True``: it inherits that script's verdict rather than
  being checked again. The map is built from ``~/.claude/hooks`` only, so a
  Cursor ``hooks.json`` entry naming a script under ``~/.cursor`` can never
  inherit a verdict this way — it is still caught if its ``command`` names
  the managed bin directory directly;
* ``.rtk-hook.sha256`` is never read for its own provenance (it holds a hex
  digest, not a path) and instead inherits ``rtk-rewrite.sh``'s
  classification; a ``<name>.lean-ctx.bak`` backup inherits ``<name>``'s
  (:func:`_classify_hook_scripts`);
* a hook script that exists but cannot be read is classified unknown —
  deleted by nothing, and named in the report so the user can remove it by
  hand;
* ``~/.local/bin/{rtk,lean-ctx}`` is unlinked only when it is a symlink into
  Headroom's own bin directory — a user's own build is never touched;
* a JSON config that does not parse is reported and **skipped**, never
  overwritten (a hand-edited typo must not cost the user their settings);
* the tools' own backups of *config* files (``~/.claude.json.lean-ctx.bak`` and
  friends) are left in place — they hold the user's real settings history. Only
  backups of the hook scripts proven to be Headroom's are cleaned up;
* two cases cannot be decided at all, and are accepted as limits rather than
  fixed:

  * ``get_lean_ctx_path`` used to check ``PATH`` before Headroom's own bin
    directory, so on a machine that already had ``lean-ctx`` on ``PATH``,
    the tool that ran was the user's own, and the config it wrote looks
    exactly like config the user wrote by hand. That leftover survives the
    purge — it still points at a binary that exists, so nothing dangles;
  * since #1698, ``rtk init``'s hook is deliberately left byte-for-byte as
    written — a bare ``rtk``, resolved through the ``~/.local/bin/rtk``
    symlink — so it never mentions :func:`paths.bin_dir` and cannot be told
    apart from a hook a user wrote by hand running their own ``rtk``.
    ``rtk-rewrite.sh``, its ``.rtk-hook.sha256`` and its ``settings.json``
    entry therefore now survive on *every* machine that ever ran Headroom's
    ``rtk init``, canonical or not — while step 3 still removes the
    ``~/.local/bin/rtk`` symlink and the managed binary it pointed at. The
    surviving hook execs a ``rtk`` that no longer resolves and silently
    no-ops (#487, #1698). Inferring provenance from the now-removed symlink
    was considered and rejected: that would still be a guess, and the
    guard's rule is that unprovable means keep, not "keep unless a further
    guess says otherwise";
* the marker-fenced guidance block is the one step with no provenance check
  to make — ``<!-- headroom:rtk-instructions -->`` is Headroom's own fence,
  and no third party writes it.
"""

from __future__ import annotations

import json
import os
import posixpath
from pathlib import Path
from typing import Any

from headroom import fsutil, paths

# Substrings that identify a hook command as one of the retired tools'.
# ``rtk init --auto-patch`` writes ``~/.claude/hooks/rtk-rewrite.sh`` and shells
# out to ``rtk rewrite``; ``lean-ctx init --agent`` writes ``lean-ctx-rewrite.sh``
# / ``lean-ctx-redirect.sh`` plus ``-native`` variants that exec ``lean-ctx hook
# rewrite``. All are specific enough that a user-authored hook cannot match by
# accident.
_HOOK_COMMAND_MARKERS = (
    "rtk-rewrite",
    "rtk rewrite",
    "lean-ctx-rewrite",
    "lean-ctx-redirect",
    "lean-ctx hook",
)

# Fence Headroom wrapped its injected guidance in. Both tools shared it — the
# lean-ctx block was written inside the same ``rtk-instructions`` markers.
_FENCE_START = "<!-- headroom:rtk-instructions -->"
_FENCE_END = "<!-- /headroom:rtk-instructions -->"

# Managed binary names (Windows ships the .exe).
_BINARY_NAMES = ("rtk", "rtk.exe", "lean-ctx", "lean-ctx.exe")

# Hook scripts the tools generated, relative to ``~/.claude/hooks``. Each is
# also cleaned up in its ``.lean-ctx.bak`` form: lean-ctx copies the previous
# script aside on every re-init, so the backups pile up alongside the originals.
_HOOK_SCRIPTS = (
    "rtk-rewrite.sh",
    ".rtk-hook.sha256",
    "lean-ctx-rewrite.sh",
    "lean-ctx-redirect.sh",
    "lean-ctx-rewrite-native",
    "lean-ctx-redirect-native",
)

# rtk's integrity digest never names a path (see ``_classify_hook_scripts``)
# and rtk-rewrite.sh is the script it authenticates.
_RTK_DIGEST_NAME = ".rtk-hook.sha256"
_RTK_SCRIPT_NAME = "rtk-rewrite.sh"

# MCP server entries the tools registered, and the config files holding them.
# lean-ctx registers itself as an MCP server during ``lean-ctx init``; rtk never
# did, but it is matched too so a stale hand-added entry is cleaned up as well.
_MCP_SERVER_NAMES = ("lean-ctx", "lean_ctx", "rtk")


def purge_context_tool_artifacts() -> list[str]:
    """Remove every rtk / lean-ctx artifact an earlier Headroom version installed.

    Returns human-readable descriptions of what was removed — plus a line for
    any config that had to be skipped because the user must fix it by hand. An
    empty list means there was nothing to do, which is the steady state after
    the first run.
    """
    home = Path.home()
    project = Path.cwd()
    report: list[str] = []

    # Classify every hook script's provenance once, up front: both step 1 (is
    # a settings.json entry pointing at *our* script?) and step 2 (is the
    # script itself ours?) need the same answer, and each file is read once.
    hooks_dir = home / ".claude" / "hooks"
    hook_classification = _classify_hook_scripts(hooks_dir)

    # 1. Hook registrations (Claude Code's settings.json, Cursor's hooks.json).
    for config in (home / ".claude" / "settings.json", home / ".cursor" / "hooks.json"):
        report += _purge_hook_config(config, hooks_dir, hook_classification)

    # 2. The generated hook scripts, their integrity digests and stale backups
    # — only the ones proven to reference Headroom's managed bin directory.
    managed_names = [name for name in _HOOK_SCRIPTS if hook_classification.get(name)]
    report += _remove_files(
        *(hooks_dir / name for name in managed_names),
        *(hooks_dir / f"{name}.lean-ctx.bak" for name in managed_names),
    )
    for name in _HOOK_SCRIPTS:
        if hook_classification.get(name, False) is not None:
            continue
        if name == _RTK_DIGEST_NAME:
            report.append(
                f"skipped {hooks_dir / name} (inherits {_RTK_SCRIPT_NAME}'s unreadable verdict)"
                " — remove any stale hook script by hand"
            )
        else:
            report.append(
                f"skipped {hooks_dir / name} (could not read to verify it was Headroom's)"
                " — remove any stale hook script by hand"
            )

    # 3. The PATH symlinks, then the managed binaries they pointed at.
    for name in ("rtk", "lean-ctx"):
        report += _remove_managed_path_link(home / ".local" / "bin" / name)
    report += _remove_files(*(paths.bin_dir() / name for name in _BINARY_NAMES))

    # 4. MCP server registrations (lean-ctx registers itself during init).
    report += _purge_mcp_entries(home / ".claude.json", "mcpServers")
    report += _purge_mcp_entries(_opencode_home(home) / "opencode.json", "mcp")

    # 5. Marker-fenced guidance in every hint file the wrap harnesses wrote to.
    for hint_file in _instruction_files(home, project):
        report += _purge_fenced_block(hint_file)
    report += _purge_continue_system_messages(project / ".continue" / "config.json")

    return report


def _instruction_files(home: Path, project: Path) -> list[Path]:
    """Hint files the wrap subcommands injected the context-tool block into.

    De-duplicated: ``CODEX_HOME`` / ``OPENCODE_HOME`` can point at the project
    directory, and a file must not be reported twice.
    """
    candidates = (
        home / ".claude" / "CLAUDE.md",
        _codex_home(home) / "AGENTS.md",
        _opencode_home(home) / "AGENTS.md",
        project / "AGENTS.md",
        project / ".github" / "copilot-instructions.md",
        # Shared by `wrap aider` and `wrap openclaude`.
        project / "CONVENTIONS.md",
        project / ".clinerules",
        project / ".goosehints",
        project / ".cursorrules",
    )
    return list(dict.fromkeys(candidates))


def _codex_home(home: Path) -> Path:
    """Codex's config directory, respecting ``CODEX_HOME``."""
    configured = os.environ.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else home / ".codex"


def _opencode_home(home: Path) -> Path:
    """OpenCode's config directory, respecting ``OPENCODE_HOME``."""
    configured = os.environ.get("OPENCODE_HOME", "").strip()
    return Path(configured).expanduser() if configured else home / ".config" / "opencode"


# --- hook registrations -------------------------------------------------------


# Characters that can never be part of a path: a match ending right before
# one of these (or at end-of-string) sits at a real word boundary. Quotes,
# `=`/`:` (`export PATH="<dir>:$PATH"`, `BIN=<dir>/x`), `;`/`,`/`|` (command
# joiners) and `()` (subshells) all end a path the same way whitespace does.
_PATH_BOUNDARY_CHARS = frozenset("\"'=:;,()|")


def _references_managed_bin(text: str) -> bool:
    """Whether ``text`` names a path inside Headroom's managed bin directory.

    A hook command or script body is free text we don't control, so this
    scans ``text`` for raw occurrences of the managed directory — the
    unresolved and resolved bin directory, and its ``~``-relative form
    (home-relative, since a script may reference it unexpanded) — matched
    against the text as given and as ``expanduser``'d, case-folded, with both
    path separators. Deliberately *not* tokenized on whitespace first: a
    quoted or ``$HOME``-derived path can itself contain a space, and slicing
    the text into words before searching would sever it.

    A hit is only a real reference at a path boundary: immediately followed
    by end-of-string or a :data:`_PATH_BOUNDARY_CHARS` character (an exact
    reference, e.g. a bare ``PATH=<dir>`` export), or by ``/`` — in which
    case the run of characters up to the next boundary is lexically
    normalized (``.``/``..`` collapsed) and re-compared, so neither a sibling
    directory like ``bin-backup``/``binfoo`` nor a ``bin/../evil`` traversal
    can borrow the managed prefix.
    # ponytail: boundary-aware substring scan, not a shell parse — upgrade to
    # shlex if a command ever embeds a managed path it does not execute.
    """
    try:
        bin_dir = paths.bin_dir()
        resolved_bin_dir = bin_dir.resolve()
    except OSError:
        return False

    def _norm(value: str) -> str:
        return os.path.normcase(value).replace("\\", "/")

    needles = {_norm(str(bin_dir)), _norm(str(resolved_bin_dir))}
    home = Path.home()
    for base in (bin_dir, resolved_bin_dir):
        try:
            needles.add(_norm(f"~/{base.relative_to(home).as_posix()}"))
        except ValueError:
            pass

    for haystack in (text, os.path.expanduser(text)):
        normalized_haystack = _norm(haystack)
        if any(_names_managed_dir(normalized_haystack, needle) for needle in needles):
            return True
    return False


def _names_managed_dir(haystack: str, needle: str) -> bool:
    """Whether a normalized ``haystack`` names ``needle`` at a path boundary."""
    search_from = 0
    while True:
        index = haystack.find(needle, search_from)
        if index < 0:
            return False
        end = index + len(needle)
        search_from = index + 1  # keep scanning; occurrences may overlap
        following = haystack[end : end + 1]
        if not following or following.isspace() or following in _PATH_BOUNDARY_CHARS:
            return True
        if following != "/":
            continue
        tail_end = end
        while tail_end < len(haystack) and not (
            haystack[tail_end].isspace() or haystack[tail_end] in _PATH_BOUNDARY_CHARS
        ):
            tail_end += 1
        candidate = posixpath.normpath(haystack[index:tail_end])
        if candidate == needle or candidate.startswith(needle + "/"):
            return True


def _classify_hook_scripts(hooks_dir: Path) -> dict[str, bool | None]:
    """Classify each existing ``_HOOK_SCRIPTS`` file by whether it is Headroom's.

    ``True`` — the file exists and its body references the managed bin
    directory. ``False`` — it exists and does not. ``None`` — it exists but
    could not be read, so its provenance is unprovable. A basename with no
    file on disk is simply absent from the map. Each file is read at most once.

    ``.rtk-hook.sha256`` holds a hex digest that can never reference a path,
    so it is never read; it inherits ``rtk-rewrite.sh``'s classification.
    """
    classification: dict[str, bool | None] = {}
    for name in _HOOK_SCRIPTS:
        if name == _RTK_DIGEST_NAME:
            continue
        path = hooks_dir / name
        if not path.is_file():
            continue
        try:
            body = fsutil.read_text(path)
        except OSError:
            classification[name] = None
            continue
        classification[name] = _references_managed_bin(body)

    if (hooks_dir / _RTK_DIGEST_NAME).is_file():
        classification[_RTK_DIGEST_NAME] = classification.get(_RTK_SCRIPT_NAME, False)

    return classification


def _references_context_tool(
    entry: Any, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> bool:
    """Whether a hook entry is one a retired context tool registered.

    A command-marker hit alone is not enough — a user could author a script
    with a matching name. It must also either name the managed bin directory
    directly, or name — by full path, not basename, so a same-named script
    of the user's own in a different directory is never caught — a hook
    script in ``hooks_dir`` the classification map marks ``True`` (see
    :func:`_classify_hook_scripts`).
    """
    if not isinstance(entry, dict):
        return False
    command = str(entry.get("command", ""))
    if not any(marker in command.lower() for marker in _HOOK_COMMAND_MARKERS):
        return False
    if _references_managed_bin(command):
        return True
    command_path = os.path.normcase(os.path.expanduser(command))
    for name, verdict in hook_classification.items():
        if command_path == os.path.normcase(str(hooks_dir / name)):
            return verdict is True
    return False


def _prune_hooks(
    hooks: Any, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> tuple[Any, bool]:
    """Drop retired-tool entries from a ``hooks`` mapping; return ``(pruned, changed)``.

    Handles both shapes Headroom's installers produced: Claude Code nests
    ``hooks.<Event>[].hooks[].command`` while Cursor's ``hooks.json`` puts the
    command directly on the event entry. Anything unrecognised is passed
    through untouched, and user-authored entries always survive.
    """
    if not isinstance(hooks, dict):
        return hooks, False

    changed = False
    pruned: dict[str, Any] = {}
    for event, entries in hooks.items():
        if not isinstance(entries, list):
            pruned[event] = entries
            continue

        retained: list[Any] = []
        for entry in entries:
            # Cursor shape: the command sits on the entry itself.
            if _references_context_tool(entry, hooks_dir, hook_classification):
                changed = True
                continue
            # Claude shape: a matcher entry holding a list of hooks.
            inner = entry.get("hooks") if isinstance(entry, dict) else None
            if isinstance(inner, list):
                kept_inner = [
                    item
                    for item in inner
                    if not _references_context_tool(item, hooks_dir, hook_classification)
                ]
                if len(kept_inner) != len(inner):
                    changed = True
                    if not kept_inner:
                        # The matcher existed only to hold the retired tool's hook.
                        continue
                    entry = {**entry, "hooks": kept_inner}
            retained.append(entry)

        if not entries:
            pruned[event] = entries
        elif retained:
            pruned[event] = retained
        else:
            # Every entry for this event was ours — drop the empty event.
            changed = True

    return pruned, changed


def _purge_hook_config(
    path: Path, hooks_dir: Path, hook_classification: dict[str, bool | None]
) -> list[str]:
    """Remove retired-tool hook registrations from a JSON hook config."""
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale hook by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale hook by hand"]

    hooks, changed = _prune_hooks(payload.get("hooks"), hooks_dir, hook_classification)
    if not changed:
        return []
    if hooks:
        payload["hooks"] = hooks
    else:
        payload.pop("hooks", None)
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")
    return [f"removed the retired context-tool hook from {path}"]


# --- MCP registrations --------------------------------------------------------


def _purge_mcp_entries(path: Path, container_key: str) -> list[str]:
    """Drop retired-tool MCP server entries from a client config.

    ``lean-ctx init`` registers lean-ctx as an MCP server in the harness's own
    config — Claude Code keeps them under ``mcpServers``, OpenCode under ``mcp``.
    A name match alone is not enough — a user can register their own server
    under the same name — so an entry is removed only when its ``command``
    also names Headroom's managed bin directory. Every other server, and
    every unrelated top-level key, is preserved byte-for-byte.
    """
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale MCP entry by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale MCP entry by hand"]

    servers = payload.get(container_key)
    if not isinstance(servers, dict):
        return []
    removed = [
        name
        for name in _MCP_SERVER_NAMES
        if isinstance(servers.get(name), dict)
        and isinstance(servers[name].get("command"), str)
        and _references_managed_bin(servers[name]["command"])
    ]
    if not removed:
        return []
    for name in removed:
        servers.pop(name, None)
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")
    return [f"removed the {', '.join(removed)} MCP server entry from {path}"]


# --- binaries -----------------------------------------------------------------


def _remove_files(*targets: Path) -> list[str]:
    """Delete Headroom-managed files, reporting each removal."""
    report: list[str] = []
    for target in targets:
        if not target.exists() and not target.is_symlink():
            continue
        try:
            target.unlink()
        except OSError as exc:
            report.append(f"could not remove {target} ({exc})")
            continue
        report.append(f"removed {target}")
    return report


def _remove_managed_path_link(link: Path) -> list[str]:
    """Unlink ``link`` only if it is a symlink into Headroom's bin directory.

    A real file, or a symlink to a user's own build, is left alone — the PATH
    entry is not ours to reclaim unless we created it.
    """
    if not link.is_symlink():
        return []
    try:
        target = link.resolve()
        managed_dir = paths.bin_dir().resolve()
    except OSError:
        return []
    if managed_dir not in target.parents:
        return []
    return _remove_files(link)


# --- marker-fenced guidance ---------------------------------------------------


def _strip_fences(content: str) -> str | None:
    """Remove every fenced context-tool block from ``content``.

    Returns the cleaned text (``""`` when nothing but the block was in the
    file), or ``None`` when no complete fence was present.
    """
    cleaned = content
    found = False
    while True:
        start = cleaned.find(_FENCE_START)
        if start < 0:
            break
        end = cleaned.find(_FENCE_END, start)
        if end < 0:
            break
        end += len(_FENCE_END)
        prefix = cleaned[:start].rstrip()
        suffix = cleaned[end:].lstrip("\n")
        cleaned = "\n\n".join(part for part in (prefix, suffix) if part)
        found = True

    if not found:
        return None
    return cleaned.rstrip() + "\n" if cleaned.strip() else ""


def _purge_fenced_block(path: Path) -> list[str]:
    """Strip the context-tool guidance block from a hint file (deleting an empty file)."""
    if not path.is_file():
        return []
    try:
        content = fsutil.read_text(path)
    except OSError:
        return []
    cleaned = _strip_fences(content)
    if cleaned is None:
        return []
    if cleaned:
        fsutil.write_text(path, cleaned)
        return [f"removed the context-tool guidance block from {path}"]
    return _remove_files(path)


def _purge_continue_system_messages(path: Path) -> list[str]:
    """Strip the context-tool block from every ``systemMessage`` in Continue's config.

    Continue supports a top-level ``systemMessage`` plus a per-model override on
    each ``models[]`` entry, and wrap injected into all of them. A field left
    empty by the strip is removed so the config returns to "not set".
    """
    if not path.is_file():
        return []
    try:
        payload = json.loads(fsutil.read_text(path))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"skipped {path} (unreadable JSON: {exc}) — remove any stale guidance by hand"]
    if not isinstance(payload, dict):
        return [f"skipped {path} (not a JSON object) — remove any stale guidance by hand"]

    containers: list[dict[str, Any]] = [payload]
    models = payload.get("models")
    if isinstance(models, list):
        containers += [model for model in models if isinstance(model, dict)]

    changed = False
    for container in containers:
        message = container.get("systemMessage")
        if not isinstance(message, str):
            continue
        cleaned = _strip_fences(message)
        if cleaned is None:
            continue
        changed = True
        if cleaned.strip():
            container["systemMessage"] = cleaned
        else:
            container.pop("systemMessage", None)

    if not changed:
        return []
    fsutil.write_text(path, json.dumps(payload, indent=2) + "\n")
    return [f"removed the context-tool guidance from {path}"]
