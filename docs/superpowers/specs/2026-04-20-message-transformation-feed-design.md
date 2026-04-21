# Spec: Message Transformation Live Feed

**Date:** 2026-04-20
**Feature:** Live-updating feed of message transformations in the dashboard

---

## 1. Overview

A right-side sidebar drawer that displays a live, streaming feed of message transformations. Each entry shows a before/after diff of messages as they pass through Headroom's compression pipeline. New messages arrive in real-time, push into the top of the feed, and old messages are pruned to maintain performance.

---

## 2. Layout

**Location:** Right sidebar drawer, toggled via a button in the dashboard header.

**Trigger:** A "Live Feed" button/icon in the header opens the drawer. The drawer slides in from the right edge and overlays the main content (does not reflow the page layout).

**States:**
- Closed (default)
- Open (drawer visible, sliding in from right)

---

## 3. Diff View Format

**Side-by-side split:**
- Left panel: original message (before compression)
- Right panel: compressed message (after compression)
- Panels scroll together (synchronized scroll)
- Content is monospace font
- Overflow text wraps or truncates with ellipsis

---

## 4. Streaming Behavior

**Auto-stream:**
- New messages slide in at the top of the feed
- Feed auto-scrolls to show the newest item

**Fixed-view detection:**
- If the user scrolls down to review older messages, the auto-scroll pauses
- A subtle indicator shows "N new messages" when new items arrive while paused
- Clicking the indicator resumes auto-scroll at the top

**Polling interval:** Every 3–5 seconds (configurable), fetching batches of messages from the backend

---

## 5. Performance Strategy

**Virtual scrolling:**
- Render only visible rows + a buffer above and below the viewport
- Handles large numbers of messages without DOM bloat
- Supports thousands of entries with consistent scroll performance

**Batch size:** Cap at ~50–100 messages in the feed to bound memory usage

---

## 6. Data Flow

1. Frontend polls a `/stats` or dedicated `/transformations/feed` endpoint every N seconds
2. Backend returns a batch of recent transformations (most recent first)
3. Frontend merges new messages into the feed array, capped at max size
4. Virtual scroll renders only visible rows

---

## 7. Backend API (TBD)

Need to determine if there's an existing endpoint that streams transformation data, or if a new one is needed:

- Option A: Reuse `/stats` (already polled)
- Option B: New dedicated endpoint `/transformations/feed` returning just the message diffs

---

## 8. Out of Scope

- Storing/archiving full transformation history
- Exporting feed data
- Filtering or searching the feed
- Click-to-expand individual messages beyond the diff view
