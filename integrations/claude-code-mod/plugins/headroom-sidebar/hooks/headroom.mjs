import { UUID, initialState, localUrl, parseResponse, safeText, count, percent } from './core.mjs';
import { renderPane } from './render.mjs';

const REF = { plugin: 'headroom-sidebar', key: 'model' };
const PANE = 'headroom-sidebar';
const PREFIX = '/headroom-mod/v1/sessions/';

const read = async $ => (await $.state.get(REF)).value ?? initialState();
const patch = async ($, ctx, changes) => {
  const s = await read($);
  if (s.owner === ctx.owner) await $.state.set(REF, { ...s, ...changes });
};
const quiet = promise => { void promise.catch(() => undefined); };

async function recoverState($, ctx) {
  if ((await $.state.get(REF)).value != null) return;
  // Native /clear and session switching can discard state without re-registering hooks.
  // Only current command events recover it; stale timers never adopt a new session.
  ctx.timer?.cancel(); ctx.scheduled?.cancel(); ctx.inspectRevision++;
  ctx.owner++;
  await $.state.set(REF, { ...initialState(ctx.owner), open: true,
    notice: 'Conversation changed. Relaunch with headroom-mod run to link this conversation.' });
}

async function checked($, ctx) {
  const s = await read($);
  if (s.owner !== ctx.owner) { ctx.timer?.cancel(); return null; }
  const actual = await $.session.id();
  if (!UUID.test(s.sessionId) || actual !== s.sessionId) {
    ctx.inspectRevision++;
    await patch($, ctx, { connection: 'setup', summary: null, detail: null, selection: null,
      notice: 'Conversation not linked (or /clear/resume changed it). Relaunch with headroom-mod run; no proxy-wide data is substituted.' });
    return null;
  }
  return s;
}

async function getJson($, ctx, url, sid, requestId) {
  if (ctx.network) throw new Error('A previous Headroom read is still pending; no overlapping request was started.');
  // The published HttpInit has no abort/timeout member. Do not invent one.
  // A UI deadline does not release the single-flight lock until the HOST read settles.
  const operation = $.http.fetch(url, { method: 'GET', headers: { Accept: 'application/json' } });
  ctx.network = operation;
  void operation.then(() => { if (ctx.network === operation) ctx.network = null; }, () => { if (ctx.network === operation) ctx.network = null; });
  let deadline;
  const timeout = new Promise((_, reject) => {
    deadline = $.clock.after(4000, () => reject(new Error('Headroom read timed out. A pending host read is not retried concurrently.')));
  });
  try { return parseResponse(await Promise.race([operation, timeout]), sid, requestId); }
  finally { deadline?.cancel(); }
}

async function refresh($, ctx) {
  if (ctx.busy || ctx.network) return;
  ctx.busy = true;
  const startedOwner = ctx.owner;
  try {
    const start = await checked($, ctx);
    if (!start || !start.open) return;
    const usage = await $.session.usage(); // no breakdown: never a token-count/model call
    const data = await getJson($, ctx, `${start.baseUrl}${PREFIX}${start.sessionId}?limit=100`, start.sessionId);
    const current = await checked($, ctx);
    if (!current?.open || current.owner !== start.owner) return;
    const restarted = current.summary && current.summary.epoch !== data.epoch;
    if (restarted) ctx.inspectRevision++;
    await patch($, ctx, { summary: data, context: usage?.context ?? null, connection: 'live',
      updatedAt: await $.clock.now(),
      notice: restarted ? 'Proxy restarted. Counters reflect its new retained window.' : 'Local, read-only · refreshes every 5 seconds while open',
      ...(restarted ? { detail: null, selection: null, tab: 'overview' } : {}) });
  } catch (error) {
    if (ctx.owner === startedOwner) await patch($, ctx, { connection: 'offline', detail: null,
      notice: `${safeText(error?.message, 230)} Last metrics, if shown, are stale.` });
  } finally { ctx.busy = false; }
}

function requestRefresh($, ctx) {
  ctx.scheduled?.cancel();
  ctx.scheduled = $.clock.after(1, () => quiet(refresh($, ctx)));
}

async function open($, ctx, focus = true) {
  await recoverState($, ctx);
  await patch($, ctx, { open: true, detail: null, selection: null, tab: 'overview' });
  const placed = await $.ui.open({ id: PANE, title: 'Headroom', columns: 48, ...(focus ? { focus: true } : {}) });
  await patch($, ctx, { band: placed.isPlaced === false });
  requestRefresh($, ctx);
}
async function close($, ctx) {
  ctx.inspectRevision++;
  await patch($, ctx, { open: false, band: false, detail: null, selection: null });
  await $.ui.close({ id: PANE });
}

async function inspectRequest($, ctx, selection) {
  const rev = ++ctx.inspectRevision;
  const start = await checked($, ctx);
  if (!start?.open) return;
  if (!start.summary?.requests.some(r => r.request_id === selection.requestId && r.inspectable_id) && start.selection?.requestId !== selection.requestId) return;
  await patch($, ctx, { selection, detail: null, tab: 'inspect', textPage: 0 });
  try {
    const q = `?side=${selection.side}&message=${selection.message}&page=${selection.page}`;
    const data = await getJson($, ctx, `${start.baseUrl}${PREFIX}${start.sessionId}/requests/${encodeURIComponent(selection.requestId)}${q}`, start.sessionId, selection.requestId);
    const current = await checked($, ctx);
    if (!current?.open || rev !== ctx.inspectRevision || current.owner !== start.owner) return;
    if (data.epoch !== current.summary?.epoch) throw new Error('Proxy restarted; refresh before inspecting.');
    await patch($, ctx, { detail: data });
  } catch (error) {
    if (rev === ctx.inspectRevision) await patch($, ctx, { detail: { available: false, reason: safeText(error?.message, 280) } });
  }
}

function actions($, ctx) {
  return {
    refresh: () => quiet(refresh($, ctx)), close: () => quiet(close($, ctx)),
    tab: tab => { ctx.inspectRevision++; return patch($, ctx, { tab, detail: null, selection: null }); },
    filter: async () => { const s = await read($); await patch($, ctx, { changedOnly: !s.changedOnly, listPage: 0 }); },
    listPage: listPage => patch($, ctx, { listPage }),
    inspect: requestId => inspectRequest($, ctx, { requestId, side: 'compressed', message: 0, page: 0 }),
    side: async side => { const s = await read($); if (s.selection) await inspectRequest($, ctx, { ...s.selection, side, message: 0, page: 0 }); },
    message: async message => { const s = await read($); if (s.selection) await inspectRequest($, ctx, { ...s.selection, message, page: 0 }); },
    moveText: async (delta, screens) => {
      const s = await read($);
      if (!s.detail?.available || !s.selection) return;
      const next = Math.min(s.textPage, screens - 1) + delta;
      if (next >= 0 && next < screens) { await patch($, ctx, { textPage: next }); return; }
      const page = s.detail.page + delta;
      if (page >= 0 && page < s.detail.pages) await inspectRequest($, ctx, { ...s.selection, page });
    },
  };
}


/** @type {import('claude-code').Register} */
export function register(on) {
  const ctx = { owner: 0, timer: null, scheduled: null, network: null, busy: false, inspectRevision: 0 };
  on('session.start', async ($, e, next) => {
    const result = await next(e);
    try {
      ctx.timer?.cancel(); ctx.scheduled?.cancel(); ctx.inspectRevision++;
      const previous = await read($);
      ctx.owner = previous.owner + 1;
      let s = initialState(ctx.owner);
      await $.state.set(REF, s);
      await $.command.register({ name: 'headroom', description: 'Show Headroom compression, retained requests, and message inspection' });
      const sid = await $.env.get('HEADROOM_MOD_SESSION_ID');
      const rawUrl = await $.env.get('HEADROOM_MOD_URL');
      if (UUID.test(sid ?? '') && rawUrl) {
        try { s = { ...s, sessionId: sid, baseUrl: localUrl(rawUrl), notice: 'Connecting to local Headroom…' }; }
        catch (error) { s.notice = safeText(error?.message, 280); }
      }
      await $.state.set(REF, s);
      if (e.isInteractive) {
        await open($, ctx, false);
        ctx.timer = $.clock.every(5000, () => quiet(refresh($, ctx)));
      }
    } catch (error) {
      // Observability must not turn a successful session start into a failed one.
      await patch($, ctx, { connection: 'setup', notice: safeText(error?.message, 280) }).catch(() => undefined);
    }
    return result;
  });
  on('turn.complete', async ($, e, next) => {
    const result = await next(e);
    try { if (!e.agentId) requestRefresh($, ctx); } catch { /* telemetry policy cannot fail a completed turn */ }
    return result;
  });
  on('session.end', async ($, e, next) => {
    try { return await next(e); } finally {
      try {
        ctx.timer?.cancel(); ctx.scheduled?.cancel(); ctx.inspectRevision++;
        await patch($, ctx, { open: false, summary: null, detail: null, selection: null });
      } catch { /* cleanup must not prevent normal session shutdown */ }
    }
  });
  on('command.run', { command: 'headroom' }, async ($, e) => {
    if (e.args?.trim() === 'close') { await close($, ctx); return { text: 'Headroom closed.' }; }
    await open($, ctx);
    return { text: 'Headroom opened. 1 Overview · 2 Requests · r Refresh · q Close.' };
  });
  on('command.run', { command: ['clear', 'resume'] }, async ($, e, next) => {
    const result = await next(e);
    await recoverState($, ctx).catch(() => undefined);
    await checked($, ctx).catch(() => undefined);
    return result;
  });
  on('ui.close', { id: PANE }, async ($, e, next) => {
    const result = await next(e);
    if (result.deny === undefined) {
      ctx.inspectRevision++;
      await patch($, ctx, { open: false, band: false, detail: null, selection: null }).catch(() => undefined);
    }
    return result;
  });
  on('ui.render', { component: 'Pane' }, async ($, e, next) => {
    if (e.requestId !== PANE) return next(e);
    const s = await read($);
    if (!s.open) return next(e);
    return renderPane($.ui.resolve(e), s, actions($, ctx), e.props.bodyColumns ?? 48, e.props.scroll?.bodyRows ?? 30);
  });
  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const s = await read($);
    if (!s.open || !s.band || e.props.hasSurvey) return next(e);
    const { Box, Text } = $.ui.resolve(e);
    const existing = await next(e);
    return Box({ flexDirection: 'column', children: [existing,
      Text({ children: `Headroom ${s.connection} · ${count(s.summary?.latest?.saved)} tokens removed · ${percent(s.summary?.latest?.percent)}. Widen the window and run /headroom for the inspector.` })] });
  });
}
