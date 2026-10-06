import { count, percent, ms, label, safeText, trend, textPages } from './core.mjs';

export function renderPane(ui, state, actions, columns = 48, rows = 30) {
  const { Box, Text, Button } = ui;
  const line = (value, extra = {}) => Text({ ...extra, children: value });
  const button = (key, title, fn, hotkey) => Button({ key, label: title, onPress: fn, ...(hotkey ? { hotkey } : {}) });
  const group = children => Box({ flexDirection: 'row', gap: 1, children });
  const children = [line('HEADROOM  /  COMPRESSION', { bold: true }),
    line(`${state.connection.toUpperCase()} · ${state.sessionId ? state.sessionId.slice(0, 8) : 'not linked'}`, { dimColor: true }),
    group([button('overview', 'Overview', () => actions.tab('overview'), '1'),
      button('requests', 'Requests', () => actions.tab('requests'), '2'),
      button('refresh', 'Refresh', actions.refresh, 'r'), button('close', 'Close', actions.close, 'q')])];
  if (state.notice) children.push(line(safeText(state.notice, 300), { dimColor: true }));
  const data = state.summary;
  if (!data) {
    children.push(line('Local companion required. Use headroom-mod doctor, then headroom-mod run.'),
      line('The mod observes Headroom; it does not enable compression or message capture.', { dimColor: true }));
    return Box({ flexDirection: 'column', paddingX: 1, children });
  }
  if (state.tab === 'overview') {
    const latest = data.latest, totals = data.totals;
    children.push(line('LATEST RECORDED REQUEST', { bold: true }));
    if (latest) {
      children.push(line(`${count(latest.before)} → ${count(latest.after)} tokens`),
        line(`${count(latest.saved)} removed · ${percent(latest.percent)} reduction`),
        line(`${label(latest.model, 50)} · ${ms(latest.overhead_ms)} compression`, { dimColor: true }));
      if (latest.failed) children.push(line('Latest request failed; it is excluded from savings totals.'));
      if (latest.accounting !== 'complete') children.push(line('Latest token accounting is unavailable/inconsistent.'));
    } else children.push(line('No tagged requests yet. Send a prompt through this launch.'));
    children.push(line('CLAUDE CONTEXT  (native usage)', { bold: true }),
      line(`${count(state.context?.tokens)} / ${count(state.context?.window)} · ${percent(state.context?.percent)}`),
      line('RETAINED REQUEST TOTALS', { bold: true }),
      line(`${totals.requests} requests · ${count(totals.saved)} tokens removed`),
      line(`${percent(totals.percent)} weighted reduction · ${totals.failed_requests} failed`),
      line(`${ms(totals.average_overhead_ms)} mean compression overhead`),
      line(`Cache reads ${count(totals.cache_read)} · writes ${count(totals.cache_write)}`),
      line(`${percent(totals.cache_read_percent)} provider cache-read share`),
      line(`Saved/request  ${trend(data.requests)}`, { dimColor: true }));
    if (totals.unaccounted_requests) children.push(line(`${totals.unaccounted_requests} requests excluded from savings (failed or missing accounting).`));
    const transforms = Object.entries(totals.transforms ?? {}).slice(0, Math.max(1, Math.min(4, rows - 25)));
    if (transforms.length) children.push(line('Transforms (request counts, not additive savings)', { dimColor: true }),
      ...transforms.map(([name, n]) => line(`${label(name, 36)}  ${n}`)));
    children.push(line('Includes inherited child requests. Cumulative request tokens are not unique context or bill savings.', { dimColor: true }));
    if (data.retention?.window_full) children.push(line('Retention window full: older requests may have been evicted.'));
  } else if (state.tab === 'requests') {
    const items = state.changedOnly ? data.requests.filter(r => r.saved !== null && r.saved !== 0) : data.requests;
    const perPage = Math.max(1, Math.min(5, Math.floor((rows - 12) / 3)));
    const pages = Math.max(1, Math.ceil(items.length / perPage));
    const page = Math.min(state.listPage, pages - 1);
    children.push(group([button('filter', state.changedOnly ? 'Changed only' : 'All requests', actions.filter),
      button('older', 'Older', () => actions.listPage(Math.min(page + 1, pages - 1))),
      button('newer', 'Newer', () => actions.listPage(Math.max(0, page - 1)))]),
      line(`Page ${page + 1}/${pages} · newest ${data.requests.length} retained requests`, { dimColor: true }));
    for (const r of items.slice(page * perPage, (page + 1) * perPage)) {
      children.push(line(`${label(r.timestamp, 19)}  ${label(r.model, 28)}`),
        line(`${count(r.before)} → ${count(r.after)} · ${percent(r.percent)}${r.failed ? ' · FAILED' : ''}`));
      if (r.inspectable_id) children.push(button(`inspect-${r.request_id}`, r.has_messages ? 'Review messages' : 'Capture status', () => actions.inspect(r.request_id)));
    }
    if (!items.length) children.push(line('No matching requests in the retained window.'));
    if (!data.log_full_messages) children.push(line('Message capture is off. Metrics still work. --log-messages is an explicit proxy opt-in.'));
  } else {
    const d = state.detail, sel = state.selection;
    children.push(line(`REQUEST ${label(sel?.requestId, 40)}`, { bold: true }),
      group(['original', 'compressed', 'diff'].map(side => button(side, side === 'diff' ? 'Request diff' : side, () => actions.side(side)))));
    if (!d) children.push(line('Loading selected preview…'));
    else if (!d.available) children.push(line(safeText(d.reason, 300)));
    else {
      const display = textPages(d.text, columns, rows);
      const textPage = Math.min(state.textPage, display.length - 1);
      children.push(line(`${d.side} · message ${d.message + 1}/${d.message_count} · chunk ${d.page + 1}/${d.pages}`, { dimColor: true }),
        group([button('message-prev', 'Prev msg', () => actions.message(Math.max(0, d.message - 1))),
          button('message-next', 'Next msg', () => actions.message(Math.min(d.message_count - 1, d.message + 1))),
          button('text-prev', 'Prev', () => actions.moveText(-1, display.length)),
          button('text-next', 'Next', () => actions.moveText(1, display.length))]),
        line(display[textPage]), line(`Screen ${textPage + 1}/${display.length}${d.truncated ? ' · TRUNCATED PREVIEW' : ''}`, { dimColor: true }));
      children.push(line('Original/compressed indices are independent. Diff is request-level.', { dimColor: true }));
    }
  }
  return Box({ flexDirection: 'column', paddingX: 1, children });
}
