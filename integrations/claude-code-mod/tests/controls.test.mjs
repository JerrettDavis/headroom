import test from 'node:test';
import assert from 'node:assert/strict';
import { host, press, words, summary, response, SID, OTHER, EPOCH, settle, flatten } from './host-fixture.mjs';

test('late control completion cannot repopulate a different native conversation', async () => {
  let release;
  const h = host({ transport: async (url) => url.endsWith('/compression')
    ? new Promise(resolve => { release = resolve; })
    : response({ ...summary(), compression_enabled: true }) });
  await h.start();
  const toggling = press(h, 'compression'); await settle();
  h.ctl.sid = OTHER;
  await h.dispatch('command.run', { command: 'clear' });
  release(response({ schema_version: 1, session_id: SID, epoch: EPOCH, compression_enabled: false }));
  await toggling;
  assert.equal(h.state().compressionEnabled, null);
  assert.equal(h.state().summary, null);
});

test('changing the window discards a pending response for the previous window', async () => {
  const h = host({ transport: async () => response({ ...summary(), compression_enabled: true }) });
  await h.start();
  let release;
  h.ctl.transport = () => new Promise(resolve => { release = resolve; });
  await h.dispatch('turn.complete'); await h.advance(2);
  await press(h, 'window-15m');
  release(response({ ...summary(), compression_enabled: true })); await settle();
  assert.equal(h.state().summary, null);
  assert.equal(h.state().timeWindow, '15m');
  h.ctl.transport = async () => response({ ...summary(), compression_enabled: true });
  await h.advance(5000);
  assert.ok(h.calls.at(-1)[1].includes('window=15m'));
});

test('legacy read-only companion exposes telemetry without unsupported control buttons', async () => {
  const h = host(); await h.start();
  const keys = flatten(await h.pane()).filter(n => n.type === 'Button').map(n => n.key);
  assert.ok(!keys.some(k => k === 'compression' || k === 'reset' || k.startsWith('window-')));
  assert.match(words(await h.pane()), /Upgrade.*companion/i);
});

test('proxy restart removes a stale paused notice and restores confirmed on state', async () => {
  const h = host({ transport: async () => response({ ...summary(), compression_enabled: false }) });
  await h.start();
  await h.$.state.set({ plugin: 'headroom-sidebar', key: 'model' }, { ...h.state(), controlNotice: 'Compression PAUSED · next requests pass through unchanged.' });
  h.ctl.transport = async () => response({ ...summary(SID, '44444444-4444-4444-8444-444444444444'), compression_enabled: true });
  await h.advance(5000);
  assert.equal(h.state().compressionEnabled, true);
  assert.doesNotMatch(words(await h.pane()), /PAUSED/);
});

test('compression toggle waits for proxy confirmation and uses a session-scoped POST', async () => {
  let enabled = true;
  const h = host({ transport: async (url, init) => {
    if (url.endsWith('/compression')) {
      assert.equal(init.method, 'POST');
      assert.ok(url.includes(SID));
      enabled = JSON.parse(init.body).enabled;
      return response({ schema_version: 1, session_id: SID, epoch: EPOCH, compression_enabled: enabled });
    }
    return response({ ...summary(), compression_enabled: enabled });
  } });
  await h.start(); await press(h, 'compression');
  assert.equal(h.state().compressionEnabled, false);
  assert.match(words(await h.pane()), /PAUSED/);
  await press(h, 'compression');
  assert.equal(h.state().compressionEnabled, true);
});

test('failed compression change keeps confirmed state and displays failure', async () => {
  const h = host({ transport: async (url) => url.endsWith('/compression')
    ? { ok: false, status: 403, text: '{}' }
    : response({ ...summary(), compression_enabled: true }) });
  await h.start(); await press(h, 'compression');
  assert.equal(h.state().compressionEnabled, true);
  assert.match(words(await h.pane()), /failed.*403/i);
});

test('reset and time window refresh scoped metrics and discard old inspector', async () => {
  let reset = false;
  const h = host({ transport: async (url, init) => {
    if (url.endsWith('/reset')) {
      assert.equal(init.method, 'POST'); reset = true;
      return response({ schema_version: 1, session_id: SID, epoch: EPOCH, compression_enabled: true });
    }
    const s = { ...summary(), compression_enabled: true };
    if (reset) { s.totals.requests = 0; s.totals.saved = null; s.requests = []; s.latest = null; }
    return response(s);
  } });
  await h.start(); await press(h, 'window-1h'); await h.advance(2);
  assert.equal(h.state().timeWindow, '1h');
  assert.ok(h.calls.some(c => c[0] === 'http.fetch' && c[1].includes('window=1h')));
  await press(h, 'reset'); await h.advance(2);
  assert.equal(h.state().summary.totals.requests, 0);
  assert.equal(h.state().detail, null);
});

test('bars show percentage with unknown values distinct from zero', async () => {
  const h = host(); await h.start();
  assert.match(words(await h.pane()), /[█░]+.*40\.0%/);
  assert.match(words(await h.pane()), /[█░]+.*0\.3%/);
  const s = summary(); s.totals.percent = null; s.latest.percent = null;
  h.ctl.transport = async () => response(s); await h.advance(5000); await settle();
  assert.match(words(await h.pane()), /Reduction unavailable/);
});
