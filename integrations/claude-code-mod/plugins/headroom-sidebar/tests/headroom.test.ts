// Native Claude test-kit gate. Run: claude plugin test plugins/headroom-sidebar
// Not executed by Node tests, and never represented as passing without Claude.
import { describe, expect, mock, test } from 'claude-code/testing';

const sid = '11111111-1111-4111-8111-111111111111';
const epoch = '33333333-3333-4333-8333-333333333333';
const payload = {
  schema_version: 1, session_id: sid, epoch,
  basis: 'retained_request_totals_not_unique_context_or_lifetime',
  totals: { requests: 0, accounted_requests: 0, failed_requests: 0, unaccounted_requests: 0, saved: null, transforms: {} },
  latest: null, requests: [], retention: { window_full: false }, log_full_messages: false,
};

describe('headroom', () => {
  test('native state, command registration and narrow terminal fallback', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, { HEADROOM_MOD_SESSION_ID: sid, HEADROOM_MOD_URL: 'http://127.0.0.1:8787' });
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('session.usage', () => ({ value: { startedAt: 0, rateLimits: [], context: { tokens: 600, window: 200000, percent: 0.3 } } }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: false } } as any));
    on('http.fetch', () => ({ value: { status: 200, ok: true, headers: {}, text: JSON.stringify(payload) } }));
    on('ui.render', { component: 'AbovePrompt' }, ($, e) => $.ui.resolve(e).Text({ children: 'existing band' }));
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2);
    await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'AbovePrompt',
      props: { hasSurvey: false, isWorking: false, maxRows: 10, bodyColumns: 100 } } as any);
    expect(await ui.find({ type: 'Text', text: /Headroom live/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /existing band/ })).toBeDefined();
    await ui.unmount();
  });

  test('native pane draws a scoped empty state, not proxy-wide zero savings', async ($, on) => {
    const clock = mock.clock(on);
    mock.env(on, { HEADROOM_MOD_SESSION_ID: sid, HEADROOM_MOD_URL: 'http://127.0.0.1:8787' });
    on('session.start', ($, e) => ({ cwd: e.cwd }));
    on('session.id', () => ({ value: sid }));
    on('session.usage', () => ({ value: { startedAt: 0, rateLimits: [], context: { tokens: 600, window: 200000, percent: 0.3 } } }));
    on('command.register', ($, e) => ({ value: { command: e.name } }));
    on('ui.open', () => ({ value: { isPlaced: true } } as any));
    on('http.fetch', () => ({ value: { status: 200, ok: true, headers: {}, text: JSON.stringify(payload) } }));
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' } as any);
    await clock.advance(2);
    await clock.settle();
    const ui = await $.ui.mount({ plugin: 'headroom-sidebar', surface: 'terminal', component: 'Pane', requestId: 'headroom-sidebar',
      props: { bodyColumns: 48, placement: 'dock', scroll: { bodyRows: 35 } } } as any);
    expect(await ui.find({ type: 'Text', text: /No tagged requests yet/ })).toBeDefined();
    expect(await ui.find({ type: 'Text', text: /RETAINED REQUEST TOTALS/ })).toBeDefined();
    await ui.unmount();
  });
});
