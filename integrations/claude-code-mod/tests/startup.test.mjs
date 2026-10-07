import test from 'node:test';
import assert from 'node:assert/strict';
import { renderPane } from '../plugins/headroom-sidebar/hooks/render.mjs';
import { initialState } from '../plugins/headroom-sidebar/hooks/core.mjs';

const ui = Object.fromEntries(['Box', 'Text', 'Button'].map(type => [type, props => ({ type, ...props })]));
function buttons(node) {
  if (!node || typeof node !== 'object') return [];
  return [...(node.type === 'Button' ? [node] : []), ...[node.children].flat().flatMap(buttons)];
}
test('setup sidebar offers startup and delegates close to the host', () => {
  let started = false;
  const pane = renderPane(ui, initialState(), { start: () => { started = true; }, tab() {}, refresh() {} });
  const controls = buttons(pane);
  assert.equal(controls.some(b => b.label === 'Close'), false);
  const start = controls.find(b => b.label === 'Start Headroom');
  assert.ok(start);
  start.onPress();
  assert.equal(started, true);
});

test('live sidebar has no startup control', () => {
  const pane = renderPane(ui, { ...initialState(), connection: 'live' }, { tab() {}, refresh() {} });
  assert.equal(buttons(pane).some(b => b.label === 'Start Headroom'), false);
});
