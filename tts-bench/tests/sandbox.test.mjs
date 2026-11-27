// The launcher's local decisions: grouping by account, names, and reading keys. No remote calls.
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { campaignArgs, dotenv, groups, sandboxName } from '../scripts/sandbox.mjs';

const entries = [
  { provider: 'elevenlabs', model: 'flash', credential: 'ELEVENLABS_API_KEY', host: 'api.elevenlabs.io' },
  { provider: 'elevenlabs-dialogue', model: 'v3', credential: 'ELEVENLABS_API_KEY', host: 'api.elevenlabs.io' },
  { provider: 'gemini', model: 'a', credential: 'GEMINI_AUTHORIZATION', host: 'generativelanguage.googleapis.com' },
];

test('two protocols on one key share a sandbox and run in sequence', () => {
  const g = groups(entries);
  assert.deepEqual(g.map(x => x.name), ['elevenlabs', 'gemini']);
  assert.deepEqual(g[0].models.map(m => `${m.provider}/${m.model}`), ['elevenlabs/flash', 'elevenlabs-dialogue/v3']);
  assert.deepEqual(g[0].hosts, ['api.elevenlabs.io']);
});

test('a provider filter keeps only its models', () => {
  assert.deepEqual(groups(entries, ['elevenlabs-dialogue'])[0].models, [{ provider: 'elevenlabs-dialogue', model: 'v3' }]);
});

test('sandbox names are stable and safe', () => {
  assert.equal(sandboxName('Campaign_A', 'deepgram'), 'tts-campaign-a-deepgram');
});

test('only the named keys are read, quoted or not', () => {
  const dir = mkdtempSync(join(tmpdir(), 'tts-env-'));
  writeFileSync(join(dir, '.env'), 'A_KEY="one"\nexport B_KEY=two\nC_KEY=three\nEMPTY=\n');
  assert.deepEqual(dotenv(join(dir, '.env'), ['A_KEY', 'B_KEY', 'EMPTY']), { A_KEY: 'one', B_KEY: 'two' });
});

test('the voice and adapter options reach the campaign command', () => {
  const args = campaignArgs({ provider: 'fish-audio-http', model: 'm', label: 'l', suite: 'full', repeats: 3, out: '/o',
    voice: 'v', options: ['latency=low', 'normalize=true'], items: [] });
  assert.deepEqual(args.slice(args.indexOf('--voice')), ['--voice', 'v', '--option', 'latency=low', '--option', 'normalize=true']);
  assert.ok(!campaignArgs({ provider: 'p', model: 'm', label: 'l', suite: 'full', repeats: 3, out: '/o' }).includes('--option'));
  assert.throws(() => campaignArgs({ provider: 'p', model: 'm', label: 'l', suite: 'full', repeats: 3, out: '/o', options: ['low'] }),
    /key=value/);
});

test('a probe subset without the sentinel reaches the campaign command', () => {
  const args = campaignArgs({ provider: 'p', model: 'm', label: 'l', suite: 'full', repeats: 3, out: '/o',
    items: ['prose.greet'], probes: ['one_shot'], sentinel: false });
  assert.deepEqual(args.slice(args.indexOf('--item')), ['--item', 'prose.greet', '--probe', 'one_shot', '--no-sentinel']);
  assert.ok(!campaignArgs({ provider: 'p', model: 'm', label: 'l', suite: 'full', repeats: 3, out: '/o' }).includes('--no-sentinel'));
});
