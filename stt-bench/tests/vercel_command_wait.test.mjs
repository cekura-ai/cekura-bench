import test from 'node:test';
import assert from 'node:assert/strict';
import { waitForCompletion } from '../scripts/vercel_command_wait.mjs';

test('reattaches to the same command after timeouts and transient connection errors', async () => {
  const observed = [];
  const failures = [new DOMException('poll ended', 'TimeoutError'), new TypeError('fetch failed')];
  const sandbox = { async getCommand(id, options) {
    observed.push(id);
    assert.ok(options.signal);
    return { async wait(options) {
      assert.ok(options.signal);
      if (failures.length) throw failures.shift();
      return { exitCode: 0 };
    }};
  }};
  assert.equal(await waitForCompletion(sandbox, 'existing-command', { retryDelayMs: 0 }), 0);
  assert.deepEqual(observed, ['existing-command', 'existing-command', 'existing-command']);
});

test('does not retry forbidden responses or conceal a remote failure exit code', async () => {
  const forbidden = Object.assign(new Error('forbidden'), { response: { status: 403 } });
  await assert.rejects(waitForCompletion({ async getCommand() { throw forbidden; } }, 'id'), /forbidden/);
  assert.equal(await waitForCompletion({ async getCommand() { return { async wait() { return { exitCode: 1 }; } }; } }, 'id'), 1);
});

test('stops after bounded persistent network failures', async () => {
  let calls = 0;
  await assert.rejects(waitForCompletion({ async getCommand() {
    calls++;
    throw new TypeError('fetch failed');
  } }, 'id', { maxFailures: 2, retryDelayMs: 0 }), /fetch failed/);
  assert.equal(calls, 3);
});
