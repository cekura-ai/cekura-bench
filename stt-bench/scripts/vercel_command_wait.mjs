import { setTimeout as delay } from 'node:timers/promises';

export async function waitForCompletion(sandbox, id, { retryDelayMs = 5000, maxFailures = 8 } = {}) {
  let failures = 0;
  for (;;) {
    try {
      const command = await sandbox.getCommand(id, { signal: AbortSignal.timeout(15000) });
      const finished = await command.wait({ signal: AbortSignal.timeout(30000) });
      return finished.exitCode;
    } catch (error) {
      // These cancel only a read-only wait, never the remote command itself.
      if (['TimeoutError', 'AbortError'].includes(error.name)) failures = 0;
      else if (error.name === 'TypeError' || error.response?.status >= 500) {
        if (++failures > maxFailures) throw error;
        await delay(retryDelayMs);
      } else throw error;
    }
  }
}
