import { canonicalConfig, realtimeWorkflowConfigs } from "../lib/canonical.mjs";

export default function handler(_request, response) {
  try {
    const config = canonicalConfig();
    const workflows = realtimeWorkflowConfigs();
    response.status(200).json({
      ok: true,
      canonical_prompt_sha256: config.promptSha256,
      first_message: config.firstMessage,
      tools: config.tools.map((tool) => tool.name),
      worker_mode: "direct-sideband-hobby-4m50s",
      voice_model: "gpt-live-1",
      delegated_backend_model: "gpt-5.6-terra",
      sip_event: "live.transport.incoming",
      openai_webhook_configured: Boolean(process.env.OPENAI_WEBHOOK_SECRET),
      workflows: workflows.map(({ key, agentId, promptSha256, tools }) => ({ key, agent_id: agentId, prompt_sha256: promptSha256, tools: tools.map((tool) => tool.name) })),
    });
  } catch (error) {
    response.status(503).json({ ok: false, error: error.message });
  }
}
