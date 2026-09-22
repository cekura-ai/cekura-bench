# GPT Realtime reference harness

This is the OpenAI Realtime SIP/sideband harness used for the benchmark. The OpenAI webhook accepts SIP invitations, connects a Realtime sideband session, forwards the canonical tools, and publishes the normalized transcript to Cekura after the call.

Run `npm ci && npm test` on Node 20+. Deploy with Vercel and configure the required OpenAI, Twilio, Cekura, and canonical-agent environment variables. The harness source deliberately excludes Vercel deployment state and credentials.
