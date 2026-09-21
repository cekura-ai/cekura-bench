# Pipecat speech-to-speech reference agent

The agent bench agent under test: one realtime speech-to-speech model doing real work
on a real phone call, with tools, a system prompt and a task to finish.

The service bench measures a provider's realtime service on its own, over a direct
websocket. This is the other half. The two are never ranked against each other —
"the model is fast" and "the deployment is fast" are different claims, and one
number that mixes them answers neither.

## What is being measured

The **whole configuration**: this file, the pinned Pipecat version, the
transport, and the provider. Not "Pipecat", not the model alone. That is why it
is a single readable file rather than a framework: anyone can read it, run it,
and disagree with a choice in it.

Deliberately absent: no cascade fallback, no barge-in tuning, no custom turn
strategies, no retries. Each of those would make the agent better and the result
harder to attribute. What gets measured should be the provider plus the plainest
sensible wiring around it.

## Configuration

| Variable | Default | Notes |
|---|---|---|
| `S2S_PROVIDER` | `openai-realtime` | native: also `gemini-live`, `grok-realtime`, `gpt-live`, `nova-sonic`. cascade: `cascade-baseline`, `cascade-openai`, `cascade-google`, `cascade-grok`, `cascade-qwen` |
| `S2S_MODEL` | provider default | pin it for a reproducible run |
| `S2S_VOICE` | provider default | |
| `S2S_BACKEND_MODEL` | `gpt-5.4-mini` | `gpt-live` only, see below |
| `AWS_REGION` | `us-east-1` | `nova-sonic` only |
| `AGENT_DIR` | `appointments` | a directory under `agent-definitions/` |
| `CEKURA_API_KEY`, `CEKURA_AGENT_ID` | unset | tracing is off without both |
| `CEKURA_MODE` | `track` | `observe` also uploads audio and starts evaluation |
| provider key | — | `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GEMINI_AUTHORIZATION`), `XAI_API_KEY`, or for Bedrock either `AWS_BEARER_TOKEN_BEDROCK` or `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`) |

## The cascade, for comparison

"Is a native speech model better than the pipeline it replaces?" is the one
question a mixed board can answer and nothing else can. It is only answerable if
the two sides differ in one thing, so the cascade is **this same file** — same
prompt, same tools, same transport, same greeting — with the speech path
swapped: speech-to-text, a text model, text-to-speech, in place of one model
doing all three.

Speech-to-text and text-to-speech are held fixed across every cascade row
(`flux-general-en` and `eleven_flash_v2_5`) and only the text model changes.
Each vendor's text model is the counterpart to its own speech model, and
`cascade-baseline` belongs to no vendor.

That fixed pipeline is also the limit of what a cascade row says. A cascade's
latency is dominated by when its endpointer decides the caller stopped and how
fast its voice starts, not by its text model. So a cascade row means "this
vendor's intelligence, delivered through one named pipeline" — never "cascades
are like this". All three services are named in every record, because a row
naming only its text model would hide the two components doing most of what a
latency column measures.

Two native rows have no counterpart yet. Nova Sonic is credential-blocked, and
`gpt-live-1` delegates its reasoning to a separate text model, so a fair pairing
for it is a cascade on *that* model rather than a vendor default.

## One provider does not do its own reasoning

`gpt-live-1` converses, and hands search, reasoning and tool work to a separate
text model. Leaving that unconfigured is a supported mode and the wrong one
here: delegated work is dropped, so a scenario needing a tool fails for want of
a backend rather than for anything about the model.

So the backend is named, pinned, and written into every build record. Its row
is not comparable with a single model's row without that name, and its cost is
two models' cost rather than one.

## Two credential forms for Bedrock

Nova Sonic takes either credential AWS issues, under the names AWS gives them.
Set an access-key pair, `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, plus
`AWS_SESSION_TOKEN` if the credentials are temporary. Or set an API key in
`AWS_BEARER_TOKEN_BEDROCK`. The pair wins when both are present, because it is
the form that carries a session token.

An API key is presented as a bearer token, which needs the auth-scheme swap in
`nova_bearer.py`: the framework signs with SigV4 only, though Bedrock's own
service model declares both schemes.

A bearer token's IAM policy is region-scoped and grants
`bedrock:CallWithBearerToken`. A key can be allowed in one region while the
model is served only in another, and from the outside those two failures look
alike — so the region is recorded with every run.

## The sample rate is not a tuning knob

These realtime services **do not resample**. Each base64-encodes the audio frame
it is handed and declares a rate separately, so the pipeline rate must match what
the provider expects: 24 kHz for OpenAI Realtime, 16 kHz for Gemini Live and
Grok. Open it at the wrong rate and the model hears the caller sped up or slowed
down, transcribes the words badly, and the run looks like a model failure when it
is a wiring failure.

The telephony serializer resamples the 8 kHz phone leg to whatever the pipeline
declares, so `PROVIDERS[...].input_rate` in `bot.py` is the only place this has
to be right.

## The opening turn

A realtime model has no separate text-to-speech to hand a greeting to, so the
greeting becomes an instruction in the opening turn and the model reads it back
verbatim. Every call therefore opens from the agent, and that first context frame
is also what installs the tools on the session.

## Tools

Answered from the published contract in `agent-definitions/<AGENT_DIR>/`, served
by `mock_tools/`, which is shared with the service bench rather than reimplemented here. Two
implementations of one contract would drift, and a difference between lanes could
then be our two servers disagreeing rather than anything about the agents.

An input the table does not recognise returns an explicit miss rather than an
invented record. Inventing one would let an agent that asked for the wrong thing
score like an agent that asked for the right thing.

State is not modelled: the published tables are stateless, so a run is scored on
the trace of tool calls. Booking, then cancelling, then verifying needs a store,
and that is a change to the contract rather than something to fake in the agent.

## Observability

With `CEKURA_API_KEY` and `CEKURA_AGENT_ID` set, the run is traced through the
Cekura Pipecat SDK: transcripts, tool calls, logs and spans land against the run
instead of in a container's stdout. `track` correlates a scenario run; `observe`
additionally uploads the call audio and starts evaluation.

Every call carries a **build record** as trace metadata, and it is also logged
once at startup so the answer survives when only container logs do:

| Field | Why a call is not evidence without it |
|---|---|
| `agent_commit`, `pipecat_version`, `cekura_version` | the agent under test is this file *plus* the framework it runs on |
| `s2s_provider`, `s2s_model`, `s2s_voice` | what was measured |
| `pipeline_sample_rate` | these services do not resample; a wrong rate makes the model hear the caller at the wrong speed, which reads as a bad model rather than bad wiring |
| `agent_definition`, `system_prompt_sha256`, `first_message_sha256`, `tools` | the task, the prompt and the contract the model was given |

A phone call cannot be replayed and the provider endpoint moves underneath us, so
a recording whose configuration is unknown is not evidence of anything.

Without credentials the agent still runs and still answers the phone. A missing
key must never be the reason a benchmark call fails.

## Running it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export OPENAI_API_KEY=... S2S_PROVIDER=openai-realtime AGENT_DIR=appointments
.venv/bin/python bot.py          # local dev runner
```

For a phone call, deploy it and point a Twilio or Telnyx number at the runner;
`create_transport` selects the transport from the runner arguments, so the same
file serves local WebRTC, Daily and both telephony providers unchanged.

## Verified

Driven end to end against OpenAI Realtime with caller audio from the service bench
corpus: it spoke the contract's greeting verbatim, called `lookup_patient` with
the caller's number and `check_availability` with the requested date, and both
hit the published table. Offline tests cover the provider table, the opening
turn, the tool schema and the handler's answers.
