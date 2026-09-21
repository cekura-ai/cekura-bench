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

**One deployment answers for every row.** What is being measured — the provider,
the model, the voice, the agent definition — is decided per call by the session
that starts it, not baked into the image. A cohort is therefore a set of run
configurations against one deployed agent, and a new model is a new row rather
than a new deployment.

These keys arrive in the session body (lowercase). The environment is the
fallback, under the uppercase name, which is what makes `python bot.py` on a
laptop work unchanged and lets a deployment carry a default:

| Key | Default | Notes |
|---|---|---|
| `s2s_provider` | `openai-realtime` | native: also `gemini-live`, `grok-realtime`, `gpt-live`, `nova-sonic`, `qwen-realtime`. cascade: `cascade-baseline`, `cascade-openai`, `cascade-google`, `cascade-grok`, `cascade-qwen` |
| `s2s_model` | provider default | pin it for a reproducible run |
| `s2s_voice` | provider default | |
| `s2s_backend_model` | `gpt-5.4-mini` | `gpt-live` only, see below |
| `aws_region` | `us-west-2` | `nova-sonic` only; must be a region serving the model *and* granted to the credentials |
| `qwen_region` | `singapore` | `qwen-realtime` only; also `beijing` |
| `qwen_workspace_id` | **required for `qwen-realtime`** | names the Alibaba workspace whose endpoint answers |
| `cascade_tts_voice` | fixed voice id | cascade rows only |
| `agent_dir` | **required** | a directory under `agent-definitions/`; no default on purpose |
| `cekura_mode` | `track` | `observe` also uploads audio and starts evaluation |

Only those keys are read from a session. The platform flattens a scenario's own
variables into the same body, so an unfiltered read would let a fixture field
decide the provider or the agent definition — a scored run against the wrong
contract, with nothing in the record saying so.

**Credentials are environment-only** and are never read from a session. A key
sent with a request is copied into every log, trace and session record that
quotes the body.

| Variable | Notes |
|---|---|
| provider key | `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GEMINI_AUTHORIZATION`), `XAI_API_KEY`, `DASHSCOPE_API_KEY`, or for Bedrock `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ `AWS_SESSION_TOKEN`); see below for why an API key cannot work |
| `DEEPGRAM_API_KEY`, `ELEVENLABS_API_KEY` | cascade rows only |
| `CEKURA_API_KEY`, `CEKURA_AGENT_ID` | tracing is off without both |
| `AGENT_COMMIT` | stamped by the build; names what was actually deployed |

## Start-up

Because the provider is unknown until a call arrives, the worker imports every
provider SDK at start-up rather than the one it will use, along with the Cekura
tracer that `create_task` would otherwise load inside the first call. Measured
here: 1.8 s and ~160 MB at boot, against 0.2–0.6 s per provider SDK and ~0.1 s
for the tracer — all of it moved out of the window a first response is timed in,
where it would read as a slow model rather than a cold container.

Nothing in the warm-up touches the network. Opening a throwaway connection per
provider does not speed up the real first one — a TLS session does not resume
across a separate context, and no DNS result is cached in-process — while every
handshake delays the moment a worker can answer at all.

A worker answers more than one call, so every record names the worker and how
many calls it had already answered. The first call on a worker is the one that
pays whatever start-up cost is left, and without that field it cannot be told
apart from a slow model.

The deployment therefore keeps one warm replica rather than scaling to zero. An
idle replica costs money and measures nothing, which is the argument for zero;
the argument against it is that the cold start lands inside the window a first
response is timed in, and a latency column that includes it is not defensible.

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

## Bedrock needs signed credentials, not an API key

Nova Sonic is reached only through `InvokeModelWithBidirectionalStream`, and AWS
excludes that operation from Bedrock API-key (bearer) authentication. A bearer
token therefore cannot open a Nova Sonic session in any region, however its
policy is written. Set an access-key pair instead: `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`, plus `AWS_SESSION_TOKEN` if the credentials are
temporary. The identity needs `bedrock:InvokeModel` and
`bedrock:InvokeModelWithBidirectionalStream` on the model's foundation-model ARN.

Prefer a long-lived key pair. Temporary credentials expire within twelve hours
and nothing here refreshes them, so a campaign running past the expiry would
fail partway through a cohort.

Region matters twice over: the model is served in four regions, and credentials
are granted in some subset of those. Both failures are the same
`AccessDeniedException` from the outside, which is why the region is recorded
with every run.

## One provider has no framework service

Pipecat ships a realtime service for every other model on this board and none
for Qwen, so `qwen_realtime.py` speaks that protocol directly. It is not a
subclass of the OpenAI Realtime service: Qwen follows the earlier shape of that
protocol, where modalities, audio formats and turn detection sit at the top
level of the session rather than inside a nested audio object, and Pipecat's
service now speaks the later one. Subclassing would mean replacing the session
encoding, the event models and the tool encoding, and would break whenever the
framework revises a shape Qwen does not follow.

Two details there are easy to get wrong and silent when wrong. Qwen takes tools
in the nested form chat completions uses rather than the flat form the realtime
protocols use, and a tool sent in the wrong form is ignored rather than
rejected, so the model fails a scenario for having no tools. And it listens at
16 kHz while speaking at 24 kHz, so outbound frames declare the higher rate and
the transport resamples; declaring one rate for both plays the voice at the
wrong speed, which reads as a bad model.

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

Answered from the published contract in `agent-definitions/<agent_dir>/`, served
by `mock_tools/`, which is shared with the service bench rather than reimplemented here. Two
implementations of one contract would drift, and a difference between lanes could
then be our two servers disagreeing rather than anything about the agents.

An input the table does not recognise returns an explicit miss rather than an
invented record. Inventing one would let an agent that asked for the wrong thing
score like an agent that asked for the right thing.

Two tools are code-defined rather than looked up: `end_call` and
`transfer_call`. Neither returns a record, so neither belongs in the published
tables — but the prompt instructs the agent to end a call and to announce a
transfer, and a scored call is judged on whether it terminated appropriately. An
agent with no way to hang up fails that for a reason having nothing to do with
the model. The transfer is a mock: a benchmark deployment has no second leg, so
the call completes after the announced handover.

`agent_dir` has no default. A call that ran the wrong agent definition would
produce a plausible, scored, wrong result, with nothing in the transcript
saying which contract it was answering.

State is not modelled: the published tables are stateless, so a run is scored on
the trace of tool calls. Booking, then cancelling, then verifying needs a store,
and that is a change to the contract rather than something to fake in the agent.

## Observability

With `CEKURA_API_KEY` and `CEKURA_AGENT_ID` set, the run is traced through the
Cekura Pipecat SDK: transcripts, tool calls, logs and spans land against the run
instead of in a container's stdout. `track` correlates a scenario run; `observe`
additionally uploads the call audio and starts evaluation.

Every call carries a **build record** as trace metadata, and it is also logged
once at startup so the answer survives when only container logs do. The record is
completed when the caller disconnects, because the tool trace is only whole once
the call is over and the record cannot be changed after it is posted:

| Field | Why a call is not evidence without it |
|---|---|
| `agent_commit`, `pipecat_version`, `cekura_version` | the agent under test is this file *plus* the framework it runs on |
| `s2s_provider`, `s2s_model`, `s2s_voice` | what was measured |
| `pipeline_sample_rate` | these services do not resample; a wrong rate makes the model hear the caller at the wrong speed, which reads as a bad model rather than bad wiring |
| `agent_definition`, `system_prompt_sha256`, `first_message_sha256`, `tools` | the task, the prompt and the contract the model was given |
| `tool_calls`, `tool_call_count`, `tool_calls_matched` | what the agent asked of its tools, with arguments, answers and ordering; recorded here rather than read from the framework's spans, which two of the five providers emit and three do not |
| `config_source` | whether the session or the image decided the configuration — one deployment answers for every row, so a row that does not say which is a row nobody can place |
| `worker_instance`, `worker_call` | which worker answered and how many calls it had already answered; the first call on a worker carries any start-up cost the warm-up did not remove |

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

## Deploying it

```bash
pipecat cloud auth login                 # once per machine
reference-agents/pipecat-s2s/deploy.sh   # cloud build; agent and secret set from pcc-deploy.toml
```

The build runs on Pipecat Cloud from the repository as a context, so no
registry is needed. `.dockerignore` decides what leaves the machine: the run
artifacts and the credentials file do not. The script refuses a dirty tree and
writes the commit into the context, because the cloud build takes no build
arguments and a record that cannot name the code it ran is not a record.

Credentials live in the secret set named in `pcc-deploy.toml`, never in the
image or the session. Create it once with `pipecat cloud secrets set
cekura-s2s-secrets --file <env file>` holding the variables in the table above;
add `CEKURA_API_KEY` and `CEKURA_AGENT_ID` when the platform agent exists.

## Verified

Driven end to end against OpenAI Realtime with caller audio from the service bench
corpus: it spoke the contract's greeting verbatim, called `lookup_patient` with
the caller's number and `check_availability` with the requested date, and both
hit the published table. Offline tests cover the provider table, the opening
turn, the tool schema and the handler's answers.
