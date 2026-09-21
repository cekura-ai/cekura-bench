# The Lane B agent as Pipecat Cloud runs it.
#
# Deliberately plain. Tracing, transcripts, tool calls and logs all go through
# the Cekura SDK the agent already imports, so there is no log shipper, no
# sidecar and no agent-specific machinery in here. What the image contains is
# the agent, the published contract it answers from, and the pinned versions
# that make a result name a configuration.
#
# Built from the repository root, not from this directory, because the agent
# reads the agent definitions and the mock-tool server from the repository:
#
#     docker build -f reference-agents/pipecat-s2s/Dockerfile .

# Pinned by digest. The base image is the harness that runs the agent, so a
# retag of a mutable tag would change the thing under test without changing
# anything in this repository. Digest is 0.1.29-py3.12.
FROM dailyco/pipecat-base:0.1.29-py3.12@sha256:c34a7c605b0f42d790a7593c9870417a098b0d6258b87119ebd6142d27c11e82

COPY reference-agents/pipecat-s2s/requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --retries 3 --timeout 300 -r requirements.txt

# The contract the agent answers from, and the server that serves it. Copied
# rather than baked into the agent so that the image and the published repository
# hold the same bytes, and a reader can diff them.
COPY mock_tools ./mock_tools
COPY agent-definitions ./agent-definitions

# The agent itself. ``bot.py`` last, because it is what changes between builds.
COPY reference-agents/pipecat-s2s/nova_bearer.py ./nova_bearer.py
COPY reference-agents/pipecat-s2s/bot.py ./bot.py

# The agent resolves its own commit to put on every call's build record, and
# there is no git history in the image. Supplied at build time instead:
#
#     docker build --build-arg AGENT_COMMIT="$(git rev-parse HEAD)" ...
#
# A build that forgets it records "unknown", which is visible rather than wrong.
ARG AGENT_COMMIT=unknown
ENV AGENT_COMMIT=${AGENT_COMMIT}
