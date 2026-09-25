#!/usr/bin/env bash
# Deploy the agent bench agent to Pipecat Cloud with a cloud build.
#
# No registry is involved: the build context goes to Pipecat Cloud and the
# image is built there, so the only thing a machine needs is a Pipecat Cloud
# login (``pipecat cloud auth login``) and this repository. What ends up in the
# context is governed by the repository's .dockerignore -- the run artifacts and
# the credentials file are excluded there, not here.
#
# The cloud build takes no ``--build-arg``, so the commit is written into the
# context as a file the Dockerfile copies if present and ``bot.py`` reads when
# no AGENT_COMMIT is set. A dirty tree is refused: the commit on the record has
# to be the code that was built, or the record says nothing.
#
#     reference-agents/pipecat-s2s/deploy.sh              # agent and secret set from pcc-deploy.toml
#     reference-agents/pipecat-s2s/deploy.sh --min-agents 1   # any extra flags go to `pipecat cloud deploy`
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
cd "$root"

if [ -n "$(git status --porcelain -- reference-agents mock_tools agent-definitions)" ]; then
  echo "refusing to deploy from a dirty tree: the commit stamp would not name what was built" >&2
  git status --short -- reference-agents mock_tools agent-definitions >&2
  exit 1
fi

commit=$(git rev-parse HEAD)
printf '%s\n' "$commit" > "$here/agent-commit"
trap 'rm -f "$here/agent-commit"' EXIT

echo "deploying $commit"
pipecat cloud deploy \
  --config-file "$here/pcc-deploy.toml" \
  --build-dir "$root" \
  --dockerfile reference-agents/pipecat-s2s/Dockerfile \
  --no-credentials \
  "$@"
