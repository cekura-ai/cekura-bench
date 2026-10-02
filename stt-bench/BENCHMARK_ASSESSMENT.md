# Benchmark assessment: provisional combined ranking

The combined benchmark improves scoring correctness, comparison on identical
inputs, and reproducibility. It supports a provisional accuracy ranking for this
frozen dataset. It does not establish a generally best provider for voice agents.
Passing software tests is separate from validating reference quality and whether
the sample represents production speech.

## Saved-evidence checks (September 15, 2026)

The combined-v1 results pool 20,865 public and 12,555 private reference words.
Reson8 has 763 errors (2.28%); Cartesia has 845 (2.53%). Their difference is
82 errors, or about 0.245 percentage points. No confidence interval has been
established for performance on new conversations.

Reson8 remained first after excluding each of the four private conversation
groups in turn, both with direct word pooling and with the original 62.4%/37.6%
dataset weights retained. This is evidence against one private conversation
alone driving its lead, not proof of generalization.

Changing the private contribution to 0%, 5%, or 10% puts Gemini first. At 15%,
25%, 37.6%, or 50%, Reson8 is first. The eight private files are speaker tracks
from four conversations, not eight independent conversations. Their 37.6% weight
comes from word counts; it has not been validated against production traffic.

The frozen public intersection excludes 136 of 1,000 clips. For providers with
usable results on all 136 excluded clips, their WER is higher there: Reson8
3.32% versus 1.92% on included clips; Cartesia 3.35% versus 2.22%; AssemblyAI
2.95% versus 1.74%. Matching inputs improves comparability but does not remove
the selection effect of using clips successfully measured for every model.

On the same 1,000 public clips, available-text WER at +500 ms is 10.79% for
Reson8 and 2.37% for Cartesia. These first-attempt snapshots can include partial
text. Eventual accuracy alone therefore cannot identify the best provider for a
fast voice agent. Historical finalization protocols and collection conditions
still differ; future configurations do not change those historical measurements.

## Reference provenance

[Pipecat's current documentation](https://github.com/pipecat-ai/stt-benchmark/blob/main/README.md#dataset)
describes Gemini-generated, human-reviewed references. Review coverage for the
exact frozen dataset revision remains unverified here. Gemini's participation
is a reason to investigate possible bias, not evidence that its lead is invalid.
Private reference provenance and independent listening verification remain
unresolved. Pooling datasets does not resolve reference bias.

## Prioritized follow-up

1. Verify references and speech-end boundaries on a random sample and major
   provider disagreements, with provider names hidden.
2. Expand independent private conversations across speakers, domains, accents,
   recording conditions, and audio quality.
3. Make deadline accuracy prominent, using matched observations and visible
   failures alongside eventual accuracy.
4. Separate core-suite reliability from FLEURS coverage. The current 1,188-item
   denominator includes 180 FLEURS clips most models did not attempt.
5. Freeze the next evaluation's sampling, primary metrics, retries, and weighting
   before seeing results. Treat the current weighting change as exploratory.
6. Validate future finalization profiles in a small controlled experiment after
   resolving capture/reconstruction issues. No new provider run is implied here.

The evidence is in the locally supplied `reports/offline-rescore-combined-v1/`
results and per-item audit; those files and private recordings are not included
in the public repository. The existing pilot is diagnostic and does not validate
the dashboard's AssemblyAI `min_latency` profile.
