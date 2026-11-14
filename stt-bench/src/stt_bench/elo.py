"""STT benchmark ELO: one rating from head-to-head comparisons on every clip (method in STT_ELO.md).

Measures and weights: Ocular turns WER 10, Pipecat WER 10, semantic WER 10, phone 12, callcenter 10, babble 8,
tv 5, reverb 5, short replies (<5 words) 10, entities 20.  A measure with no data is dropped and the rest renormalized.
Per clip (every clip with reference words): score = 100 - WER; draw if |gap| <= 5; matches weighted by word count;
a failed clip loses to a valid one; clips invalid for our-side reasons (pacing) are missing, not failures.
Short replies: score = -errors, no margin.  Entities: score = entities captured exactly, no margin.
Each measure's matches are scaled to its weight.  Bradley-Terry fit, ELO = 1000 + 400 log10(p).
Bootstrap 95% ranges, resampling conversations (shared across measures) and Pipecat clips; a model's rank is
1 + the number of models whose lower bound is above its upper bound.
Usage: python -m stt_bench.elo sources.json out.json   (env: STT_ELO_JUDGMENTS, DRAWS)
"""
import json, math, os, random, re, sys, collections, statistics

# Semantic-WER counting below is vendored from pipecat-ai/stt-benchmark (commit 94f5672, BSD-2-Clause),
# so every model's judge output is counted with Pipecat's own rules.
_CONTRACTION_ENDINGS = ("n't", "'m", "'re", "'ve", "'ll", "'d")
_IS_CONTRACTIONS = {
    "it",
    "that",
    "what",
    "there",
    "here",
    "he",
    "she",
    "who",
    "where",
    "how",
    "when",
    "why",
    "let",
}


def count_reference_words(text: str) -> int:
    """Count the words in a reference transcription.

    WER divides by this count, so it is computed here rather than by the
    judge: every service is scored against the same count for the same
    reference. Punctuation is dropped, contractions count as the two words
    they expand to, and a hyphenated compound or a number as written
    ("2,000", "7:30") counts as one word.
    """
    count = 0
    for token in text.lower().replace("’", "'").split():
        token = token.strip('.,!?;:"()[]{}…')
        if not any(ch.isalnum() for ch in token):
            continue
        count += 1
        if token.endswith(_CONTRACTION_ENDINGS) or (
            token.endswith("'s") and token[:-2] in _IS_CONTRACTIONS
        ):
            count += 1
    return count


def error_weight(error: dict) -> int:
    """Number of errors one entry of the judge's error list counts for.

    The judge is asked to list errors word by word, except that a split,
    merged, or compound word is one substitution. An entry that spans more
    words than that still counts for each word it covers, so a deleted phrase
    listed as a single entry isn't scored as one error.
    """
    reference_words = count_reference_words(error.get("reference") or "")
    hypothesis_words = count_reference_words(error.get("hypothesis") or "")
    if error["type"] == "deletion":
        return max(reference_words, 1)
    if error["type"] == "insertion":
        return max(hypothesis_words, 1)
    if min(reference_words, hypothesis_words) <= 1:
        return 1
    return reference_words


JUDGMENT_DIRS = [d for d in os.environ.get('STT_ELO_JUDGMENTS', '').split(os.pathsep) if d]


WEIGHTS = {'ocular': .10, 'pipecat': .10, 'semantic': .10, 'phone': .12, 'callcenter': .10, 'babble': .08,
           'tv': .05, 'reverb': .05, 'short': .10, 'entities': .20}
SHORT, MARGIN = 5, 5.0
OUR_SIDE = {'invalid_pacing'}


def load(path, model=None):
    """clip -> (state, word_errors, reference); state in valid|failed|missing."""
    d = json.load(open(path)); out = {}
    if model and 'models' in d:  # unified Pipecat results (items with attempts)
        for it in d['models'][model]['items']:
            if it.get('cohort') not in (None, 'public'): continue
            a = it['attempts'][(it.get('selected_attempt') or len(it['attempts'])) - 1] if it['attempts'] else {}
            out[it['clip_id']] = row(a, a.get('word_errors'))
        return out
    for c in d['clips']:
        if c.get('condition') not in (None, 'public_anchor'): continue
        out[c['clip_id']] = r = row(c, c.get('word_errors'))
        if r[0] == 'valid': HYP.setdefault(CUR[0], {})[c['clip_id']] = c.get('transcript') or ''
    return out


def row(c, we):
    t = c.get('turn_timing') or {}
    if c.get('attempted') is False: return ('missing', None, '')  # never run: not a model failure
    if c.get('our_side_failure'): return ('missing', None, '')  # failed for our own reasons (flagged from raw events)
    if c.get('valid') and we: return ('valid', we, c.get('reference') or '')
    if t.get('ttft_status') in OUR_SIDE or (c.get('pacing') or {}).get('valid') is False: return ('missing', None, '')
    return ('failed', None, c.get('reference') or '')


# Module state shared by the loaders: the model being merged, each model's transcript per clip, the
# transcript of the winning (newest valid) row, and per-clip normalised references / entity counts.
CUR = [None]


def merge(paths, model=None):
    """Several runs of one model on one set: newest valid row wins; failed only if no run has it valid."""
    CUR[0] = model
    out = {}
    for p in paths:  # oldest first
        for cid, r in (load(*p) if isinstance(p, list) else load(p)).items():
            old = out.get(cid)
            if r[0] == 'valid' or old is None or old[0] == 'missing':
                out[cid] = r
                if r[0] == 'valid': WIN[(model, cid)] = HYP.get(model, {}).get(cid, '')
    return out


def semantic_rows(runs):
    """Semantic WER per clip from judge outputs (Pipecat's prompt; errors counted with Pipecat's functions, vendored above).
    Pairs whose normalised texts are identical score 0 errors without judging."""
    import glob, hashlib
    J = {}
    for f in sorted(p for d in JUDGMENT_DIRS for p in glob.glob(os.path.join(d, '*.jsonl'))):
        for l in open(f):
            try: r = json.loads(l); J[r['id']] = r['errors']
            except Exception: pass
    rows = collections.defaultdict(dict); wts = {}; missing = collections.Counter()
    spoken = {cid for data in runs.values() for cid, (st, we, ref) in data.items() if we and we['reference_words']}
    for m, data in runs.items():
        for cid, (st, we, ref) in data.items():
            if st == 'missing' or cid not in spoken: continue  # same clips as WER: a reference with words after normalising
            n = count_reference_words(ref) if ref else (we or {}).get('reference_words', 0)
            if n == 0: continue
            if st == 'failed': rows[cid][m] = None; wts[cid] = max(wts.get(cid, 0), n); continue
            if we['hypothesis_normalized'] == we['reference_normalized']: e = 0
            else:
                pid = hashlib.sha1((ref + '\x00' + WIN[(m, cid)]).encode()).hexdigest()[:16]
                if pid not in J: missing[m] += 1; continue
                e = sum(error_weight(x) for x in J[pid])
            rows[cid][m] = max(0.0, 100 - 100 * e / n); wts[cid] = max(wts.get(cid, 0), n)
    if missing: print('semantic: unjudged pairs skipped', dict(missing))
    return rows, wts


HYP = {}
REFN = {}
ENTN = {}
WIN = {}


def clip_score(we):
    return max(0.0, 100 - 100 * (we['substitutions'] + we['insertions'] + we['deletions']) / max(1, we['reference_words']))


def outcome(a, b, margin):
    if a is None and b is None: return None
    if a is None: return 0.0
    if b is None: return 1.0
    return 0.5 if abs(a - b) <= margin else float(a > b)


def ent_scoreable(ents, ref_norm):
    """Entities normalised with the benchmark's own normaliser; keep only those present in the normalised reference."""
    out = []
    for s in ents:
        if s and (' ' + s + ' ') in (' ' + ref_norm + ' '): out.append(s)
    return out


def ent_captured(ents, hyp_norm):
    toks = ' ' + hyp_norm + ' '
    return sum(1 for s in ents if ' ' + s + ' ' in toks or ' ' + s.replace(' ', '') + ' ' in toks)


def measure_rows(kind, runs, labels=None):
    """clip -> {model: score or None(failed)}, clip -> weight (words)."""
    rows = collections.defaultdict(dict); wts = {}
    W = {}  # one word count per clip, from any valid row (normalised reference), so every model sees the same clip set
    for data in runs.values():
        for cid, (st, we, ref) in data.items():
            if we: W.setdefault(cid, we['reference_words'])
    for m, data in runs.items():
        for cid, (st, we, ref) in data.items():
            if st == 'missing' or cid not in W: continue
            words = W[cid]
            if kind == 'short':
                if words >= SHORT or words == 0: continue
                rows[cid][m] = None if st == 'failed' else -(we['substitutions'] + we['insertions'] + we['deletions'])
            elif kind == 'entities':
                if not we and st == 'valid': continue
                ref_norm = we['reference_normalized'] if we else REFN.get(cid, '')
                ents = ent_scoreable([e['norm'] for e in labels.get(cid, {}).get('entities', [])], ref_norm) if ref_norm else []
                if we: REFN[cid] = ref_norm; ENTN[cid] = len(ents)
                if not ents: continue
                rows[cid][m] = None if st == 'failed' else ent_captured(ents, we['hypothesis_normalized'])
            else:
                if words == 0: continue
                rows[cid][m] = None if st == 'failed' else clip_score(we)
            wts[cid] = max(wts.get(cid, 0), words)
    return rows, wts


def matches(rows, wts, models, margin, clips):
    out = []
    for c in clips:
        r = rows.get(c, {})
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                if a in r and b in r:
                    o = outcome(r[a], r[b], margin)
                    if o is not None: out.append((a, b, o, wts[c]))
    return out


_CONVERSATION = re.compile(r'^(c\d+)-s\d+-seg\d+$')


def unit(clip_id):
    """Bootstrap unit: an Ocular turn (cNNN-sN-segNNNN) belongs to its conversation; any other clip is its own unit.
    Turns of one conversation share speakers and audio, so they are resampled together."""
    m = _CONVERSATION.match(str(clip_id))
    return m.group(1) if m else clip_id


def bootstrap_sample(M, rng):
    """One bootstrap draw. Conversations are drawn once with replacement and that draw is shared by every measure built
    on them; measures of independent clips (Pipecat) resample their clips."""
    groups = {k: collections.defaultdict(list) for k in M}
    for k in M:
        for c in M[k][0]: groups[k][unit(c)].append(c)
    conversations = sorted({u for k in M for u in groups[k] if _CONVERSATION.match(str(next(iter(groups[k][u]))))})
    drawn = [rng.choice(conversations) for _ in conversations]
    sample = {}
    for k in M:
        if any(u in groups[k] for u in conversations):
            sample[k] = [c for u in drawn for c in groups[k].get(u, [])]
        else:
            keys = list(M[k][0]); sample[k] = [rng.choice(keys) for _ in keys]
    return sample


def bt(weighted, models, iters=400):
    p = {m: 1.0 for m in models}; wins = collections.Counter(); pairs = collections.Counter()
    for a, b, s, w in weighted:
        wins[a] += w * s; wins[b] += w * (1 - s); pairs[(a, b)] += w
    for _ in range(iters):
        new = {}
        for m in models:
            den = sum(n / (p[a] + p[b]) for (a, b), n in pairs.items() if m in (a, b))
            new[m] = (wins[m] + 0.5) / (den + 1.0 / p[m])
        g = statistics.geometric_mean(new.values()); p = {m: v / g for m, v in new.items()}
    return {m: 1000 + 400 * math.log10(v) for m, v in p.items()}


def fit(M, models, weights, sample=None):
    allw = []
    for k, w in weights.items():
        rows, wts = M[k]
        clips = sample[k] if sample else list(rows)
        ms = matches(rows, wts, models, 0.0 if k in ('short', 'entities') else MARGIN, clips)
        tot = sum(x[3] for x in ms)
        if tot: allw += [(a, b, s, 1000 * w * cw / tot) for a, b, s, cw in ms]
    return bt(allw, models)


def wer_totals(runs):
    """model -> (word errors, reference words) over its valid clips: the plain WER shown in the output table.
    The matches use per-clip scores floored at 0; this total is not floored, so inserted words all count."""
    out = {}
    for m, data in runs.items():
        e = n = 0
        for st, we, ref in data.values():
            if st == 'valid' and we and we['reference_words']:
                e += we['substitutions'] + we['insertions'] + we['deletions']; n += we['reference_words']
        out[m] = (e, n)
    return out


def main(src_path, out_path, draws=int(os.environ.get("DRAWS", 200))):
    S = json.load(open(src_path)); labels = json.load(open(S['entity_labels']))['clips']
    models = S['models']; M = {}; coverage = {}; totals = {}
    for k in WEIGHTS:
        spec = S['measures'].get(k)
        if not spec: continue
        base = spec.get('from', k)
        runs = {m: merge(S['measures'][base]['runs'][m], m) for m in models if S['measures'][base]['runs'].get(m)}
        M[k] = semantic_rows(runs) if k == 'semantic' else measure_rows({'short': 'short', 'entities': 'entities'}.get(k, 'wer'), runs, labels)
        if k not in ('semantic', 'short', 'entities'): totals[k] = wer_totals(runs)
        coverage[k] = {m: sum(1 for r in M[k][0].values() if m in r) for m in models}
    weights = {k: v for k, v in WEIGHTS.items() if k in M}; t = sum(weights.values()); weights = {k: v / t for k, v in weights.items()}
    dropped = [m for m in models if not any(coverage[k].get(m) for k in M)]
    models = [m for m in models if m not in dropped]
    elo = fit(M, models, weights)
    rng = random.Random(42); boots = {m: [] for m in models}
    for _ in range(draws):
        r = fit(M, models, weights, bootstrap_sample(M, rng))
        for m in models: boots[m].append(r[m])
    ci = {m: (sorted(v)[int(.025 * draws)], sorted(v)[int(.975 * draws) - 1]) for m, v in boots.items()}
    rank = {m: 1 + sum(1 for o in models if o != m and ci[o][0] > ci[m][1]) for m in models}
    # per measure x model summary for the page, and per pair wins/draws/losses per measure
    table = {}
    for k, (rows, wts) in M.items():
        for m in models:
            vals = [(r[m], wts[c]) for c, r in rows.items() if m in r]
            ok = [(v, w) for v, w in vals if v is not None]
            t = table.setdefault(m, {})
            if k == 'short': t[k] = dict(errors=-sum(v for v, _ in ok), clips=len(vals), failed=len(vals) - len(ok))
            elif k == 'entities':
                tot = sum(ENTN.get(c, 0) for c, r in rows.items() if m in r)
                t[k] = dict(captured=sum(v for v, _ in ok), total=tot, pct=round(100 * sum(v for v, _ in ok) / tot, 2) if tot else None, failed=len(vals) - len(ok))
            elif k in totals:
                e, n = totals[k].get(m, (0, 0))
                t[k] = dict(wer=round(100 * e / n, 2) if n else None, clips=len(vals), failed=len(vals) - len(ok))
            else:
                sw = sum(w for _, w in ok)
                t[k] = dict(wer=round(100 - sum(v * w for v, w in ok) / sw, 2) if sw else None, clips=len(vals), failed=len(vals) - len(ok))
    pairs = {}
    for k, (rows, wts) in M.items():
        for a, b, o, w in matches(rows, wts, models, 0.0 if k in ('short', 'entities') else MARGIN, list(rows)):
            d = pairs.setdefault(f'{a}|{b}', {}).setdefault(k, [0, 0, 0])
            d[0 if o == 1 else 1 if o == 0.5 else 2] += 1
    json.dump(dict(weights=weights, dropped_no_data=dropped, coverage=coverage, table=table, pairs=pairs, elo=elo, ci95=ci, rank=rank), open(out_path, 'w'), indent=1)
    for m in sorted(models, key=lambda m: -elo[m]):
        print(f"{rank[m]:2d}  {m:44s} {elo[m]:6.0f}  [{ci[m][0]:.0f}-{ci[m][1]:.0f}]")
    print('weights', {k: round(v, 3) for k, v in weights.items()})


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
