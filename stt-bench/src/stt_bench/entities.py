"""Conservative English entity-value rules, separate from transcript normalization.

Only expressions understood by these rules receive a value score. Extraction is a
bounded grammar, not an exhaustive semantic judge. Original character spans remain.
"""
import re
from collections import Counter
from decimal import Decimal

VERSION = 'english-values-v1'
SMALL = dict(zip('zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen'.split(), range(20)))
TENS = dict(zip('twenty thirty forty fifty sixty seventy eighty ninety'.split(), range(20, 100, 10)))
ORDINAL = dict(zip('first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth sixteenth seventeenth eighteenth nineteenth'.split(), range(1, 20)))
ORDINAL.update({'twentieth': 20, 'thirtieth': 30})
NUM_WORDS = '|'.join([*SMALL, *TENS, *ORDINAL, 'hundred', 'thousand', 'million', 'billion', 'point', 'minus', 'negative'])
NUM = rf'(?:-?\d+(?:,\d{{3}})*(?:\.\d+)?(?:st|nd|rd|th)?|(?:a[ -]+)?(?:{NUM_WORDS})(?:[ -]+(?:{NUM_WORDS}|and))*)'
MONTH_NAMES = 'january february march april may june july august september october november december'.split()
MONTH = '|'.join(MONTH_NAMES)
CURRENCIES = {'$': 'USD', 'dollar': 'USD', 'dollars': 'USD', 'usd': 'USD',
              '€': 'EUR', 'euro': 'EUR', 'euros': 'EUR', 'eur': 'EUR',
              '£': 'GBP', 'pound': 'GBP', 'pounds': 'GBP', 'british pound': 'GBP', 'british pounds': 'GBP', 'gbp': 'GBP'}
CURRENCY = '|'.join(sorted((re.escape(x) for x in CURRENCIES), key=len, reverse=True))
KINDS = ('number', 'identifier', 'date', 'phone', 'email', 'amount', 'spelled_sequence')


def number_value(text):
    s = text.lower().strip()
    if re.fullmatch(r'a (hundred|thousand|million|billion)', s):
        s = 'one ' + s[2:]
    s = re.sub(r'(?<=\d)(st|nd|rd|th)$', '', s)
    if re.fullmatch(r'-?\d+(?:,\d{3})*(?:\.\d+)?', s):
        digits = s.replace(',', '')
        if re.fullmatch(r'0\d+', digits):
            return 'digits:' + digits  # Never erase potentially significant zeros.
        return str(Decimal(digits).normalize())
    tokens = s.replace('-', ' ').split()
    if not tokens:
        return None
    sign = 1
    if tokens[0] in ('minus', 'negative'):
        sign, tokens = -1, tokens[1:]
    if not tokens:
        return None
    if 'point' in tokens:
        if tokens.count('point') != 1:
            return None
        i = tokens.index('point')
        left = number_value(' '.join(tokens[:i]))
        right = tokens[i + 1:]
        if left is None or left.startswith('digits:') or not right or any(w not in SMALL or SMALL[w] > 9 for w in right):
            return None
        fraction = Decimal('0.' + ''.join(str(SMALL[w]) for w in right))
        return str(((Decimal(left) + fraction) * sign).normalize())
    total, current, previous = 0, 0, None
    scales = {'thousand': 1000, 'million': 1000000, 'billion': 1000000000}
    for token in tokens:
        if token == 'and':
            if previous not in ('hundred', *scales):
                return None
            previous = token
            continue
        if token in SMALL or token in TENS or token in ORDINAL:
            value = (SMALL | TENS | ORDINAL)[token]
            if previous in SMALL or previous in ORDINAL or (previous in TENS and value >= 10):
                return None
            current += value
        elif token == 'hundred':
            if not 1 <= current <= 9:
                return None
            current *= 100
        elif token in scales:
            if not current:
                return None
            total += current * scales[token]
            current = 0
        else:
            return None
        previous = token
    if previous == 'and':
        return None
    return str(Decimal(sign * (total + current)).normalize())


def canonical(kind, text):
    s = text.strip()
    if kind == 'number':
        return number_value(s)
    if kind == 'amount':
        match = re.fullmatch(rf'({CURRENCY})\s*({NUM})|({NUM})\s*({CURRENCY})', s, re.I)
        if not match:
            return None
        currency, value = (match[1], match[2]) if match[1] else (match[4], match[3])
        n = number_value(value)
        return f'{CURRENCIES[currency.lower()]}:{n}' if n is not None and not n.startswith('digits:') else None
    if kind == 'date':
        match = re.fullmatch(rf'({MONTH})\s+({NUM})(?:,?\s+(\d{{4}}))?', s, re.I)
        if not match:
            return None
        day = number_value(match[2])
        if day is None or day.startswith('digits:'):
            return None
        n = Decimal(day)
        # Month + year is distinct from a month + day.
        if not match[3] and n == int(n) and 1000 <= n <= 9999:
            return f'{int(n):04d}-{MONTH_NAMES.index(match[1].lower())+1:02d}'
        if n != int(n) or not 1 <= n <= 31:
            return None
        return f'{match[3] or "????"}-{MONTH_NAMES.index(match[1].lower())+1:02d}-{int(n):02d}'
    if kind == 'phone':
        if not re.fullmatch(r'\+?[\d\s().-]+', s):
            return None
        digits = re.sub(r'\D', '', s)
        return digits if 7 <= len(digits) <= 15 else None
    if kind == 'email':
        if not re.fullmatch(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', s):
            return None
        local, domain = s.rsplit('@', 1)
        return local + '@' + domain.lower()  # Local-part case is deliberately preserved.
    if kind == 'identifier':
        if re.fullmatch(r'[A-Za-z0-9]+(?:[ -][A-Za-z0-9]+)*', s):
            return re.sub(r'[ -]', '', s).upper()
        return None
    if kind == 'spelled_sequence':
        if re.fullmatch(r'[A-Za-z](?:[ -]+[A-Za-z])+', s):
            return re.sub(r'[ -]', '', s).upper()
    return None


def extract(text):
    """Extract raw occurrences, with specific types taking precedence over numbers."""
    found = []

    def add(kind, pattern, group=0):
        for match in re.finditer(pattern, text, re.I):
            start, end = match.span(group)
            if any(start < e['end'] and end > e['start'] for e in found):
                continue
            value = canonical(kind, text[start:end])
            if value is not None:
                found.append(dict(type=kind, text=text[start:end], start=start, end=end, value=value))
    add('email', r'(?<![\w.+-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?!\w)')
    add('date', rf'\b(?:{MONTH})\s+{NUM}(?:,?\s+\d{{4}})?\b')
    add('amount', rf'(?<!\w)(?:{CURRENCY})\s*{NUM}\b|\b{NUM}\s*(?:{CURRENCY})(?!\w)')
    add('phone', r'\b(?:phone|telephone|call|mobile)(?:\s+number)?(?:\s+is)?\s*[:#]?\s*(\+?\d[\d ().-]{5,}\d)', 1)
    add('phone', r'(?<!\w)(?:\+\d[\d ().-]{6,}\d|\(?\d{3}\)?[ -]\d{3}[ -]\d{4})(?!\w)')
    add('identifier', r'\b(?:account|identifier|id|code)(?:\s+(?:number|id))?(?:\s+is)?\s*[:#]?\s*([A-Za-z]*\d+[A-Za-z0-9]*)(?!\w)', 1)
    add('spelled_sequence', r'\b[A-Za-z](?:[ -]+[A-Za-z])+\b')
    add('number', rf'(?<![\w.]){NUM}(?!\w|\.\d)')
    return sorted(found, key=lambda e: e['start'])


def align_values(expected, predicted):
    """Ordered occurrence alignment; substitutions, deletions and insertions separate."""
    n, m = len(expected), len(predicted)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j-1] + (expected[i-1] != predicted[j-1]), dp[i-1][j] + 1, dp[i][j-1] + 1)
    counts = Counter(correct=0, wrong=0, missing=0, spurious=0)
    i, j = n, m
    while i or j:
        if i and j and dp[i][j] == dp[i-1][j-1] + (expected[i-1] != predicted[j-1]):
            counts['correct' if expected[i-1] == predicted[j-1] else 'wrong'] += 1
            i, j = i-1, j-1
        elif i and dp[i][j] == dp[i-1][j] + 1:
            counts['missing'] += 1
            i -= 1
        else:
            counts['spurious'] += 1
            j -= 1
    return dict(counts)


def value_errors(reference, hypothesis, entities):
    if entities is None:
        return dict(status='not_annotated', version=VERSION, by_type={}, unsupported=[])
    predicted = extract(hypothesis)
    reference_extracted = extract(reference)
    supported, unsupported = [], []
    previous_end = 0
    for e in entities:
        if (e['type'] not in KINDS or not 0 <= e['start'] < e['end'] <= len(reference)
                or e['start'] < previous_end or reference[e['start']:e['end']] != e['text']):
            raise ValueError('Invalid value entity annotation')
        previous_end = e['end']
        v = canonical(e['type'], e['text'])
        (supported if v is not None else unsupported).append({**e, 'value': v})
    by_type = {}
    for kind in KINDS:
        refs = [e for e in supported if e['type'] == kind]
        hyps = [e for e in predicted if e['type'] == kind]
        # Do not interpret parser omissions/unannotated reference occurrences as hallucinations.
        extracted = [e for e in reference_extracted if e['type'] == kind]
        inventory_matches = [(e['start'], e['end'], e['value']) for e in refs] == [(e['start'], e['end'], e['value']) for e in extracted]
        unsupported_kind = any(e['type'] == kind for e in unsupported)
        eligible = inventory_matches and not unsupported_kind
        counts = align_values([e['value'] for e in refs], [e['value'] for e in hyps]) if eligible else None
        by_type[kind] = dict(status='measured' if eligible else 'unsupported_reference_inventory',
                             reference_entities=len(refs) if eligible else 0,
                             predicted_entities=len(hyps), counts=counts)
    return dict(status='bounded_grammar', version=VERSION, by_type=by_type,
                unsupported=unsupported, predicted=predicted,
                limitation='English bounded grammar; unmatched predictions are spurious candidates, not exhaustive hallucination detection.')


def aggregate_values(items):
    by_type = {}
    for kind in KINDS:
        groups = [x['by_type'][kind] for x in items if kind in x.get('by_type', {})]
        eligible = [g for g in groups if g['counts'] is not None]
        counts = {k: sum(g['counts'][k] for g in eligible) for k in ('correct', 'wrong', 'missing', 'spurious')}
        n = sum(g['reference_entities'] for g in eligible)
        by_type[kind] = dict(**counts, reference_entities=n,
                             error_rate=(counts['wrong'] + counts['missing']) / n if n else None,
                             scored_clips=len(eligible), unsupported_clips=len(groups)-len(eligible))
    return dict(version=VERSION, by_type=by_type)
