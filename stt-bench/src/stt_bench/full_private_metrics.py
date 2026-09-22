"""Receipt-time final-word delay, with alignment ambiguity explicitly excluded."""
from collections import Counter
import math
import jiwer
import numpy as np
from .score import NORMALIZER, word_errors, aggregate_wer

def matches(reference, hypothesis):
    if not reference or not hypothesis: return {}
    result=jiwer.process_words(' '.join(reference),' '.join(hypothesis))
    return {i:j for c in result.alignments[0] if c.type=='equal'
        for i,j in zip(range(c.ref_start_idx,c.ref_end_idx),range(c.hyp_start_idx,c.hyp_end_idx))}

def latency(clip, snapshots, frames, valid):
    result={'status':'unavailable','median_ms':None,'p95_ms':None,'measured_words':0,
            'reference_words':0,'exclusions':{},'words':[],
            'definition':'Correct, unambiguously aligned finalized word receipt minus actual source-word-end frame delivery; first attempt only. Stable prefix required.'}
    ref=[]; ends=[]
    for w in clip['words']:
        tokens=NORMALIZER(w['text']).split()
        ref.extend(tokens);ends.extend([w['end'] if w['timing_valid'] else None]*len(tokens))
    # Contextual normalization can merge numbers across source word annotations.
    # Keep unambiguous word times and exclude only tokens whose time cannot map.
    canonical=NORMALIZER(clip['reference']).split()
    mapping=matches(canonical,ref)
    rev={len(canonical)-1-i:len(ref)-1-j for i,j in matches(canonical[::-1],ref[::-1]).items()}
    ends=[ends[mapping[i]] if i in mapping and rev.get(i)==mapping[i] else None for i in range(len(canonical))]
    ref=canonical;result['reference_words']=len(ref)
    if not valid or not snapshots:
        result['exclusions']={'invalid_first_attempt':len(ref)};return result
    final=NORMALIZER(snapshots[-1]['text']).split()
    times=[None]*len(final)
    for s in snapshots:
        tokens=NORMALIZER(s['text']).split();common=0
        for a,b in zip(tokens,final):
            if a!=b:break
            common+=1
        for i in range(len(final)):
            if i>=common:times[i]=None
            elif times[i] is None:times[i]=s['time_seconds']
    forward=matches(ref,final)
    reverse={len(ref)-1-i:len(final)-1-j for i,j in matches(ref[::-1],final[::-1]).items()}
    exclusions=Counter();delays=[]
    for i,word in enumerate(ref):
        j=forward.get(i)
        reason=None
        if ends[i] is None:reason='invalid_reference_timing'
        elif j is None:reason='word_incorrect_or_missing'
        elif reverse.get(i)!=j:reason='ambiguous_alignment'
        elif times[j] is None:reason='final_receipt_unavailable'
        frame=math.ceil((ends[i] or 0)*50)-1
        if not reason and frame not in frames:reason='word_end_frame_unavailable'
        if not reason:
            delay=(times[j]-frames[frame])*1000
            if delay < 0:reason='text_precedes_reference_word_end'
        if reason:exclusions[reason]+=1
        else:
            delays.append(delay);result['words'].append({'reference_index':i,'word':word,'delay_ms':delay})
    result.update(status='measured' if delays else 'unavailable',measured_words=len(delays),exclusions=dict(exclusions))
    if delays:result.update(median_ms=float(np.median(delays)),p95_ms=float(np.percentile(delays,95)))
    return result
