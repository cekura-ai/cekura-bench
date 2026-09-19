"""Turn timing v1, reconstructed offline from original receipt/send events."""
import math
from copy import deepcopy

from .providers import is_nova, transcript_at

MEASUREMENT = 'private-turns-controlled-v1'


def transcript_timeline(events, config):
    """Replay semantic events, excluding high-volume audio-send evidence.

    A same-range Nova final revision replaces its earlier final in this new
    contract. Historical Nova replay remains untouched. Partial-range overlaps
    still fail closed through the shared reconstruction checks.
    """
    if not is_nova(config) and config['provider']!='google':
        # These adapters all use the same stateful replay engine. Feeding each
        # event once is equivalent to replaying every prefix from scratch.
        from . import providers as adapters
        from .provider_protocol import Protocol, ProviderError
        factory=(adapters.gemini_live.Protocol if config['model'] in adapters.gemini_live.MODELS else
                 adapters.speechmatics_agent.Protocol if config['model']=='linden-1' else
                 adapters.assembly_adapter(config).Protocol if config['provider']=='assemblyai' else
                 adapters.reson8.Protocol if config['provider']=='reson8' else
                 adapters.gradium.Protocol if config['provider']=='gradium' else
                 adapters.trial_providers.Protocol if config['provider'] in adapters.trial_providers.MODELS else Protocol)
        protocol=factory(config);snapshots=[]
        for e in sorted(events,key=lambda e:e['time_seconds']):
            if e['kind']=='finalize_requested':protocol.requested=True
            elif e['kind']=='close_stream_requested':protocol.closing=True
            elif e['kind']=='provider_message':
                try:protocol.feed(e['message'])
                except ProviderError:pass
            if e['kind'] in ('provider_message','provider_terminal'):
                snapshots.append(dict(time_seconds=e['time_seconds'],**deepcopy(protocol.snapshot())))
        return snapshots
    semantic, snapshots, final_ranges = [], [], {}
    for e in sorted(events, key=lambda e: e['time_seconds']):
        if e['kind'] == 'audio_sent':
            continue
        message = e.get('message', {})
        if is_nova(config) and e['kind'] == 'provider_message' and message.get('type') == 'Results' and message.get('is_final'):
            text = (message.get('channel', {}).get('alternatives') or [{}])[0].get('transcript', '')
            if text:
                key = (tuple(message.get('channel_index', [])), message.get('start', 0), message.get('duration', 0))
                previous = final_ranges.get(key)
                if previous is not None:
                    semantic.remove(previous)
                final_ranges[key] = e
        semantic.append(e)
        if e['kind'] not in ('provider_message', 'provider_terminal'):
            continue
        state = transcript_at(semantic, float('inf'), config)
        snapshots.append(dict(time_seconds=e['time_seconds'], **state))
    return snapshots


def measure(events, config, assessment, *, dry_run=False):
    snapshots = transcript_timeline(events, config)
    sends = [e for e in events if e['kind']=='audio_sent']
    start = min((e['time_seconds'] for e in sends), default=None)
    ends = [e for e in events if e['kind']=='speech_end']
    end = ends[0]['time_seconds'] if len(ends)==1 else None
    first = next((s for s in snapshots if s.get('text','').strip()), None)
    last_final, previous = None, ''
    for s in snapshots:
        final = s.get('final_text', '')
        if final != previous:
            last_final = s['time_seconds']
            previous = final
    final_state = snapshots[-1] if snapshots else {}
    controlled = config.get('turn_finalization_class') == 'controlled'
    invalid = ('dry_run' if dry_run else 'invalid_pacing' if not assessment['pacing']['valid'] else
               'invalid_attempt' if not assessment['valid'] else
               final_state.get('reconstruction_status') if final_state.get('reconstruction_status','supported') != 'supported' else
               'incomplete_final_transcript' if final_state.get('partial_text') else None)
    ttft_status = invalid or ('missing_audio_start' if start is None else 'no_text' if first is None else 'measured')
    ttfs_status = invalid or ('missing_speech_end' if end is None else 'no_final_text' if not previous or last_final is None else 'measured')
    if ttft_status == 'measured' and (not math.isfinite(first['time_seconds']-start) or first['time_seconds'] < start):
        ttft_status='text_before_audio_start'
    signed = (last_final-end)*1000 if ttfs_status == 'measured' else None
    if signed is not None and not math.isfinite(signed):
        ttfs_status='invalid_timestamp'; signed=None
    return dict(measurement_profile=MEASUREMENT, audio_start_seconds=start, speech_end_seconds=end,
        first_text_received_seconds=first['time_seconds'] if first else None,
        first_text_kind=('partial' if first.get('partial_text') else 'final') if first else None,
        first_text=first.get('text') if first else None,
        ttft_ms=(first['time_seconds']-start)*1000 if ttft_status=='measured' else None, ttft_status=ttft_status,
        final_text_received_seconds=last_final, final_text=previous,
        ttfs_signed_ms=signed, ttfs_ms=max(0.,signed) if signed is not None and controlled else None,
        observed_speech_end_to_final_ms=max(0.,signed) if signed is not None else None,
        final_text_before_boundary=signed is not None and signed<0,
        ttfs_status='unsupported_controlled_finalization' if ttfs_status=='measured' and not controlled else ttfs_status,
        finalization_class=config.get('turn_finalization_class'),
        reconstruction_status=final_state.get('reconstruction_status','supported'),
        provisional_final=bool(final_state.get('partial_text')))
