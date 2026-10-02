"""Separate observed wire events from local harness completion; never subtract a tail."""

VERSION = 'receipt-events-v1'


def observations(events, config, assessment, *, cutoff=float('inf')):
    events = [e for e in events if e['time_seconds'] <= cutoff]
    ends = [e['time_seconds'] for e in events if e['kind'] == 'speech_end']
    t0 = ends[0] if len(ends) == 1 else None
    def delta(at):
        return (at-t0)*1000 if at is not None and t0 is not None else None
    def last(kind):
        return next((e['time_seconds'] for e in reversed(events) if e['kind'] == kind), None)
    messages = [e for e in events if e['kind'] == 'provider_message']
    text_at = terminal_at = None
    if config['provider'] == 'gradium':
        text_at = next((e['time_seconds'] for e in reversed(messages)
                        if e['message'].get('type') == 'text' and e['message'].get('text', '').strip()), None)
        terminal_at = next((e['time_seconds'] for e in reversed(messages)
                            if e['message'].get('type') == 'end_of_stream'), None)
    valid = assessment.get('valid', assessment.get('transcript_complete', False))
    status = 'measured' if valid and t0 is not None else 'invalid_or_incomplete'
    return dict(version=VERSION, status=status,
                last_text_arrival_ms=delta(text_at) if status == 'measured' else None,
                final_text_arrival_ms=delta(assessment.get('final_transcript_received_seconds')) if status == 'measured' else None,
                finalization_ack_ms=assessment.get('finalize_latency_ms') if status == 'measured' else None,
                server_completion_ms=delta(terminal_at) if status == 'measured' else None,
                harness_completion_ms=delta(last('provider_terminal')) if status == 'measured' else None,
                audio_send_complete_ms=delta(last('audio_complete')),
                transmitted_silence_frames=sum(e['kind'] == 'audio_sent' and e.get('phase') == 'silence' for e in events),
                definition='Receipt timestamps relative to actual speech-end send completion. Harness completion is not a provider signal.')
