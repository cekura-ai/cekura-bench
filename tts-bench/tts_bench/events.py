"""Event names for the TTS bench. The log itself is ``tts_bench.common.events.EventLog``."""

SESSION_OPEN = "session.open"            # connected and warm; nothing below this line is timed
SESSION_CLOSED = "session.closed"
SESSION_ERROR = "session.error"

CONTEXT_OPEN = "tts.context.open"        # a synthesis context exists; no text yet
TEXT_SENT = "tts.text"                   # a text frame left us (chars, cumulative chars)
INPUT_DONE = "tts.input.done"            # we told the provider the text is complete
AUDIO_FIRST = "tts.audio.first"          # first non-empty audio chunk arrived
AUDIO_DONE = "tts.audio.done"            # provider signalled the context's audio is complete
CANCEL_SENT = "tts.cancel"               # we asked the provider to stop this context
CANCEL_ACK = "tts.cancel.ack"            # the provider acknowledged the stop
PROVIDER_META = "tts.provider.meta"      # anything else the provider said that is not audio
PROVIDER_ERROR = "tts.provider.error"
