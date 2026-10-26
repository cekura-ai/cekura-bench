"""TTS component bench: a text-to-speech service measured at its own interface.

Text goes in, timestamped audio chunks come out, and everything below that line
-- transport, trunk, codec, telephony platform -- is out of the benchmark. The
unit measured is "provider model under configuration X"; a number here is a
property of the service's API, and is recomputable from the audio and the
per-chunk arrival times every cell writes to disk.
"""

METHODOLOGY_VERSION = "tts/0.1"
