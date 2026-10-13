"""Authored TurnBench datasets: scripted two-party conversations, rendered with
TTS and placed on an exact timeline, so the turn-taking labels are known by
construction rather than annotated after the fact.

Pipeline (one module each, in order):

    spec      a versioned JSON conversation script (schema in ``spec.py``)
    render    text + voice -> mono audio; ``MockRenderer`` (offline, deterministic)
              or ``ElevenLabsRenderer`` (the service bench's TTS path); disk cache
    bounds    speech segments inside a rendered clip, incl. the pauses that
              ``<break time="1.5s"/>`` markers produce
    compose   both speakers on one 48 kHz clock at exact sample offsets; barge-ins
              cut the agent, backchannels do not; yields the ground-truth events
    labels    Sesame annotation tracks (a/b/c identical, script-derived) and the
              gold events the script *plants*, for checking against ``turnbench.gold``
    package   the local dataset directory ``turnbench.data`` loads: one parquet
              shard (FLAC bytes, annotation columns, metadata), durations,
              ground truth, and a hashed manifest
    check     planted-vs-derived gold table and the oracle round trip through
              Sesame's scorer
    listen    listening pack: a stereo WAV (caller left, agent right) and a cue
              sheet of every labelled event per conversation, to check by ear

Usage::

    turn-bench author build --specs datasets/authored/specs --out datasets/authored/build/mock \\
        [--renderer mock|elevenlabs] [--transform clean|noise-20db|telephone|...]
    turn-bench author check --dataset datasets/authored/build/mock
    turn-bench author listen --dataset datasets/authored/build/mock
    turn-bench run --model rms_vad --dataset datasets/authored/build/mock

The ElevenLabs renderer needs ``ELEVENLABS_API_KEY`` in the environment or in
the gitignored ``.env`` next to this package's project. Breaks are what the
script asks for; the labels follow the pauses *detected* in the render, and
``build`` prints scripted vs detected length per break. The
degradation transforms, the onset detector's frame/threshold constants and the
ElevenLabs request all come from the service bench (``service/``) through
``service_bridge``; nothing is copied.
"""
