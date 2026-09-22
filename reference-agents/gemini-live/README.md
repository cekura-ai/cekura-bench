# Gemini Live reference harness

This is the Gemini Live Cloud Run harness used for the benchmark. It bridges Twilio Media Streams to Gemini Live, selects the frozen workflow by called number, dispatches fixture-backed tools, and publishes the normalized transcript to Cekura.

Build with `docker build -t gemini-live-reference .` or install `requirements.txt` and run `uvicorn app:app --host 0.0.0.0 --port 8080`. Configure Google, Twilio, public-service, and Cekura environment variables before deployment. Credentials and generated deployment artifacts are excluded.
