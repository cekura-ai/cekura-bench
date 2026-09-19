"""Placing the call and receiving its media, so the deterministic caller can speak on a phone.

The caller side of Lane B is the same authored audio Lane A sends, put on a real
phone call instead of a websocket to a provider. Twilio places an outbound call
to the agent's number; the TwiML it fetches connects the call's media to a socket
we run; from then on the adapter in ``lane_b.adapters.twilio_stream`` has the
same shape as any other realtime adapter and the rest of the harness does not
know the difference.

Two endpoints, because they answer different questions:

* ``/media`` -- a benchmark call. One connection is handed to whoever is waiting
  for it, and the harness drives it.
* ``/loopback`` -- returns every inbound frame immediately. Nothing reasons at
  the far end, so what a chirp measures across it is the carrier and the codec
  and nothing else. This is what ``lane_b.tone`` needs and the reason Lane B can
  state a transport cost rather than assume one.

**The socket must be reachable by the carrier**, which means a public hostname
and TLS. That is a deployment property, not a benchmark one, and it is the only
part of Lane B that cannot be exercised from a laptop.

No credential is read at import time and none is ever written into a record. The
account identifier and token are arguments, taken from the environment by the
runner that calls this.
"""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

from aiohttp import BasicAuth, ClientSession, web

from lane_b.adapters.twilio_stream import WebsocketMediaStream

TWILIO_API = "https://api.twilio.com/2010-04-01"


def connect_stream_twiml(stream_url: str) -> str:
    """TwiML connecting the call's media, both directions, to our socket.

    ``<Connect>`` rather than ``<Start>``: ``<Start>`` forks a copy of the audio
    for listening, which cannot speak back into the call. The deterministic
    caller has to be heard, so the call's media has to be handed over rather
    than copied.
    """
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        f'<Response><Connect><Stream url="{stream_url}"/></Connect></Response>'
    )


@dataclass
class StreamServer:
    """Accepts the carrier's media socket and hands it to whoever is waiting."""

    host: str = "0.0.0.0"
    port: int = 8080
    _pending: asyncio.Queue = field(default_factory=asyncio.Queue, init=False)
    _runner: web.AppRunner | None = field(default=None, init=False)

    async def start(self) -> None:
        app = web.Application()
        app.add_routes([
            web.get("/media", self._media),
            web.get("/loopback", self._loopback),
            web.post("/twiml", self._twiml),
        ])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        await web.TCPSite(self._runner, self.host, self.port).start()

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def accept(self, timeout_s: float = 60.0) -> WebsocketMediaStream:
        """The next media socket the carrier opens."""
        return await asyncio.wait_for(self._pending.get(), timeout=timeout_s)

    async def _twiml(self, request: web.Request) -> web.Response:
        """Served to the carrier when it dials, pointing it back at ``/media``."""
        target = request.query.get("stream") or ""
        return web.Response(text=connect_stream_twiml(target), content_type="text/xml")

    async def _media(self, request: web.Request) -> web.WebSocketResponse:
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        done: asyncio.Future = asyncio.get_running_loop().create_future()
        await self._pending.put(_HandedOver(socket, done))
        await done                                   # hold the handler open while it is in use
        return socket

    async def _loopback(self, request: web.Request) -> web.WebSocketResponse:
        """Return each inbound frame immediately, adding nothing but our own hop.

        Deliberately the dumbest possible far end. Anything that buffered,
        resampled or re-encoded here would be measured as carrier latency and
        would inflate every transport correction taken from it.
        """
        socket = web.WebSocketResponse(heartbeat=20)
        await socket.prepare(request)
        stream_sid: str | None = None
        async for message in socket:
            if message.type is not web.WSMsgType.TEXT:
                continue
            payload = json.loads(message.data)
            event = payload.get("event")
            if event == "start":
                stream_sid = payload.get("start", {}).get("streamSid")
            elif event == "media" and stream_sid:
                await socket.send_str(json.dumps({
                    "event": "media",
                    "streamSid": stream_sid,
                    "media": {"payload": payload["media"]["payload"]},
                }))
            elif event == "stop":
                break
        return socket


class _HandedOver(WebsocketMediaStream):
    """A media socket whose handler stays alive until the harness is finished."""

    def __init__(self, socket: Any, done: asyncio.Future) -> None:
        super().__init__(socket)
        self._done = done

    async def close(self) -> None:
        if not self._done.done():
            self._done.set_result(None)
        await super().close()


async def place_call(
    *, account_sid: str, auth_token: str, to_number: str, from_number: str,
    twiml: str, status_callback: str | None = None, timeout_s: float = 30.0,
) -> str:
    """Dial ``to_number`` and connect it to the media socket in ``twiml``.

    Returns the carrier's call identifier, which is the only handle a later
    lookup has -- so it goes into the cell record, while the credentials that
    created it never do.
    """
    form = {"To": to_number, "From": from_number, "Twiml": twiml}
    if status_callback:
        form["StatusCallback"] = status_callback
    url = f"{TWILIO_API}/Accounts/{quote(account_sid)}/Calls.json"
    async with ClientSession(auth=BasicAuth(account_sid, auth_token)) as session:
        async with session.post(url, data=form, timeout=timeout_s) as response:
            body = await response.json()
            if response.status >= 300:
                raise RuntimeError(f"carrier refused the call: {response.status} {body.get('message')}")
            return body["sid"]


async def hang_up(*, account_sid: str, auth_token: str, call_sid: str) -> None:
    url = f"{TWILIO_API}/Accounts/{quote(account_sid)}/Calls/{quote(call_sid)}.json"
    async with ClientSession(auth=BasicAuth(account_sid, auth_token)) as session:
        await session.post(url, data={"Status": "completed"})


def silence_frame(ms: float = 20.0) -> str:
    """A μ-law silence payload, for keeping a stream alive without speaking."""
    return base64.b64encode(b"\xff" * int(8 * ms)).decode("ascii")
