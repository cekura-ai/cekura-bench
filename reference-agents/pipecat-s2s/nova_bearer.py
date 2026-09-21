"""Nova Sonic authenticated with a Bedrock API key instead of an access-key pair.

Pipecat's Nova Sonic service signs its Bedrock requests with SigV4, which needs
an access key id and a secret. Bedrock also accepts an API key as a bearer
token, and that is the credential a team is issued first -- so the only thing
standing between an issued key and a benchmark row would otherwise be a
different credential form.

The service builds its Bedrock client in one method, so that is the seam: the
client is constructed with a bearer auth scheme in place of the SigV4 one.
Nothing else changes. Same endpoint, same bidirectional stream, same event
grammar; only who signs the request is different.

Bedrock's own service model declares both schemes (``aws.auth#sigv4`` and
``smithy.api#httpBearerAuth``), so this is the documented second form rather
than a way around the first.

One caveat worth knowing before a run: the IAM policy behind an API key is
region-scoped and grants ``bedrock:CallWithBearerToken``. A key that works in
one region can be refused in another while the model itself is only served in
the second, and the two failures look alike from the outside.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService

BEARER_SCHEME = "smithy.api#httpBearerAuth"


@dataclass(frozen=True)
class _Token:
    """A bearer token in the shape the signer expects of an identity."""

    token: str
    expiration: Any = None


class _StaticTokenResolver:
    """Resolves to one token. The key is supplied per process, not fetched."""

    def __init__(self, token: str) -> None:
        self._token = token

    async def get_identity(self, *, properties: Any) -> _Token:
        return _Token(self._token)


class _BearerSigner:
    """Sets the Authorization header. There is no request canonicalisation here.

    That is the whole difference from SigV4: a bearer token is presented, not
    used to sign the request's contents, so nothing depends on the body or the
    ordering of headers.
    """

    async def sign(self, *, request: Any, identity: _Token, properties: Any) -> Any:
        from smithy_http import Field

        request.fields.set_field(Field(name="Authorization", values=[f"Bearer {identity.token}"]))
        return request


class _BearerAuthScheme:
    def __init__(self, token: str) -> None:
        from smithy_core.shapes import ShapeID

        self.scheme_id = ShapeID(BEARER_SCHEME)
        self._resolver = _StaticTokenResolver(token)

    def identity_properties(self, *, context: Any) -> dict[str, Any]:
        return {}

    def identity_resolver(self, *, context: Any) -> _StaticTokenResolver:
        return self._resolver

    def signer_properties(self, *, context: Any) -> dict[str, Any]:
        return {}

    def signer(self) -> _BearerSigner:
        return _BearerSigner()

    def event_signer(self, *, request: Any) -> None:
        # SigV4 signs each event of the stream; a bearer token authenticates the
        # connection and the events ride it unsigned. Returning None is what
        # tells the client not to look for a per-event signer.
        return None


class _BearerResolver:
    """Chooses bearer auth for every operation, in place of the generated resolver."""

    def resolve_auth_scheme(self, auth_parameters: Any) -> list[Any]:
        from smithy_core.auth import AuthOption
        from smithy_core.shapes import ShapeID

        return [AuthOption(scheme_id=ShapeID(BEARER_SCHEME), identity_properties={}, signer_properties={})]


class BearerTokenNovaSonic(AWSNovaSonicLLMService):
    """Nova Sonic, authenticated with a Bedrock API key.

    The base class takes an access key id and a secret as required arguments.
    They are unused here, so placeholders are passed and the client that would
    have consumed them is replaced.
    """

    def __init__(self, *, token: str, region: str, **kwargs: Any) -> None:
        super().__init__(
            access_key_id="unused-with-bearer-auth",
            secret_access_key="unused-with-bearer-auth",
            region=region,
            **kwargs,
        )
        self._bearer_token = token

    def create_client(self) -> Any:
        """The same client the base class builds, with the auth scheme swapped."""
        from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
        from aws_sdk_bedrock_runtime.config import Config
        from smithy_core.shapes import ShapeID

        scheme = _BearerAuthScheme(self._bearer_token)
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self._region}.amazonaws.com",
            region=self._region,
            auth_scheme_resolver=_BearerResolver(),
            auth_schemes={ShapeID(BEARER_SCHEME): scheme},
        )
        return BedrockRuntimeClient(config=config)
