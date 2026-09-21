"""Nova Sonic authenticated with a Bedrock API key instead of an access-key pair.

Pipecat's Nova Sonic service signs its Bedrock requests with SigV4, which needs
an access key id and a secret. Bedrock also accepts an API key presented as a
bearer token, and that is the credential a team is issued first -- so the only
thing standing between an issued key and a benchmark row would otherwise be a
different credential form.

The service builds its Bedrock client in one method, so that is the seam: the
client is constructed with a bearer auth scheme in place of the SigV4 one.
Nothing else changes. Same endpoint, same bidirectional stream, same event
grammar; only who signs the request is different.

Bedrock's own service model declares both schemes (``aws.auth#sigv4`` and
``smithy.api#httpBearerAuth``), so this is the documented second form rather
than a way around the first.

The signing itself is the library's. ``APIKeyAuthScheme`` already puts a token
in a header behind a scheme word, which is all bearer auth is -- there is no
canonicalisation, so nothing depends on the body or the header order. Only two
methods need replacing, because the stock ones read the token off the client
config and Bedrock's generated config has no field for one.

One caveat worth knowing before a run: the IAM policy behind an API key is
region-scoped and grants ``bedrock:CallWithBearerToken``. A key that works in
one region can be refused in another while the model itself is only served in
the second, and the two failures look alike from the outside.
"""

from __future__ import annotations

from typing import Any

from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient
from aws_sdk_bedrock_runtime.config import Config
from pipecat.services.aws.nova_sonic.llm import AWSNovaSonicLLMService
from smithy_core.auth import AuthOption
from smithy_core.shapes import ShapeID
from smithy_core.traits import APIKeyLocation
from smithy_http.aio.auth.apikey import APIKeyAuthScheme
from smithy_http.aio.identity.apikey import APIKeyIdentityResolver

BEARER_SCHEME_ID = ShapeID("smithy.api#httpBearerAuth")


class _BearerAuthScheme(APIKeyAuthScheme):
    """The library's API-key scheme, carrying a token supplied per process.

    The stock class looks the token up on the client config. Bedrock's generated
    config declares SigV4 credential fields and nothing else, so the lookup would
    fail; the token is held here instead and the two lookup methods return it.
    """

    scheme_id = BEARER_SCHEME_ID

    def __init__(self, token: str) -> None:
        super().__init__(name="Authorization", location=APIKeyLocation.HEADER, scheme="Bearer")
        self._token = token
        self._resolver = APIKeyIdentityResolver()

    def identity_properties(self, *, context: Any) -> dict[str, str]:
        return {"api_key": self._token}

    def identity_resolver(self, *, context: Any) -> APIKeyIdentityResolver:
        return self._resolver


class _BearerResolver:
    """Chooses bearer auth for every operation.

    Hand-written because the generated resolver only ever offers SigV4; there is
    no configuration that makes it offer the second scheme Bedrock declares.
    """

    def resolve_auth_scheme(self, auth_parameters: Any) -> list[AuthOption]:
        return [AuthOption(scheme_id=BEARER_SCHEME_ID, identity_properties={}, signer_properties={})]


class BearerTokenNovaSonic(AWSNovaSonicLLMService):
    """Nova Sonic, authenticated with a Bedrock API key.

    The base class takes an access key id and a secret as required arguments.
    They are unused here, so placeholders are passed and the client that would
    have consumed them is replaced. ``create_client`` is the only place the
    service builds one, including on the session-continuation path that keeps a
    long call alive, so the override covers the whole call rather than its start.
    """

    def __init__(self, *, token: str, region: str, **kwargs: Any) -> None:
        super().__init__(
            access_key_id="unused-with-bearer-auth",
            secret_access_key="unused-with-bearer-auth",
            region=region,
            **kwargs,
        )
        self._bearer_token = token

    def create_client(self) -> BedrockRuntimeClient:
        """The same client the base class builds, with the auth scheme swapped."""
        return BedrockRuntimeClient(
            config=Config(
                endpoint_uri=f"https://bedrock-runtime.{self._region}.amazonaws.com",
                region=self._region,
                auth_scheme_resolver=_BearerResolver(),
                auth_schemes={BEARER_SCHEME_ID: _BearerAuthScheme(self._bearer_token)},
            )
        )
