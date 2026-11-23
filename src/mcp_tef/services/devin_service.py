"""Structured output from a Devin session, through the Devin v3 API.

The ``devin`` model provider has no completion endpoint: each request creates
one Devin session with a structured-output schema, then polls it until the
session returns output that validates against that schema.
"""

import asyncio
import time
from typing import Any

import httpx
import structlog
from pydantic import BaseModel, ValidationError

from mcp_tef.api.errors import LLMProviderAPIKeyError, LLMProviderError
from mcp_tef.config.settings import Settings

logger = structlog.get_logger(__name__)

PROVIDER = "devin"

# Session statuses after which the session produces no more output.
_FINAL_STATUSES = {"exit", "error", "suspended"}


def structured_output_schema[T: BaseModel](output_type: type[T]) -> dict[str, Any]:
    """Return output_type's JSON Schema as Draft 7, which the Devin API takes."""
    schema = output_type.model_json_schema(ref_template="#/definitions/{model}")
    if "$defs" in schema:
        schema["definitions"] = schema.pop("$defs")
    return schema


class DevinSessionService:
    """Creates a Devin session per request and returns its structured output."""

    def __init__(
        self,
        api_key: str,
        org_id: str,
        settings: Settings,
        devin_mode: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._api_key = api_key
        self._org_id = org_id
        self._settings = settings
        self._devin_mode = devin_mode
        self._transport = transport

    @classmethod
    def from_request(
        cls, api_key: str | None, model: str, settings: Settings | None
    ) -> "DevinSessionService":
        """Resolve the PAT and organization the way the other providers resolve keys.

        The PAT comes from the request (X-Model-API-Key), then settings
        (DEVIN_API_KEY); the organization from settings (DEVIN_ORG_ID).
        model is the session's devin_mode, or "default" for the org's default.
        """
        settings = settings or Settings()
        api_key = api_key or settings.devin_api_key
        if not api_key:
            raise LLMProviderAPIKeyError(
                PROVIDER,
                "Devin PAT required. Provide via X-Model-API-Key header, set "
                "DEVIN_API_KEY environment variable, or configure in settings.",
            )
        org_id = settings.devin_org_id
        if not org_id:
            raise LLMProviderError(PROVIDER, "DEVIN_ORG_ID is not set")
        devin_mode = None if model == "default" else model
        return cls(api_key, org_id, settings, devin_mode=devin_mode)

    async def run[T: BaseModel](self, title: str, prompt: str, output_type: type[T]) -> T:
        """Run prompt in a new session and return its output as output_type."""
        body: dict[str, Any] = {
            "title": title,
            "prompt": prompt,
            "structured_output_schema": structured_output_schema(output_type),
            "structured_output_required": True,
            "repos": [],
            "resumable": False,
            "memory_enabled": False,
            "tags": ["mcp-tef"],
        }
        if self._devin_mode:
            body["devin_mode"] = self._devin_mode
        sessions = f"/v3/organizations/{self._org_id}/sessions"
        async with httpx.AsyncClient(
            base_url=self._settings.devin_api_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=self._settings.devin_request_timeout,
            transport=self._transport,
        ) as client:
            created = self._json(await client.post(sessions, json=body))
            session_id = created["session_id"]
            logger.info("Created Devin session", session_id=session_id, url=created.get("url"))
            deadline = time.monotonic() + self._settings.devin_session_timeout
            while True:
                session = self._json(await client.get(f"{sessions}/{session_id}"))
                output = session.get("structured_output")
                if output is not None:
                    try:
                        return output_type.model_validate(output)
                    except ValidationError:
                        logger.debug("Ignoring incomplete structured output", session_id=session_id)
                status = session.get("status")
                if status in _FINAL_STATUSES:
                    raise LLMProviderError(
                        PROVIDER,
                        f"session {session_id} ended ({status}) without valid structured output",
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LLMProviderError(
                        PROVIDER,
                        f"session {session_id} returned no structured output within "
                        f"{self._settings.devin_session_timeout:g}s",
                    )
                await asyncio.sleep(min(self._settings.devin_poll_interval, remaining))

    @staticmethod
    def _json(response: httpx.Response) -> dict[str, Any]:
        if response.status_code in (401, 403):
            raise LLMProviderAPIKeyError(
                PROVIDER, f"Devin API refused the PAT ({response.status_code})"
            )
        if response.is_error:
            raise LLMProviderError(
                PROVIDER, f"Devin API returned {response.status_code}: {response.text[:500]}"
            )
        return response.json()
