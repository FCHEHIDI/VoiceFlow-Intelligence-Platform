"""
Load secrets from environment (development) or AWS Secrets Manager (production).
Never logs secret values — only key names and ARNs.
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Optional

import boto3
from botocore.exceptions import ClientError


class SecretsLoadError(Exception):
    """A required secret could not be loaded."""


class SecretsLoader:
    """
    - development / test: read from process environment
    - production: use boto3 to resolve AWS Secrets Manager ARNs
    Caches GetSecretValue responses for 300s per ARN.
    """

    _aws_cache: dict[str, tuple[float, str]] = {}
    _cache_ttl_s = 300.0

    def __init__(self) -> None:
        self._aws_client: Any = None

    def _get_client(self) -> Any:
        if self._aws_client is None:
            self._aws_client = boto3.client("secretsmanager")
        return self._aws_client

    @staticmethod
    def _is_production() -> bool:
        return (os.environ.get("ENV") or os.environ.get("env") or "development").lower() == "production"

    def get_aws_secret_string(self, secret_arn: str) -> str:
        """Return SecretString for an ARN, with in-memory TTL cache."""
        if not secret_arn or not str(secret_arn).strip():
            raise SecretsLoadError("Invalid empty secret ARN")

        now = time.monotonic()
        if secret_arn in self._aws_cache:
            exp, value = self._aws_cache[secret_arn]
            if now < exp:
                return value

        if not self._is_production():
            raise SecretsLoadError("AWS secret ARN is only supported when ENV=production")

        try:
            client = self._get_client()
            resp = client.get_secret_value(SecretId=secret_arn)
        except ClientError as e:
            raise SecretsLoadError(f"Failed to load secret (ARN metadata only: {secret_arn[:48]}...): {e}") from e

        s = resp.get("SecretString")
        if s is None:
            raise SecretsLoadError(f"Secret at ARN has no SecretString: {secret_arn[:48]}...")
        if not s.strip():
            raise SecretsLoadError("Secret string is empty")

        self._aws_cache[secret_arn] = (now + self._cache_ttl_s, s)
        return s

    def get_env(self, *names: str) -> str:
        """Load first non-empty value from the given environment variable names."""
        for n in names:
            v = os.environ.get(n)
            if v is not None and str(v).strip() != "":
                return v
        raise SecretsLoadError(
            f"None of the required environment variables are set: {', '.join(names)}"
        )


@lru_cache(maxsize=1)
def get_secrets_loader() -> SecretsLoader:
    """Singleton compatible with FastAPI Depends."""
    return SecretsLoader()
