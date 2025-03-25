from enum import Enum
from typing import ClassVar, Literal

from pydantic import BaseModel, ValidationInfo, field_validator


class TracingProviderEnum(Enum):
    LANGFUSE = "langfuse"
    LANGSMITH = "langsmith"
    OPIK = "opik"
    ARIZE_PHOENIX = "arizePhoenix"


class BaseTracingConfig(BaseModel):
    """
    Base model class for tracing
    """

    ...


class LangfuseConfig(BaseTracingConfig):
    """
    Model class for Langfuse tracing config.
    """

    public_key: str
    secret_key: str
    host: str = "https://api.langfuse.com"

    @field_validator("host")
    @classmethod
    def set_value(cls, v, info: ValidationInfo):
        if v is None or v == "":
            v = "https://api.langfuse.com"
        if not v.startswith("https://") and not v.startswith("http://"):
            raise ValueError("host must start with https:// or http://")

        return v


class LangSmithConfig(BaseTracingConfig):
    """
    Model class for Langsmith tracing config.
    """

    api_key: str
    project: str
    endpoint: str = "https://api.smith.langchain.com"

    @field_validator("endpoint")
    @classmethod
    def set_value(cls, v, info: ValidationInfo):
        if v is None or v == "":
            v = "https://api.smith.langchain.com"
        if not v.startswith("https://"):
            raise ValueError("endpoint must start with https://")

        return v


class OpikConfig(BaseTracingConfig):
    """
    Model class for Opik tracing config.
    """

    api_key: str | None = None
    project: str | None = None
    workspace: str | None = None
    url: str = "https://www.comet.com/opik/api/"

    @field_validator("project")
    @classmethod
    def project_validator(cls, v, info: ValidationInfo):
        if v is None or v == "":
            v = "Default Project"

        return v

    @field_validator("url")
    @classmethod
    def url_validator(cls, v, info: ValidationInfo):
        if v is None or v == "":
            v = "https://www.comet.com/opik/api/"
        if not v.startswith(("https://", "http://")):
            raise ValueError("url must start with https:// or http://")
        if not v.endswith("/api/"):
            raise ValueError("url should ends with /api/")

        return v
    

class ArizePhoenixConfig(BaseTracingConfig):
    """
    Model class for Arize Phoenix tracing config.
    """

    DEFAULT_ENDPOINT: ClassVar[str] = "https://app.phoenix.arize.com/v1/traces"

    api_key: str | None = None
    project: str | None = None
    protocol: Literal["grpc", "http"] = "grpc"
    host: str = DEFAULT_ENDPOINT

    @field_validator("project")
    @classmethod
    def project_validator(cls, v, info: ValidationInfo):
        if v is None or v == "":
            v = "Default Project"

        return v

    @field_validator("host")
    @classmethod
    def host_validator(cls, v, info: ValidationInfo):
        if v is None or v == "":
            v = cls.DEFAULT_HOST
        if not v.startswith(("https://", "http://")):
            raise ValueError("host must start with https:// or http://")
        # Remove any trailing slashes for consistent handling
        v = v.rstrip('/')
        return v

    @field_validator("protocol")
    @classmethod
    def protocol_validator(cls, v):
        if v not in ["grpc", "http"]:
            raise ValueError("protocol must be either 'grpc' or 'http'")
        return v



OPS_FILE_PATH = "ops_trace/"
OPS_TRACE_FAILED_KEY = "FAILED_OPS_TRACE"
