import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional, cast

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as GrpcOTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as HttpOTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Tracer
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from core.ops.base_trace_instance import BaseTraceInstance
from core.ops.entities.config_entity import ArizePhoenixConfig
from core.ops.entities.trace_entity import (
    BaseTraceInfo,
    DatasetRetrievalTraceInfo,
    GenerateNameTraceInfo,
    MessageTraceInfo,
    ModerationTraceInfo,
    SuggestedQuestionTraceInfo,
    ToolTraceInfo,
    TraceTaskName,
    WorkflowTraceInfo,
)
from extensions.ext_database import db
from models.model import EndUser, MessageFile
from models.workflow import WorkflowNodeExecution

logger = logging.getLogger(__name__)


def setup_tracer(arize_phoenix_config: ArizePhoenixConfig) -> Tracer:
    """Configure OpenTelemetry tracer with OTLP exporter for Phoenix"""
    endpoint = arize_phoenix_config.host.rstrip('/')  
    
    # Set headers for authentication
    headers = {"api_key": arize_phoenix_config.api_key, "space_key": arize_phoenix_config.space_key}
        
    try:
        # Choose the appropriate exporter based on protocol
        if arize_phoenix_config.protocol == "grpc":
            exporter = GrpcOTLPSpanExporter(
                endpoint=endpoint,
                headers=headers,
                timeout=30 
            )
        else:
            exporter = HttpOTLPSpanExporter(
                endpoint=endpoint,
                headers=headers,
                timeout=30  
            )
        
        resource = Resource(attributes={
            "openinference.project.name": arize_phoenix_config.project
        })
        provider = trace_sdk.TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(exporter)
        provider.add_span_processor(processor)
        
        # Create a named tracer instead of setting the global provider
        tracer_name = f"arize_phoenix_{arize_phoenix_config.project}"
        return trace.get_tracer(tracer_name, tracer_provider=provider)
    except Exception as e:
        logger.error(f"Failed to setup Arize Phoenix tracer: {str(e)}", exc_info=True)
        raise


def datetime_to_nanos(dt: datetime) -> int:
    """Convert datetime to nanoseconds since epoch"""
    return int(dt.timestamp() * 1_000_000_000)


class ArizePhoenixDataTrace(BaseTraceInstance):
    def __init__(
        self,
        arize_phoenix_config: ArizePhoenixConfig,
    ):
        super().__init__(arize_phoenix_config)
        self.arize_phoenix_config = arize_phoenix_config
        self.tracer = setup_tracer(arize_phoenix_config)
        self.project = arize_phoenix_config.project
        self.file_base_url = os.getenv("FILES_URL", "http://127.0.0.1:5001")

    def trace(self, trace_info: BaseTraceInfo):
        logger.info(f"Arize Phoenix trace: {trace_info}")
        try:
            if isinstance(trace_info, WorkflowTraceInfo):
                self.workflow_trace(trace_info)
            if isinstance(trace_info, MessageTraceInfo):
                self.message_trace(trace_info)
            if isinstance(trace_info, ModerationTraceInfo):
                self.moderation_trace(trace_info)
            if isinstance(trace_info, SuggestedQuestionTraceInfo):
                self.suggested_question_trace(trace_info)
            if isinstance(trace_info, DatasetRetrievalTraceInfo):
                self.dataset_retrieval_trace(trace_info)
            if isinstance(trace_info, ToolTraceInfo):
                self.tool_trace(trace_info)
            if isinstance(trace_info, GenerateNameTraceInfo):
                self.generate_name_trace(trace_info)
        except Exception as e:
            logger.error(f"Error in Arize Phoenix trace: {str(e)}", exc_info=True)
            raise

    def workflow_trace(self, trace_info: WorkflowTraceInfo):
        workflow_metadata = {
            "workflow_id": trace_info.workflow_run_id,
            "message_id": trace_info.message_id,
            "workflow_app_log_id": trace_info.workflow_app_log_id,
            "start_time": trace_info.start_time.isoformat(),
            "end_time": trace_info.end_time.isoformat(),
        }
        workflow_metadata.update(trace_info.metadata)

        with self.tracer.start_as_current_span(
            name=TraceTaskName.WORKFLOW_TRACE.value,
            attributes={
                SpanAttributes.INPUT_VALUE: json.dumps(trace_info.workflow_run_inputs, ensure_ascii=False),
                SpanAttributes.OUTPUT_VALUE: json.dumps(trace_info.workflow_run_outputs, ensure_ascii=False),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
                SpanAttributes.METADATA: json.dumps(workflow_metadata)
            }
        ) as workflow_span:
            # Process workflow nodes
            for node_execution in self._get_workflow_nodes(trace_info.workflow_run_id):
                created_at = node_execution.created_at or datetime.now()
                elapsed_time = node_execution.elapsed_time
                finished_at = created_at + timedelta(seconds=elapsed_time)
                
                process_data = json.loads(node_execution.process_data) if node_execution.process_data else {}
                
                node_metadata = {
                    "node_id": node_execution.id,
                    "node_type": node_execution.node_type,
                    "node_status": node_execution.status,
                    "start_time": created_at.isoformat(),
                    "end_time": finished_at.isoformat(),
                    "tenant_id": node_execution.tenant_id,
                    "app_id": node_execution.app_id,
                    "app_name": node_execution.title,
                }

                # Add execution metadata
                if node_execution.execution_metadata:
                    node_metadata.update(json.loads(node_execution.execution_metadata))

                # Determine the correct span kind based on node type
                span_kind = OpenInferenceSpanKindValues.CHAIN.value
                if node_execution.node_type == "llm":
                    span_kind = OpenInferenceSpanKindValues.LLM.value
                elif node_execution.node_type == "dataset_retrieval":
                    span_kind = OpenInferenceSpanKindValues.RETRIEVER.value
                elif node_execution.node_type in ["tool", "moderation"]:
                    span_kind = OpenInferenceSpanKindValues.TOOL.value
                
                with self.tracer.start_span(
                    name=node_execution.node_type,
                    attributes={
                        SpanAttributes.INPUT_VALUE: node_execution.inputs or "{}",
                        SpanAttributes.OUTPUT_VALUE: node_execution.outputs or "{}",
                        SpanAttributes.OPENINFERENCE_SPAN_KIND: span_kind,
                        SpanAttributes.METADATA: json.dumps(node_metadata)
                    },
                    parent=workflow_span
                ) as node_span:
                    if node_execution.node_type == "llm":
                        provider = process_data.get("model_provider")
                        model = process_data.get("model_name")
                        if provider:
                            node_span.set_attribute(SpanAttributes.LLM_PROVIDER, provider)
                        if model:
                            node_span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model)
                        
                        usage = json.loads(node_execution.outputs).get("usage", {}) if node_execution.outputs else {}
                        if usage:
                            node_span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, usage.get("total_tokens", 0))
                            node_span.set_attribute(
                                SpanAttributes.LLM_TOKEN_COUNT_PROMPT, usage.get("prompt_tokens", 0))
                            node_span.set_attribute(
                                SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, usage.get("completion_tokens", 0))

    def message_trace(self, trace_info: MessageTraceInfo):
        if trace_info.message_data is None:
            return

        file_list = cast(list[str], trace_info.file_list) or []
        message_file_data: Optional[MessageFile] = trace_info.message_file_data

        if message_file_data is not None:
            file_url = f"{self.file_base_url}/{message_file_data.url}" if message_file_data else ""
            file_list.append(file_url)

        message_metadata = {
            "message_id": trace_info.message_id,
            "conversation_mode": str(trace_info.conversation_mode),
            "user_id": trace_info.message_data.from_account_id,
            "file_list": file_list,
        }
        message_metadata.update(trace_info.metadata)

        # Add end user data if available
        if trace_info.message_data.from_end_user_id:
            end_user_data: Optional[EndUser] = (
                db.session.query(EndUser)
                .filter(EndUser.id == trace_info.message_data.from_end_user_id)
                .first()
            )
            if end_user_data is not None:
                message_metadata["end_user_id"] = end_user_data.session_id

        attributes = {
            SpanAttributes.INPUT_VALUE: trace_info.message_data.query,
            SpanAttributes.OUTPUT_VALUE: trace_info.message_data.answer,
            SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.CHAIN.value,
            SpanAttributes.METADATA: json.dumps(message_metadata),
        }

        # Only add attributes if they are not None
        if trace_info.total_tokens is not None:
            attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = trace_info.total_tokens
        if trace_info.message_tokens is not None:
            attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = trace_info.message_tokens
        if trace_info.answer_tokens is not None:
            attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = trace_info.answer_tokens
        if trace_info.message_data.model_id is not None:
            attributes[SpanAttributes.LLM_MODEL_NAME] = trace_info.message_data.model_id
        if trace_info.message_data.model_provider is not None:
            attributes[SpanAttributes.LLM_PROVIDER] = trace_info.message_data.model_provider
        
        attributes["start_time"] = trace_info.start_time.isoformat()
        attributes["end_time"] = trace_info.end_time.isoformat()

        with self.tracer.start_as_current_span(
            name=TraceTaskName.MESSAGE_TRACE.value,
            attributes=attributes,
        ) as message_span:
            if trace_info.error:
                message_span.add_event(
                    "exception",
                    attributes={
                        "exception.message": trace_info.error,
                        "exception.type": "Error",
                        "exception.stacktrace": trace_info.error
                    }
                )
                        
            llm_attributes = {
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
                SpanAttributes.OUTPUT_VALUE: trace_info.message_data.answer,
                SpanAttributes.METADATA: json.dumps(message_metadata),
            }
            
            # Add input messages with indexed attributes
            if isinstance(trace_info.inputs, list):
                for i, msg in enumerate(trace_info.inputs):
                    if isinstance(msg, dict):
                        llm_attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.message.content"] = msg.get("text", "")
                        llm_attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.{i}.message.role"] = msg.get(
                            "role", "user")
            elif isinstance(trace_info.inputs, dict):
                # If inputs is a dict, treat it as a single message
                llm_attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.content"] = json.dumps(trace_info.inputs)
                llm_attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role"] = "user"
            elif isinstance(trace_info.inputs, str):
                # If inputs is a string, treat it as a single message
                llm_attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.content"] = trace_info.inputs
                llm_attributes[f"{SpanAttributes.LLM_INPUT_MESSAGES}.0.message.role"] = "user"

            # Only add open inference LLM attributes if they are not None
            if trace_info.total_tokens is not None:
                llm_attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = trace_info.total_tokens
            if trace_info.message_tokens is not None:
                llm_attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = trace_info.message_tokens
            if trace_info.answer_tokens is not None:
                llm_attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = trace_info.answer_tokens
            if trace_info.message_data.model_id is not None:
                llm_attributes[SpanAttributes.LLM_MODEL_NAME] = trace_info.message_data.model_id
            if trace_info.message_data.model_provider is not None:
                llm_attributes[SpanAttributes.LLM_PROVIDER] = trace_info.message_data.model_provider
            
            
            if trace_info.message_data and trace_info.message_data.message_metadata:
                metadata_dict = json.loads(trace_info.message_data.message_metadata)
                if model_params := metadata_dict.get("model_parameters"):
                    llm_attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = json.dumps(model_params)

            llm_attributes["start_time"] = trace_info.start_time.isoformat()
            llm_attributes["end_time"] = trace_info.end_time.isoformat()

            with self.tracer.start_span(
                name="llm",
                attributes=llm_attributes,
                context=trace.set_span_in_context(message_span),
            ) as llm_span:
                if trace_info.error:
                    llm_span.add_event(
                        "exception",
                        attributes={
                            "exception.message": trace_info.error,
                            "exception.type": "Error",
                            "exception.stacktrace": trace_info.error
                        }
                    )

    def moderation_trace(self, trace_info: ModerationTraceInfo):
        if trace_info.message_data is None:
            return

        metadata = {
            "message_id": trace_info.message_id,
            "tool_name": "moderation"
        }
        metadata.update(trace_info.metadata)

        with self.tracer.start_as_current_span(
            name=TraceTaskName.MODERATION_TRACE.value,
            attributes={
                SpanAttributes.INPUT_VALUE: json.dumps(trace_info.inputs),
                SpanAttributes.OUTPUT_VALUE: json.dumps({
                    "action": trace_info.action,
                    "flagged": trace_info.flagged,
                    "preset_response": trace_info.preset_response,
                    "inputs": trace_info.inputs,
                }),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.METADATA: json.dumps(metadata),
                "start_time": trace_info.start_time.isoformat(),
                "end_time": trace_info.end_time.isoformat(),
            },
        ) as span:
            if trace_info.error:
                span.add_event(
                    "exception",
                    attributes={
                        "exception.message": trace_info.error,
                        "exception.type": "Error",
                        "exception.stacktrace": trace_info.error
                    }
                )

    def suggested_question_trace(self, trace_info: SuggestedQuestionTraceInfo):
        if trace_info.message_data is None:
            return

        start_time = trace_info.start_time or trace_info.message_data.created_at
        end_time = trace_info.end_time or trace_info.message_data.updated_at

        metadata = {
            "message_id": trace_info.message_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "tool_name": "suggested_question"
        }
        metadata.update(trace_info.metadata)

        with self.tracer.start_as_current_span(
            name=TraceTaskName.SUGGESTED_QUESTION_TRACE.value,
            attributes={
                SpanAttributes.INPUT_VALUE: json.dumps(trace_info.inputs),
                SpanAttributes.OUTPUT_VALUE: json.dumps(trace_info.suggested_question),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.METADATA: json.dumps(metadata),
                "start_time": trace_info.start_time.isoformat(),
                "end_time": trace_info.end_time.isoformat(),
            },
        ):
            pass

    def dataset_retrieval_trace(self, trace_info: DatasetRetrievalTraceInfo):
        if trace_info.message_data is None:
            return

        start_time = trace_info.start_time or trace_info.message_data.created_at
        end_time = trace_info.end_time or trace_info.message_data.updated_at

        metadata = {
            "message_id": trace_info.message_id,
            "tool_name": "dataset_retrieval"
        }
        metadata.update(trace_info.metadata)

        with self.tracer.start_as_current_span(
            name=TraceTaskName.DATASET_RETRIEVAL_TRACE.value,
            attributes={
                SpanAttributes.INPUT_VALUE: json.dumps(trace_info.inputs, ensure_ascii=False),
                SpanAttributes.OUTPUT_VALUE: json.dumps({"documents": trace_info.documents}, ensure_ascii=False),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.RETRIEVER.value,
                SpanAttributes.METADATA: json.dumps(metadata),
            },
            start_time=datetime_to_nanos(start_time),
            end_time=datetime_to_nanos(end_time),
        ):
            pass

    def tool_trace(self, trace_info: ToolTraceInfo):
        metadata = {
            "message_id": trace_info.message_id,
            "tool_name": trace_info.tool_name
        }
        metadata.update(trace_info.metadata)

        with self.tracer.start_as_current_span(
            name=trace_info.tool_name,
            attributes={
                SpanAttributes.INPUT_VALUE: json.dumps(trace_info.tool_inputs, ensure_ascii=False),
                SpanAttributes.OUTPUT_VALUE: json.dumps(trace_info.tool_outputs, ensure_ascii=False),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.METADATA: json.dumps(metadata, ensure_ascii=False),
                "start_time": trace_info.start_time.isoformat(),
                "end_time": trace_info.end_time.isoformat(),
            },
        ) as span:
            if trace_info.error:
                span.add_event(
                    "exception",
                    attributes={
                        "exception.message": trace_info.error,
                        "exception.type": "Error",
                        "exception.stacktrace": trace_info.error
                    }
                )

    def generate_name_trace(self, trace_info: GenerateNameTraceInfo):
        metadata = {
            "project_name": self.project,
            "message_id": trace_info.message_id,
        }
        metadata.update(trace_info.metadata)

        with self.tracer.start_as_current_span(
            name=TraceTaskName.GENERATE_NAME_TRACE.value,
            attributes={
                SpanAttributes.INPUT_VALUE: json.dumps(trace_info.inputs, ensure_ascii=False),
                SpanAttributes.OUTPUT_VALUE: json.dumps(trace_info.outputs, ensure_ascii=False),
                SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.TOOL.value,
                SpanAttributes.METADATA: json.dumps(metadata),
                "start_time": trace_info.start_time.isoformat(),
                "end_time": trace_info.end_time.isoformat(),
            },
        ):
            pass

    def api_check(self):
        try:
            # Create a test span to verify connection
            with self.tracer.start_span("api_check") as span:
                span.set_attribute("test", "true")
            return True
        except Exception as e:
            logger.info(f"Arize Phoenix API check failed: {str(e)}", exc_info=True)
            raise ValueError(f"Arize Phoenix API check failed: {str(e)}")

    def get_project_url(self):
        try:
            return f"{self.arize_phoenix_config.host}/projects/{self.project}"
        except Exception as e:
            logger.info(f"Arize Phoenix get run url failed: {str(e)}", exc_info=True)
            raise ValueError(f"Arize Phoenix get run url failed: {str(e)}")

    def _get_workflow_nodes(self, workflow_run_id: str):
        """Helper method to get workflow nodes"""
        workflow_nodes = (
            db.session.query(
                WorkflowNodeExecution.id,
                WorkflowNodeExecution.tenant_id,
                WorkflowNodeExecution.app_id,
                WorkflowNodeExecution.title,
                WorkflowNodeExecution.node_type,
                WorkflowNodeExecution.status,
                WorkflowNodeExecution.inputs,
                WorkflowNodeExecution.outputs,
                WorkflowNodeExecution.created_at,
                WorkflowNodeExecution.elapsed_time,
                WorkflowNodeExecution.process_data,
                WorkflowNodeExecution.execution_metadata,
            )
            .filter(WorkflowNodeExecution.workflow_run_id == workflow_run_id)
            .all()
        )
        return workflow_nodes
