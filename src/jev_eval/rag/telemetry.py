"""OpenTelemetry span export to local immutable JSON records."""

from pathlib import Path

from jev_eval.io import write_json

_configured = False


def setup_tracing(root: str | Path):
    global _configured
    if _configured:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            SimpleSpanProcessor,
            SpanExporter,
            SpanExportResult,
        )
    except ImportError:
        return
    root = Path(root)

    class FileExporter(SpanExporter):
        def export(self, spans):
            for span in spans:
                context = span.get_span_context()
                write_json(root / "spans" / f"{context.trace_id:032x}-{context.span_id:016x}.json",
                           {"name": span.name, "trace_id": f"{context.trace_id:032x}",
                            "span_id": f"{context.span_id:016x}",
                            "start_time_unix_ns": span.start_time,
                            "end_time_unix_ns": span.end_time,
                            "attributes": dict(span.attributes or {})})
            return SpanExportResult.SUCCESS

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(FileExporter()))
    trace.set_tracer_provider(provider)
    _configured = True
