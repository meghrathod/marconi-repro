import json
import os

def create_panel(id, title, target_expr, gridPos, format="short"):
    return {
        "id": id,
        "title": title,
        "type": "timeseries",
        "gridPos": gridPos,
        "targets": [
            {
                "expr": target_expr,
                "legendFormat": "{{trace_file}}",
                "refId": "A"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "linear",
                    "showPoints": "auto"
                },
                "unit": format
            }
        }
    }

def create_sgl_panel(id, title, target_expr, gridPos, format="short"):
    return {
        "id": id,
        "title": title,
        "type": "timeseries",
        "gridPos": gridPos,
        "targets": [
            {
                "expr": target_expr,
                "legendFormat": "SGLang",
                "refId": "A"
            }
        ],
        "fieldConfig": {
            "defaults": {
                "custom": {
                    "drawStyle": "line",
                    "lineInterpolation": "linear",
                    "showPoints": "auto"
                },
                "unit": format
            }
        }
    }

panels = [
    create_panel(1, "TTFT (P50)", "marconi_ttft_ms", {"h": 8, "w": 8, "x": 0, "y": 0}, "ms"),
    create_panel(2, "Total Latency (P50)", "marconi_total_latency_ms", {"h": 8, "w": 8, "x": 8, "y": 0}, "ms"),
    create_panel(3, "Output Throughput", "marconi_output_throughput_tok_s", {"h": 8, "w": 8, "x": 16, "y": 0}, "op/s"),
    create_panel(4, "Token Cache Hit Rate", "marconi_token_hit_rate_pct", {"h": 8, "w": 8, "x": 0, "y": 8}, "percent"),
    create_panel(5, "Total Cached Tokens", "marconi_total_cached_tokens", {"h": 8, "w": 8, "x": 8, "y": 8}, "short"),
    create_panel(6, "Total Prompt Tokens", "marconi_total_prompt_tokens", {"h": 8, "w": 8, "x": 16, "y": 8}, "short"),
    
    create_sgl_panel(7, "SGLang Cache Hit Rate", "sglang:cache_hit_rate{job=\"sglang\"}", {"h": 8, "w": 12, "x": 0, "y": 16}, "percentunit"),
    create_sgl_panel(8, "SGLang Prefill Tokens / sec", "rate(sglang:prompt_tokens_total{job=\"sglang\"}[1m])", {"h": 8, "w": 12, "x": 12, "y": 16}, "op/s"),
]


dashboard = {
    "__inputs": [],
    "__elements": {},
    "__requires": [],
    "annotations": {
        "list": []
    },
    "editable": True,
    "fiscalYearStartMonth": 0,
    "graphTooltip": 0,
    "id": None,
    "links": [],
    "panels": panels,
    "refresh": "5s",
    "schemaVersion": 39,
    "tags": [],
    "templating": {
        "list": []
    },
    "time": {
        "from": "now-15m",
        "to": "now"
    },
    "timepicker": {},
    "timezone": "",
    "title": "Marconi Metrics (SGLang + Trace Replayer)",
    "uid": "marconi_repro",
    "version": 1
}

os.makedirs('grafana/dashboards', exist_ok=True)
with open('grafana/dashboards/marconi.json', 'w') as f:
    json.dump(dashboard, f, indent=2)

print("Dashboard JSON generated successfully.")
