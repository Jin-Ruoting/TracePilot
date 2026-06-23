PYTHON ?= python
MODULE_PATH ?= src
TRACE_DIR ?= traces

.PHONY: help check check-json compile test tool-contracts eval-report promotion-artifacts review-workbench demo reference-catalog service-interface service-runtime service-api service-mcp portfolio-index serve-portfolio clean-generated

help:
	@echo "TracePilot commands"
	@echo "  make check                Run the full local quality gate"
	@echo "  make check-json           Run the quality gate and write JSON"
	@echo "  make test                 Run unit tests"
	@echo "  make compile              Compile Python sources"
	@echo "  make tool-contracts       Validate tool contracts"
	@echo "  make eval-report          Regenerate eval coverage reports"
	@echo "  make promotion-artifacts  Regenerate promotion trace/packets"
	@echo "  make review-workbench     Regenerate the approval review workbench"
	@echo "  make demo                 Regenerate portfolio demo JSON"
	@echo "  make reference-catalog    Regenerate GitHub/source reference catalog"
	@echo "  make service-interface    Regenerate MCP/FastAPI-style service interface reports"
	@echo "  make service-runtime      Regenerate session-backed service runtime demo"
	@echo "  make service-api          Regenerate optional FastAPI adapter readiness report"
	@echo "  make service-mcp          Regenerate optional MCP adapter readiness report"
	@echo "  make portfolio-index      Regenerate traces/index.html"
	@echo "  make serve-portfolio      Serve portfolio on localhost:8765"

check:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.quality_gate \
		--out $(TRACE_DIR)/quality-gate-report.md

check-json:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.quality_gate \
		--json \
		--out $(TRACE_DIR)/quality-gate-report.json

compile:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m compileall -q src

test:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m unittest discover -s tests

tool-contracts:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.tool_contracts --validate

eval-report:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.eval_report \
		--out $(TRACE_DIR)/eval-coverage-report.md
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.eval_report \
		--json \
		--out $(TRACE_DIR)/eval-coverage-report.json

promotion-artifacts:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.cli \
		"Should we try retrieval-aware conditioning for promoter expression prediction?" \
		--experiment-log examples/epicetra_experiment_log.json \
		--eval \
		--audit \
		--trace-out $(TRACE_DIR)/promotion-run.json \
		--report-out $(TRACE_DIR)/promotion-decision-packet.md \
		--report-html-out $(TRACE_DIR)/promotion-decision-review.html

review-workbench:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.review_workbench \
		$(TRACE_DIR)/promotion-run.json \
		--out $(TRACE_DIR)/review-workbench.html
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.review_workbench \
		$(TRACE_DIR)/promotion-run.json \
		--json \
		--out $(TRACE_DIR)/review-workbench.json

demo:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.demo \
		--json \
		--out $(TRACE_DIR)/portfolio-demo.json

reference-catalog:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.reference_catalog \
		--out $(TRACE_DIR)/reference-catalog-report.md
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.reference_catalog \
		--json \
		--out $(TRACE_DIR)/reference-catalog-report.json

service-interface:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_interface \
		--out $(TRACE_DIR)/service-interface-report.md
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_interface \
		--json \
		--out $(TRACE_DIR)/service-interface-report.json

service-runtime:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_runtime \
		--out $(TRACE_DIR)/service-runtime-demo.md
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_runtime \
		--json \
		--out $(TRACE_DIR)/service-runtime-demo.json

service-api:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_api \
		--out $(TRACE_DIR)/service-api-readiness.md
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_api \
		--json \
		--out $(TRACE_DIR)/service-api-readiness.json

service-mcp:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_mcp \
		--out $(TRACE_DIR)/service-mcp-readiness.md
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.service_mcp \
		--json \
		--out $(TRACE_DIR)/service-mcp-readiness.json

portfolio-index:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.portfolio_index build \
		--out $(TRACE_DIR)/index.html

serve-portfolio:
	PYTHONPATH=$(MODULE_PATH) $(PYTHON) -m bioml_agent.portfolio_index serve \
		--trace-dir $(TRACE_DIR) \
		--root . \
		--host 127.0.0.1 \
		--port 8765

clean-generated:
	rm -rf __pycache__ src/bioml_agent/__pycache__ tests/__pycache__ .pytest_cache build dist *.egg-info
