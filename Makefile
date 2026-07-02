PY := .venv/bin/python
# engine CLI keys (map to dataset dirs internally)
KEYS := axt nebius r2egym-sft r2egym-verifier openhands moatless sharegpt

.PHONY: setup download analyze dashboard all clean

setup:                      ## create venv + install deps (uv if available, else venv+pip)
	@if command -v uv >/dev/null 2>&1; then \
	  echo "using uv"; uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt; \
	else \
	  echo "using venv+pip"; python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt; \
	fi

download:                   ## fetch all datasets into agentic_traces/ (~7 GB)
	$(PY) download_datasets.py

analyze:                    ## run the analysis engine over every dataset
	@for k in $(KEYS); do $(PY) analysis_engine.py $$k; done

dashboard:                  ## build dashboard.html from the *.stats.json files
	$(PY) build_dashboard.py

all: download analyze dashboard   ## full pipeline, end to end

clean:                      ## remove venv + generated dashboard (keeps reports)
	rm -rf .venv dashboard.html **/__pycache__
