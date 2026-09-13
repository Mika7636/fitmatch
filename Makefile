# FitMatch.
#
#   make data      build all three CSVs, then validate them
#   make test      run the test suite
#   make audit     measure feasibility/coverage, write reports/
#   make evaluate  offline evaluation, five configurations, write reports/
#   make coldstart cold-start segments and the case against CF
#   make demo      the three demo profiles, printed
#   make api       serve the session-3 API
#   make frontend  serve the session-4 UI (needs npm install first)
#
# GNU make is NOT available on the development machine, so build_data.bat and
# run_tests.bat exist alongside this file and are the supported entry points
# there. On Windows, run the same commands by hand:
#   python -m src.data.build_products
#   python -m src.data.build_users
#   python -m src.data.build_interactions
#   python -m src.data.validate
#   python -m pytest
#   python -m src.recommender.audit
#   python -m src.evaluation.evaluate
#   python -m src.evaluation.cold_start
#   python scripts/seed_demo.py
#   python -m src.api
#   cd frontend && npm install && npm run dev

PYTHON ?= python

.PHONY: help data products users interactions validate test audit evaluate coldstart demo recommend api frontend clean clean-cache

help:
	@echo "make data         - build products.csv, users.csv, interactions.csv and validate"
	@echo "make products     - build data/processed/products.csv"
	@echo "make users        - build data/processed/users.csv"
	@echo "make interactions - build data/processed/interactions.csv (needs the other two)"
	@echo "make validate     - check schema, vocabularies and referential integrity"
	@echo "make test         - run the recommender test suite (pytest)"
	@echo "make audit        - write reports/feasibility_report.md"
	@echo "make evaluate     - offline evaluation, five configs, write reports/"
	@echo "make coldstart    - cold-start segments and the CF counterfactual"
	@echo "make demo         - print the three demo profiles and their results"
	@echo "make recommend    - sample run: user 42, top 10"
	@echo "make api          - serve the API on http://127.0.0.1:8000"
	@echo "make frontend     - serve the UI on http://localhost:5173"
	@echo "make clean        - delete the processed CSVs (raw downloads are left alone)"
	@echo "make clean-cache  - delete the cached TF-IDF model"

# products and users are independent; interactions reads both.
data: products users interactions validate

products:
	$(PYTHON) -m src.data.build_products

users:
	$(PYTHON) -m src.data.build_users

interactions:
	$(PYTHON) -m src.data.build_interactions

validate:
	$(PYTHON) -m src.data.validate

# ----------------------------------------------------------------------
# Session 2 -- the recommendation engine
# ----------------------------------------------------------------------
test:
	$(PYTHON) -m pytest

audit:
	$(PYTHON) -m src.recommender.audit

evaluate:
	$(PYTHON) -m src.evaluation.evaluate

coldstart:
	$(PYTHON) -m src.evaluation.cold_start

demo:
	$(PYTHON) scripts/seed_demo.py

recommend:
	$(PYTHON) -m src.recommender.pipeline --user-id 42 --top-n 10

# ----------------------------------------------------------------------
# Session 3 -- the API
# ----------------------------------------------------------------------
api:
	$(PYTHON) -m src.api

# ----------------------------------------------------------------------
# Session 4 -- the frontend
# ----------------------------------------------------------------------
frontend:
	cd frontend && npm install && npm run dev

clean:
	-rm -f data/processed/products.csv data/processed/users.csv data/processed/interactions.csv

clean-cache:
	-rm -f data/cache/tfidf_model.joblib
