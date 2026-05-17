# Northstar Analytics Handbook

Northstar Analytics is a fictional enterprise data platform company used for local RAG demos.

## What We Do

Northstar Analytics helps retail and logistics teams unify reporting, forecasting, and anomaly detection.

The company product combines:

- a document ingestion pipeline
- a metrics dashboard
- a retrieval assistant for internal knowledge
- a monitoring service for operational alerts

## Support Policy

Standard support hours are Monday through Friday, 9:00 AM to 6:00 PM India Standard Time.

Enterprise support customers receive:

- email support
- shared Slack support
- incident response within 2 hours for P1 issues

## Deployment Notes

The preferred local deployment stack is:

- FastAPI for backend APIs
- Streamlit for the lightweight demo UI
- OpenAI models for generation and embeddings
- local documents stored in the `data/` directory

## Demo Talking Point

This demo is designed to answer company or product questions from local documents first.
If the answer is not available locally, the assistant should use a web-search tool instead of guessing.
