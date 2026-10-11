import mlflow
import uuid
from genie_api import query_genie  # Assume this is a utility to query Genie Spaces

PHARMA_GENIE_NAME = "Pharma RAG & Agentic AI Platform - SciLifeLab"
BAKEHOUSE_GENIE_NAME = "Bakehouse Sales Starter Space"

PHARMA_KEYWORDS = ["kinase", "compound", "assay", "protocol", "IC50", "publication"]
BAKEHOUSE_KEYWORDS = ["sales", "inventory", "customer", "product", "revenue"]

def classify_with_llm(query):
    # Placeholder for LLM function, e.g. Databricks Model Serving endpoint
    # Implement actual LLM call or return mock for now
    return "pharma" if "drug" in query.lower() else "bakehouse"

def route_query(query):
    if any(kw in query.lower() for kw in PHARMA_KEYWORDS):
        return "pharma", "deterministic"
    if any(kw in query.lower() for kw in BAKEHOUSE_KEYWORDS):
        return "bakehouse", "deterministic"
    domain = classify_with_llm(query)
    return domain, "llm"

def supervisor_agent(query, user_id, mlflow_run_name="SupervisorAgentQuery"):
    trace_id = str(uuid.uuid4())
    domain, routing_method = route_query(query)
    genie_name = PHARMA_GENIE_NAME if domain == "pharma" else BAKEHOUSE_GENIE_NAME

    with mlflow.start_run(run_name=mlflow_run_name):
        mlflow.log_param("user_id", user_id)
        mlflow.log_param("query", query)
        mlflow.log_param("domain", domain)
        mlflow.log_param("routing_method", routing_method)
        mlflow.log_param("trace_id", trace_id)

        result = query_genie(genie_name, query)
        mlflow.log_param("genie_space", genie_name)
        mlflow.log_metric("result_length", len(result.get("answer", "")))

        mlflow.set_tag("deployment_trace_id", trace_id)

        return {
            "answer": result.get("answer"),
            "domain": domain,
            "routing_method": routing_method,
            "trace_id": trace_id,
            "mlflow_run_id": mlflow.active_run().info.run_id
        }

# Usage Example:
if __name__ == "__main__":
    query = "List all kinase inhibitors published in 2025"
    user_id = "test_user"
    response = supervisor_agent(query, user_id)
    print(response)
