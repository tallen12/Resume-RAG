from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from seriacade.implementations.pydantic import PydanticJsonCodec

from rag_resume.agentic.backends.langchain.embeddings import LangchainEmbeddingModel, LangchainInMemoryVectorStore
from rag_resume.agentic.backends.langchain.graph import LangchainGraph
from rag_resume.agentic.backends.langchain.llms import LangChainChatLLM
from rag_resume.pipelines.resume_builder import ResumeBuilderPipeline, ResumeBuilderState, ResumeBuilderVectorMetadata


@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


def main() -> None:
    ollama_llm = ChatOllama(model="llama3.1:latest", base_url="https://ollama.ml.interfior.moe")
    ollama_embedding = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="https://ollama.ml.interfior.moe")
    embedding_wrapper = LangchainEmbeddingModel(ollama_embedding)
    llm_chat = LangChainChatLLM(ollama_llm)

    llmm_with_tools = ollama_llm.bind_tools([multiply])
    r = llmm_with_tools.invoke("Multiply 5 and 3")
    print(r)

    vector_store = LangchainInMemoryVectorStore(embedding_wrapper, PydanticJsonCodec(ResumeBuilderVectorMetadata))

    vector_store.add(
        [
            r"Implemented a fine-tuned BERT machine learning model to replace a PyTorch CNN used to classify background check data deployed using Docker on Kubernetes; reduced manual reviews by 50\% and cut review time by 90\%",
            "Built an authentication microservice for API Gateway on Kubernetes providing OAuth JWT validation for all frontend services, collaborated with Frontend and Devops teams to ensure functionality; implemented in Python using FastAPI",
            "Developed backend services for the Third Party Risk Management Platform, enabling automated risk compliance checks for 4M+ members; implemented with Scala/Akka",
            "Led Development of a REST microservice to centralize notification and branding configurations for internal and external services to reduce data duplication; implemented in Python using Flask and SQLAlchemy with PostgreSQL",
            "Introduced new workflows and Python scripts that reduced development time for routine new features in core services from weeks to days",
            "Maintained public facing API documentation for various services using OpenAPI and Swagger",
            r"Led development of an automated background check service in Python, integrating data from multiple REST/SOAP APIs, used NLP and machine learning to extract and validate information against customer profiles while ensuring legal compliance; this service generated 50\% of company revenue during this period",
            "Scaled backend services with an event-driven architecture using RabbitMQ, Celery for python services, and Redis caching; achieved a 5x performance improvement",
            "Implemented comprehensive end-to-end automated testing with PyTest, Selenium, and Cucumber/Gherkin for Behavior-Driven Development; enabled thorough regression testing for all services",
            "Configured Jenkins workflows for CI/CD to run linting, unit and integration tests, and builds using Docker and Helm; enabled automated deployments for development environments",
            r"Developed and maintained a REST API in Elixir Phoenix with PostgreSQL to manage customer configurations; enabled customer self-service and reduced onboarding time by 90\%",
            "Served as on-call backend support during U.S. hours for to debug and fix production incidents; used Prometheus, Grafana, Logz.io, and Sentry to aid in diagnosing issues",
            "Developed, as one of the first four developers at a fast-paced startup, the core backend service for a secure and privacy focused platform to share sensitive data using end-to-end encryption; implemented with Scala and MongoDB",
            "Added Firebase support to core platform allowing integration with Android and IOS applications",
            "Developed the Python SDK for the Evident platform and used it to implement several of the first revenue-generating services; deployed on AWS EC2 using Ansible and Terraform",
        ]
    )

    pipeline = ResumeBuilderPipeline(llm_chat, vector_store)
    materialized_pipeline = LangchainGraph(pipeline)
    result: ResumeBuilderState = materialized_pipeline.invoke(
        ResumeBuilderState(description="Python Engineer building APIs in using FastAPI and Postgres")
    )
    print(result)
