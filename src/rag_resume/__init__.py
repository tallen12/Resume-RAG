from rag_resume.backends.langchain.graph import LangGraphPipeline
from rag_resume.pipelines.resume_builder import ResumeBuilderPipeline, ResumeBuilderState


def main() -> None:
    graph = LangGraphPipeline(ResumeBuilderPipeline())
    print(graph.invoke(ResumeBuilderState("Query", [])))
