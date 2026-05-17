import pytest
from day5.deployment import (
    generate_docker_compose, DockerConfig,
    show_gitops_pattern, show_auth_pattern,
    show_langsmith_tracing, get_deployment_checklist
)


class TestDockerCompose:
    def test_generates_valid_yaml_structure(self):
        services = [
            DockerConfig("backend", "my-app:latest", 8000, 8000, ["OPENAI_API_KEY"]),
        ]
        yaml = generate_docker_compose(services)
        assert "version:" in yaml
        assert "services:" in yaml
        assert "backend:" in yaml
        assert "8000:8000" in yaml

    def test_depends_on_included(self):
        services = [
            DockerConfig("backend",  "api:1.0", 8000, 8000, ["API_KEY"]),
            DockerConfig("frontend", "ui:1.0",  8501, 8501, [], depends_on=["backend"]),
        ]
        yaml = generate_docker_compose(services)
        assert "depends_on:" in yaml
        assert "backend" in yaml

    def test_healthcheck_included(self):
        services = [
            DockerConfig("api", "api:1.0", 8000, 8000, [], healthcheck_path="/health"),
        ]
        yaml = generate_docker_compose(services)
        assert "healthcheck:" in yaml
        assert "/health" in yaml


class TestPatterns:
    def test_gitops_has_github_actions(self):
        code = show_gitops_pattern()
        assert "github" in code.lower() or "actions" in code.lower()
        assert "pytest" in code

    def test_auth_pattern_has_jwt(self):
        code = show_auth_pattern()
        assert "jwt" in code.lower() or "token" in code.lower()

    def test_langsmith_pattern_has_tracing(self):
        code = show_langsmith_tracing()
        assert "traceable" in code
        assert "LANGCHAIN_TRACING" in code


class TestChecklist:
    def test_checklist_is_list(self):
        items = get_deployment_checklist()
        assert isinstance(items, list)
        assert len(items) >= 8

    def test_checklist_has_key_items(self):
        items = get_deployment_checklist()
        text = " ".join(items).lower()
        assert "test" in text
        assert "docker" in text
        assert "health" in text
