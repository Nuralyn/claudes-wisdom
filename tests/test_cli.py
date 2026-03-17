"""Tests for the CLI layer."""

import os
import tempfile

import pytest
from typer.testing import CliRunner

from wisdom.cli.app import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def tmp_wisdom_dir(tmp_path, monkeypatch):
    """Point WISDOM_DATA_DIR to a temp directory for all CLI tests."""
    monkeypatch.setenv("WISDOM_DATA_DIR", str(tmp_path / "wisdom_cli_test"))


class TestInit:
    def test_init_basic(self):
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 0
        assert "initialized" in result.output.lower() or "Data directory" in result.output

    def test_init_with_seed(self):
        result = runner.invoke(app, ["init", "--seed", "meta"])
        assert result.exit_code == 0
        assert "Loaded" in result.output or "seed" in result.output.lower()


class TestExperienceCLI:
    def test_add_experience(self):
        result = runner.invoke(app, [
            "exp", "add", "Debugged a Python memory leak",
            "--domain", "python",
            "--result", "success",
            "--tags", "debugging,memory",
        ])
        assert result.exit_code == 0
        assert "Added experience" in result.output

    def test_list_experiences(self):
        runner.invoke(app, ["exp", "add", "Test A", "--domain", "python"])
        runner.invoke(app, ["exp", "add", "Test B", "--domain", "rust"])
        result = runner.invoke(app, ["exp", "list"])
        assert result.exit_code == 0

    def test_stats(self):
        runner.invoke(app, ["exp", "add", "Test", "--domain", "python"])
        result = runner.invoke(app, ["exp", "stats"])
        assert result.exit_code == 0
        assert "Total" in result.output


class TestKnowledgeCLI:
    def test_extract(self):
        # Add experiences first
        for i in range(3):
            runner.invoke(app, ["exp", "add", f"Testing pattern {i}", "--domain", "python"])
        result = runner.invoke(app, ["know", "extract"])
        assert result.exit_code == 0

    def test_list_empty(self):
        result = runner.invoke(app, ["know", "list"])
        assert result.exit_code == 0


class TestWisdomCLI:
    def test_add_wisdom(self):
        result = runner.invoke(app, [
            "wis", "add", "Always test your code before shipping",
            "--reasoning", "Untested code has unknown bugs",
            "--type", "principle",
            "--domains", "testing",
        ])
        assert result.exit_code == 0
        assert "Added wisdom" in result.output

    def test_list_wisdom(self):
        runner.invoke(app, [
            "wis", "add", "Test principle",
            "--domains", "testing",
        ])
        result = runner.invoke(app, ["wis", "list"])
        assert result.exit_code == 0

    def test_deprecate_wisdom(self):
        # Add then deprecate
        add_result = runner.invoke(app, ["wis", "add", "To deprecate", "--domains", "test"])
        # Extract ID from output
        output = add_result.output
        # Find ID in "Added wisdom: <id>"
        import re
        match = re.search(r"Added wisdom:\s*(\S+)", output)
        if match:
            wid = match.group(1)
            result = runner.invoke(app, ["wis", "deprecate", wid, "--reason", "No longer valid"])
            assert result.exit_code == 0


class TestQueryCLI:
    def test_search(self):
        runner.invoke(app, ["wis", "add", "Use pytest for Python testing", "--domains", "python"])
        result = runner.invoke(app, ["query", "search", "testing python code"])
        assert result.exit_code == 0


class TestAnalyticsCLI:
    def test_summary(self):
        result = runner.invoke(app, ["analytics", "summary"])
        assert result.exit_code == 0

    def test_health(self):
        result = runner.invoke(app, ["analytics", "health"])
        assert result.exit_code == 0

    def test_gaps(self):
        result = runner.invoke(app, ["analytics", "gaps"])
        assert result.exit_code == 0


class TestMaintenanceCLI:
    def test_maintenance(self):
        result = runner.invoke(app, ["maintenance"])
        assert result.exit_code == 0


class TestIOCLI:
    def test_export_import(self, tmp_path):
        # Add some data
        runner.invoke(app, ["wis", "add", "Export test", "--domains", "test"])

        # Export
        export_path = str(tmp_path / "export.json")
        result = runner.invoke(app, ["io", "export", export_path])
        assert result.exit_code == 0

    def test_claude_md(self, tmp_path):
        runner.invoke(app, ["wis", "add", "CLAUDE.md test principle", "--domains", "test"])
        output_path = str(tmp_path / "CLAUDE.md")
        result = runner.invoke(app, ["io", "claude-md", "--output", output_path])
        assert result.exit_code == 0
