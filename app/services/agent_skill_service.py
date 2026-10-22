from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_SKILL_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_RESOURCE_EXTENSIONS = {".csv", ".json", ".md", ".txt", ".xml", ".yaml", ".yml"}


class AgentSkillError(ValueError):
    """Raised when a local Agent Skill is invalid or cannot be read safely."""


class AgentSkillService:
    """Discover and progressively load trusted, repository-local Agent Skills."""

    def __init__(self, skills_dir: Path, max_resource_bytes: int = 100_000):
        self.skills_dir = skills_dir.resolve()
        self.max_resource_bytes = max_resource_bytes

    @staticmethod
    def _split_skill_document(content: str) -> tuple[dict[str, Any], str]:
        if not content.startswith("---\n"):
            raise AgentSkillError("SKILL.md must start with YAML frontmatter")
        try:
            _, frontmatter, body = content.split("---", 2)
        except ValueError as exc:
            raise AgentSkillError("SKILL.md frontmatter is not closed") from exc
        metadata = yaml.safe_load(frontmatter) or {}
        if not isinstance(metadata, dict):
            raise AgentSkillError("SKILL.md frontmatter must be a mapping")
        return metadata, body.strip()

    def _skill_dir(self, name: str) -> Path:
        if not _SKILL_NAME.fullmatch(name):
            raise AgentSkillError(f"Invalid skill name: {name}")
        skill_dir = (self.skills_dir / name).resolve()
        if skill_dir.parent != self.skills_dir:
            raise AgentSkillError("Skill path escapes the configured skills directory")
        return skill_dir

    def _read_skill(self, skill_dir: Path) -> tuple[dict[str, Any], str]:
        path = skill_dir / "SKILL.md"
        if not path.is_file():
            raise AgentSkillError(f"Skill not found: {skill_dir.name}")
        if path.stat().st_size > self.max_resource_bytes:
            raise AgentSkillError(f"Skill instructions exceed {self.max_resource_bytes} bytes")
        metadata, body = self._split_skill_document(path.read_text(encoding="utf-8"))
        name = metadata.get("name")
        description = metadata.get("description")
        if name != skill_dir.name or not _SKILL_NAME.fullmatch(str(name or "")):
            raise AgentSkillError(f"Skill name must match directory: {skill_dir.name}")
        if not isinstance(description, str) or not description.strip():
            raise AgentSkillError(f"Skill description is required: {skill_dir.name}")
        return metadata, body

    def list_skills(self) -> list[dict[str, str]]:
        """Advertise only skill metadata; instructions remain unloaded."""
        if not self.skills_dir.is_dir():
            return []
        skills = []
        for skill_dir in sorted(path for path in self.skills_dir.iterdir() if path.is_dir()):
            try:
                metadata, _ = self._read_skill(skill_dir)
            except AgentSkillError:
                continue
            skills.append({"name": metadata["name"], "description": metadata["description"]})
        return skills

    def load_skill(self, name: str) -> dict[str, Any]:
        """Load one skill's instructions and advertise its readable resources."""
        skill_dir = self._skill_dir(name)
        metadata, body = self._read_skill(skill_dir)
        resources = [
            path.relative_to(skill_dir).as_posix()
            for path in sorted(skill_dir.rglob("*"))
            if path.is_file()
            and path.name != "SKILL.md"
            and "agents" not in path.relative_to(skill_dir).parts
            and path.suffix.lower() in _RESOURCE_EXTENSIONS
        ]
        return {
            "name": metadata["name"],
            "description": metadata["description"],
            "instructions": body,
            "resources": resources,
        }

    def read_skill_resource(self, name: str, resource_path: str) -> dict[str, str]:
        """Read one declared text resource without allowing absolute paths or traversal."""
        skill_dir = self._skill_dir(name)
        self._read_skill(skill_dir)
        relative = Path(resource_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AgentSkillError("Resource path must stay inside the selected skill")
        path = (skill_dir / relative).resolve()
        if not path.is_relative_to(skill_dir) or not path.is_file():
            raise AgentSkillError(f"Skill resource not found: {resource_path}")
        if "agents" in relative.parts or path.suffix.lower() not in _RESOURCE_EXTENSIONS:
            raise AgentSkillError(f"Unsupported skill resource: {resource_path}")
        if path.stat().st_size > self.max_resource_bytes:
            raise AgentSkillError(f"Skill resource exceeds {self.max_resource_bytes} bytes")
        return {
            "skill": name,
            "resource": relative.as_posix(),
            "content": path.read_text(encoding="utf-8"),
        }
