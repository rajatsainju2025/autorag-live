#!/usr/bin/env python3
"""
Development setup script for AutoRAG-Live.

This script sets up the development environment with all necessary tools,
pre-commit hooks, and dependencies.
"""
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

def run_command(cmd: List[str], description: str, check: bool = True) -> Optional[subprocess.CompletedProcess]:
    """Run a command with proper error handling."""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Exit code: {e.returncode}")
        if e.stdout:
            print(f"  Stdout: {e.stdout}")
        if e.stderr:
            print(f"  Stderr: {e.stderr}")
        if check:
            sys.exit(1)
        return None

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python 3.10+ required, found {version.major}.{version.minor}")
        sys.exit(1)
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")

def install_poetry():
    """Install Poetry if not available."""
    print("📝 Checking Poetry installation...")
    result = run_command(["poetry", "--version"], "Check Poetry", check=False)
    
    if result is None or result.returncode != 0:
        print("🔧 Installing Poetry...")
        install_cmd = [
            sys.executable, "-m", "pip", "install", "poetry"
        ]
        run_command(install_cmd, "Install Poetry")
    else:
        print("✅ Poetry is already installed")

def setup_poetry_environment():
    """Set up Poetry virtual environment and install dependencies."""
    print("🏗️ Setting up Poetry environment...")
    
    # Configure Poetry
    run_command(["poetry", "config", "virtualenvs.in-project", "true"], "Configure Poetry")
    
    # Install dependencies
    run_command(["poetry", "install"], "Install dependencies")
    
    # Install development dependencies
    dev_deps = [
        "pytest>=7.0",
        "pytest-cov>=4.0", 
        "pytest-mock>=3.0",
        "black>=23.0",
        "isort>=5.0",
        "ruff>=0.1.0",
        "mypy>=1.0",
        "bandit>=1.7",
        "pre-commit>=3.0",
        "jupyter>=1.0",
        "ipykernel>=6.0"
    ]
    
    for dep in dev_deps:
        run_command(["poetry", "add", "--group", "dev", dep], f"Install {dep.split('>=')[0]}")

def setup_pre_commit():
    """Set up pre-commit hooks."""
    print("🔨 Setting up pre-commit hooks...")
    
    # Install pre-commit
    run_command(["poetry", "run", "pre-commit", "install"], "Install pre-commit hooks")
    
    # Run pre-commit on all files to verify setup
    run_command(["poetry", "run", "pre-commit", "run", "--all-files"], "Run pre-commit check", check=False)

def create_development_config():
    """Create development configuration files."""
    print("⚙️ Creating development configuration...")
    
    # Create .env template
    env_template = """# Development Environment Variables
# Copy to .env and customize as needed

# AutoRAG Configuration
AUTORAG_DEBUG=true
AUTORAG_LOG_LEVEL=DEBUG
AUTORAG_CONFIG_DIR=./config

# Model Configuration  
AUTORAG_MODEL_CACHE_DIR=~/.cache/huggingface
AUTORAG_DEVICE=auto

# Database/Cache
AUTORAG_CACHE_DIR=~/.autorag/cache
AUTORAG_TEMP_DIR=/tmp/autorag

# Testing
AUTORAG_TEST_MODE=true
PYTEST_CURRENT_TEST=""
"""

    env_file = Path(".env.template")
    if not env_file.exists():
        env_file.write_text(env_template)
        print("✅ Created .env.template")
    
    # Create development config
    dev_config = """# Development Configuration
version: "0.1.0"
debug: true

logging:
  level: "DEBUG"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/autorag-dev.log"

data:
  cache_dir: "~/.autorag/cache"
  temp_dir: "/tmp/autorag"
  cleanup_on_exit: false

retrieval:
  default_top_k: 5  # Smaller for development
  timeout_seconds: 10.0
  batch_size: 10

evaluation:
  batch_size: 8  # Smaller for development
  timeout_seconds: 30.0
"""

    config_dir = Path("config")
    dev_config_file = config_dir / "dev.yaml"
    
    if not dev_config_file.exists():
        config_dir.mkdir(exist_ok=True)
        dev_config_file.write_text(dev_config)
        print("✅ Created config/dev.yaml")

def setup_vscode():
    """Set up VS Code configuration."""
    print("🔧 Setting up VS Code configuration...")
    
    vscode_dir = Path(".vscode")
    vscode_dir.mkdir(exist_ok=True)
    
    # Settings
    settings = {
        "python.defaultInterpreterPath": "./.venv/bin/python",
        "python.testing.pytestEnabled": True,
        "python.testing.pytestArgs": ["tests"],
        "python.linting.enabled": True,
        "python.linting.ruffEnabled": True,
        "python.formatting.provider": "black",
        "editor.formatOnSave": True,
        "editor.codeActionsOnSave": {
            "source.organizeImports": True
        },
        "files.associations": {
            "*.yaml": "yaml",
            "*.yml": "yaml"
        }
    }
    
    import json
    settings_file = vscode_dir / "settings.json"
    if not settings_file.exists():
        settings_file.write_text(json.dumps(settings, indent=2))
        print("✅ Created .vscode/settings.json")
    
    # Launch configuration
    launch_config = {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Current File",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "envFile": "${workspaceFolder}/.env"
            },
            {
                "name": "AutoRAG CLI",
                "type": "python", 
                "request": "launch",
                "module": "autorag_live.cli",
                "args": ["--help"],
                "console": "integratedTerminal"
            },
            {
                "name": "Run Tests",
                "type": "python",
                "request": "launch",
                "module": "pytest",
                "args": ["tests/", "-v"],
                "console": "integratedTerminal"
            }
        ]
    }
    
    launch_file = vscode_dir / "launch.json"
    if not launch_file.exists():
        launch_file.write_text(json.dumps(launch_config, indent=2))
        print("✅ Created .vscode/launch.json")

def create_makefile():
    """Create Makefile for common development tasks."""
    print("📋 Creating Makefile...")
    
    makefile_content = """# AutoRAG-Live Development Makefile

.PHONY: help install test lint format clean docs serve-docs build

help: ## Show this help message
	@echo "AutoRAG-Live Development Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \\033[36m%-20s\\033[0m %s\\n", $$1, $$2}'

install: ## Install development dependencies
	poetry install
	poetry run pre-commit install

test: ## Run tests with coverage
	poetry run pytest tests/ --cov=autorag_live --cov-report=html --cov-report=term

test-quick: ## Run tests without coverage
	poetry run pytest tests/ -x --tb=short

lint: ## Run linting
	poetry run ruff check autorag_live/ tests/
	poetry run mypy autorag_live/
	poetry run bandit -r autorag_live/

format: ## Format code
	poetry run black autorag_live/ tests/
	poetry run isort autorag_live/ tests/

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/ .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf build/ dist/ *.egg-info/

docs: ## Build documentation
	cd docs && mkdocs build

serve-docs: ## Serve documentation locally
	cd docs && mkdocs serve

build: ## Build package
	poetry build

dev-setup: ## Complete development setup
	python scripts/dev-setup.py

config-validate: ## Validate all configuration files
	find config/ -name "*.yaml" -exec poetry run python -m autorag_live.cli.config_migrate validate {} \\;

benchmark: ## Run performance benchmarks
	poetry run python -m autorag_live.cli benchmark --components all

pre-commit: ## Run pre-commit hooks on all files
	poetry run pre-commit run --all-files

update-deps: ## Update dependencies
	poetry update
	poetry run pre-commit autoupdate
"""

    makefile = Path("Makefile")
    if not makefile.exists():
        makefile.write_text(makefile_content)
        print("✅ Created Makefile")

def main():
    """Main setup function."""
    print("🚀 AutoRAG-Live Development Environment Setup")
    print("=" * 50)
    
    # Check prerequisites
    check_python_version()
    
    # Install and configure Poetry
    install_poetry()
    setup_poetry_environment()
    
    # Set up development tools
    setup_pre_commit()
    create_development_config()
    setup_vscode()
    create_makefile()
    
    print("\n✨ Development environment setup complete!")
    print("\nNext steps:")
    print("1. Copy .env.template to .env and customize")
    print("2. Run 'make test' to verify everything works")
    print("3. Use 'make help' to see available commands")
    print("4. Start developing! 🎉")

if __name__ == "__main__":
    main()