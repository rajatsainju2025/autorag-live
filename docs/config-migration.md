# Configuration Migration Guide

This guide helps you migrate configurations between different versions of AutoRAG-Live.

## Overview

Configuration migration is important when:
- Upgrading to a new version of AutoRAG-Live
- Changing configuration schema or structure
- Moving from development to production environments
- Adopting new features that require configuration changes

## Migration Strategies

### 1. Version-Based Migration

The system supports automatic migration between configuration versions:

```python
from autorag_live.utils.validation import migrate_config
from omegaconf import OmegaConf

# Load old config
old_config = OmegaConf.load("config_v1.yaml")

# Migrate to new version
new_config = migrate_config(old_config, from_version="1.0", to_version="2.0")

# Save migrated config
OmegaConf.save(new_config, "config_v2.yaml")
```

### 2. Environment Variable Migration

Migrate from hardcoded values to environment variables:

**Before (v0.1.0):**
```yaml
# config.yaml
retrieval:
  dense:
    model_name: "all-MiniLM-L6-v2"
    cache_embeddings: true
```

**After (v0.2.0+):**
```yaml
# config.yaml
retrieval:
  dense:
    model_name: ${AUTORAG_MODEL_NAME:all-MiniLM-L6-v2}
    cache_embeddings: ${AUTORAG_CACHE_EMBEDDINGS:true}
```

Set environment variables:
```bash
export AUTORAG_MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
export AUTORAG_CACHE_EMBEDDINGS=true
```

### 3. Schema Structure Changes

#### Adding New Fields

When new optional fields are added:

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")

# Add new fields with defaults
if "new_feature" not in config:
    config.new_feature = {"enabled": true, "threshold": 0.5}

OmegaConf.save(config, "config.yaml")
```

#### Renaming Fields

When fields are renamed:

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")

# Migrate old field name to new name
if "old_field" in config.retrieval:
    config.retrieval.new_field = config.retrieval.old_field
    del config.retrieval.old_field

OmegaConf.save(config, "config.yaml")
```

#### Restructuring Nested Configs

When configuration structure changes:

**Before:**
```yaml
retrieval:
  bm25_config:
    k1: 1.5
    b: 0.75
```

**After:**
```yaml
retrieval:
  bm25:
    parameters:
      k1: 1.5
      b: 0.75
    enabled: true
```

Migration script:
```python
from omegaconf import OmegaConf

config = OmegaConf.load("old_config.yaml")

# Restructure
if "bm25_config" in config.retrieval:
    config.retrieval.bm25 = {
        "parameters": config.retrieval.bm25_config,
        "enabled": True
    }
    del config.retrieval.bm25_config

OmegaConf.save(config, "new_config.yaml")
```

## Common Migration Scenarios

### Scenario 1: Upgrading from 0.1.0 to 0.2.0

**Changes:**
- Added `cache` section
- Renamed `eval` to `evaluation`
- Added new metrics configuration

**Migration:**
```python
from omegaconf import OmegaConf

# Load v0.1.0 config
config = OmegaConf.load("config_v0.1.0.yaml")

# Add cache section (new in v0.2.0)
config.cache = {
    "enabled": True,
    "backend": "memory",
    "ttl": 3600
}

# Rename eval to evaluation
if "eval" in config:
    config.evaluation = config.eval
    del config.eval

# Add new metrics
if "metrics" not in config.evaluation:
    config.evaluation.metrics = {
        "relevance": {"enabled": True},
        "faithfulness": {"enabled": True}
    }

# Save v0.2.0 config
OmegaConf.save(config, "config_v0.2.0.yaml")
```

### Scenario 2: Moving to Component-Based Configuration

**Before (single file):**
```yaml
# config.yaml
app_name: "autorag-live"
retrieval:
  bm25: {...}
  dense: {...}
evaluation:
  metrics: {...}
pipeline:
  optimizer: {...}
```

**After (multiple files):**
```
config/
├── config.yaml          # Base configuration
├── retrieval/
│   └── default.yaml    # Retrieval config
├── evaluation/
│   └── default.yaml    # Evaluation config
└── pipeline/
    └── default.yaml    # Pipeline config
```

Migration:
```python
from omegaconf import OmegaConf
from pathlib import Path

# Load monolithic config
config = OmegaConf.load("config.yaml")

# Create directories
Path("config/retrieval").mkdir(parents=True, exist_ok=True)
Path("config/evaluation").mkdir(parents=True, exist_ok=True)
Path("config/pipeline").mkdir(parents=True, exist_ok=True)

# Split into components
OmegaConf.save(config.retrieval, "config/retrieval/default.yaml")
OmegaConf.save(config.evaluation, "config/evaluation/default.yaml")
OmegaConf.save(config.pipeline, "config/pipeline/default.yaml")

# Save base config (without split sections)
base_config = OmegaConf.create({
    "app_name": config.app_name,
    "version": config.get("version", "0.2.0")
})
OmegaConf.save(base_config, "config/config.yaml")
```

### Scenario 3: Type Changes

When configuration value types change:

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config.yaml")

# Convert string to list (v0.1.0 -> v0.2.0)
if isinstance(config.retrieval.models, str):
    config.retrieval.models = [config.retrieval.models]

# Convert int to float (v0.2.0 -> v0.3.0)
if isinstance(config.evaluation.threshold, int):
    config.evaluation.threshold = float(config.evaluation.threshold)

OmegaConf.save(config, "config.yaml")
```

## CLI Migration Tools

Use built-in CLI tools for migration:

### Validate Configuration
```bash
python -m autorag_live.cli config validate --config config.yaml
```

### Migrate Configuration
```bash
python -m autorag_live.cli config migrate \
    --from-version 0.1.0 \
    --to-version 0.2.0 \
    --input config_old.yaml \
    --output config_new.yaml
```

### Show Configuration Diff
```bash
python -m autorag_live.cli config diff \
    config_old.yaml \
    config_new.yaml
```

## Best Practices

### 1. Backup Before Migration
```bash
cp config.yaml config.yaml.backup
cp -r config/ config.backup/
```

### 2. Test Migrated Configuration
```python
from autorag_live.utils.config import ConfigManager
from autorag_live.utils.validation import validate_config
from autorag_live.utils.schema import AutoRAGConfig

# Load and validate
config_manager = ConfigManager()
validate_config(config_manager.config, AutoRAGConfig)
```

### 3. Use Version Control
Track configuration changes in git:
```bash
git add config/
git commit -m "chore: migrate config to v0.2.0"
```

### 4. Document Custom Changes
Add comments to configuration files:
```yaml
# Migrated from v0.1.0 to v0.2.0 on 2025-12-03
# Changed: Added cache section, renamed eval to evaluation
cache:
  enabled: true  # New in v0.2.0
  backend: memory
```

### 5. Gradual Migration
For large projects, migrate incrementally:

1. Test with new config in development
2. Deploy to staging environment
3. Monitor for issues
4. Deploy to production

## Troubleshooting

### Validation Errors After Migration

```python
from autorag_live.utils.validation import validate_config, ConfigurationError

try:
    validate_config(config, AutoRAGConfig)
except ConfigurationError as e:
    print(f"Validation error: {e}")
    # Fix issues in config
```

### Missing Required Fields

Check the schema for required fields:
```python
from autorag_live.utils.schema import AutoRAGConfig
import dataclasses

for field in dataclasses.fields(AutoRAGConfig):
    print(f"{field.name}: required={field.default is dataclasses.MISSING}")
```

### Type Mismatches

Use OmegaConf utilities to check types:
```python
from omegaconf import OmegaConf

# Get type of config value
print(OmegaConf.get_type(config, "retrieval.bm25.k1"))
```

## Additional Resources

- [Configuration Documentation](configuration.md)
- [Schema Reference](../autorag_live/utils/schema.py)
- [Validation Guide](error-handling-guide.md)
- [Migration CLI Reference](cli-reference.md)

## Version-Specific Migration Guides

### v0.1.0 → v0.2.0
- Added cache configuration section
- Renamed `eval` to `evaluation`
- Added metrics configuration
- Introduced component-based config structure

### v0.2.0 → v0.3.0 (Planned)
- Add FAISS/Qdrant integration configs
- Enhanced bandit optimizer settings
- LLM judge configuration options
