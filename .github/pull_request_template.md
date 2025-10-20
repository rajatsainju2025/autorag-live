## Summary

Brief description of the changes made in this PR.

### Changes Made
- [ ] **Scope**: Changes are minimal and focused
- [ ] **Tests**: Adds/updates tests where behavior changes
- [ ] **Docs**: Updates documentation (README/docs/*) where applicable
- [ ] **Breaking**: Introduces breaking changes (with migration guide)

## Changes

### Code Changes
- Detailed description of code modifications
- New features added
- Bug fixes implemented
- Refactoring performed

### Testing
- Unit tests added/updated
- Integration tests added/updated
- Manual testing performed

### Documentation
- README updated
- API docs updated
- Examples added/updated

## How to test

### Automated Testing
```bash
# Run relevant tests
pytest tests/path/to/relevant/tests.py -v

# Run with coverage
pytest --cov=autorag_live --cov-report=html
```

### Manual Testing
1. Step-by-step testing instructions
2. Expected outcomes
3. Edge cases to verify

## Checklist

### Code Quality
- [ ] pre-commit passed (black, isort, ruff)
- [ ] Type hints added/updated
- [ ] Docstrings added/updated
- [ ] No new linting errors

### Testing
- [ ] pytest passed locally
- [ ] New tests added for new functionality
- [ ] Existing tests still pass
- [ ] Test coverage maintained/improved

### Security & Safety
- [ ] No secrets committed
- [ ] Input validation added for new user inputs
- [ ] Error handling improved

### CI/CD
- [ ] CI checks pass
- [ ] No performance regressions
- [ ] Documentation builds successfully

### Review
- [ ] Self-reviewed code
- [ ] Changes are logically organized
- [ ] Commit messages are clear and descriptive

## Related Issues
Closes #issue_number
Relates to #issue_number

## Screenshots (if applicable)
Add screenshots of UI changes, before/after comparisons, etc.

## Performance Impact
- [ ] No performance impact
- [ ] Performance improved
- [ ] Performance impact acceptable (explain below)

Performance impact details:
- Benchmark results
- Memory usage changes
- Latency changes
