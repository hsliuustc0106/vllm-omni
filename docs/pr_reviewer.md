# VLLM-Omni PR Reviewer Bot

## Overview

The VLLM-Omni PR Reviewer Bot is an automated code reviewer that uses the GLM-4.7 API to provide intelligent, comprehensive code reviews for pull requests in the vllm-omni repository.

## Features

- **Intelligent Code Analysis**: Leverages GLM-4.7 for understanding code context and providing meaningful feedback
- **Comprehensive Reviews**: Covers code quality, architecture, security, testing, and documentation
- **Structured Output**: Provides well-formatted reviews with clear sections and actionable suggestions
- **Rate Limiting**: Built-in cooldown mechanism to prevent excessive API usage
- **Retry Logic**: Automatic retries with exponential backoff for transient API failures
- **Defensive Parsing**: Robust validation of API responses to handle malformed data
- **Cost Control**: Only repository members/collaborators/owners can trigger reviews

## How to Use

### Triggering a Review

To trigger an automated code review on any pull request:

1. Navigate to the pull request you want to review
2. Add a comment with: `@vllm-omni-reviewer`
3. The bot will automatically analyze the changes and post a comprehensive review

### Permissions

Only the following users can trigger the PR reviewer bot:
- Repository **Owners**
- Repository **Members**
- Repository **Collaborators** (with write access)

This restriction helps control API costs and ensures reviews are only requested by trusted contributors.

### Review Output

The bot provides reviews in the following format:

1. **Overview** - Summary of changes and overall assessment
2. **Code Quality** - Style, potential bugs, performance, error handling
3. **Architecture & Design** - Integration, design patterns, improvements
4. **Security & Safety** - Security concerns, resource management, validation
5. **Testing & Documentation** - Test coverage, documentation completeness
6. **Specific Suggestions** - Line-by-line feedback with `file:line` references
7. **Approval Status** - LGTM, LGTM with suggestions, or Changes requested

## Rate Limiting and Cooldown

The bot includes a cooldown mechanism to prevent excessive API usage:

- **Default cooldown**: 5 minutes between reviews per PR
- **Configurable**: Can be adjusted via `PR_REVIEWER_COOLDOWN_MINUTES` environment variable
- **Smart detection**: Checks for previous bot comments before starting a review

If you trigger a review within the cooldown period, the bot will log a message and skip the review.

## Cost Estimate

Based on GLM-4.7 API pricing:
- **Input**: ~0.5 CNY per 1M tokens
- **Output**: ~2 CNY per 1M tokens

Typical PR reviews cost approximately **0.50-5 CNY** depending on the size of the diff.

## Architecture

```
┌─────────────────┐
│  PR Comment     │
│  @vllm-omni-    │
│  reviewer       │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GitHub Actions Workflow        │
│  (.github/workflows/            │
│   pr-reviewer.yml)              │
│                                 │
│  - Python 3.10                  │
│  - requests==2.31.0             │
│  - pyyaml==6.0.1                │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  PR Reviewer Script             │
│  (.github/scripts/              │
│   pr_reviewer.py)               │
│                                 │
│  1. Check cooldown              │
│  2. Fetch PR details & diff     │
│  3. Build review prompt         │
│  4. Call GLM-4.7 API            │
│     (with retry logic)          │
│  5. Validate response           │
│  6. Post review comment         │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  GLM-4.7 API                    │
│  (open.bigmodel.cn)             │
└─────────────────────────────────┘
```

## Testing

### How to Test the Bot

Team members can test the PR reviewer bot by:

1. **Create a test PR** - Make a small, safe change (e.g., documentation update)
2. **Open the PR** - Create a pull request with a descriptive title
3. **Trigger the review** - Comment `@vllm-omni-reviewer` on the PR
4. **Monitor results** - Check the Actions tab for workflow execution logs

### Running Unit Tests

The bot includes comprehensive unit tests that can be run locally:

```bash
# Run all tests
pytest .github/tests/test_pr_reviewer.py -v

# Run specific test
pytest .github/tests/test_pr_reviewer.py::TestCheckTrigger -v

# Run with coverage
pytest .github/tests/test_pr_reviewer.py --cov=.github/scripts/pr_reviewer.py --cov-report=term-missing
```

### What to Look For

When testing, verify that:
- [ ] The workflow triggers on the `@vllm-omni-reviewer` comment
- [ ] The cooldown mechanism works correctly
- [ ] The GLM API call completes without errors (with retry if needed)
- [ ] A review comment is posted to the PR
- [ ] The review content is meaningful and well-structured
- [ ] The cost is within the expected range (0.50-5 CNY)

### Safe Test Changes

For testing, consider making these types of safe changes:
- Documentation updates (like adding this Testing section)
- Comment improvements
- README enhancements
- Non-functional file additions

### Example Test PR

A good test PR might:
- Update a documentation file
- Add explanatory comments
- Improve code formatting
- Fix a minor typo

These changes are safe to merge if the test is successful and won't affect functionality.

## Troubleshooting

### Review Not Appearing

If the review doesn't appear after triggering:

1. **Check permissions** - Verify you have Owner/Member/Collaborator access
2. **Check Actions tab** - Look for workflow execution and view logs
3. **Check cooldown** - If another review was posted recently, wait for the cooldown period
4. **Check API key** - Ensure `GLM_API_KEY` is configured in repository secrets

### Permission Denied Error

If you see "Permission denied" or similar:
- Only repository members/collaborators/owners can trigger reviews
- Contact a repository maintainer for access

### API Errors

If the GLM API call fails:
- Check the Actions tab for detailed error logs
- Verify the `GLM_API_KEY` secret is correctly configured
- Ensure sufficient API quota is available
- The bot will automatically retry up to 3 times with exponential backoff

### Review Seems Truncated

If the review appears incomplete:
- Large diffs may be truncated at 100,000 characters
- Check the logs for truncation warnings
- Consider breaking large PRs into smaller chunks

## Configuration

### Required Secrets

The following secret must be configured in the repository settings:

- `GLM_API_KEY` - Your GLM (BigModel) API key for accessing the GLM-4.7 API

To add the secret:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GLM_API_KEY`
4. Value: Your GLM API key

### Optional Configuration

The following optional environment variables can be set in the workflow file:

| Variable | Default | Description |
|----------|---------|-------------|
| `GLM_API_URL` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | GLM API endpoint |
| `GLM_MODEL` | `glm-4.7` | Model to use for reviews |
| `PR_REVIEWER_COOLDOWN_MINUTES` | `5` | Cooldown period between reviews |
| `PR_REVIEWER_MAX_RETRIES` | `3` | Maximum API retry attempts |
| `PR_REVIEWER_RETRY_DELAY` | `1.0` | Base delay for retry backoff (seconds) |
| `PR_REVIEWER_MAX_DIFF_SIZE` | `100000` | Maximum diff size before truncation |

### Workflow Customization

The workflow can be customized in `.github/workflows/pr-reviewer.yml`:
- Change Python version (default: 3.10)
- Adjust timeout value (default: 10 minutes)
- Modify trigger conditions
- Add additional dependencies

## Code Quality

The PR reviewer script follows vllm-omni coding standards:

- **Type hints**: All functions have complete type hints following mypy strict mode
- **Logging**: Uses Python's logging module for structured logging
- **Testing**: Comprehensive unit tests with pytest
- **Pre-commit**: Script is checked by pre-commit hooks (black, isort, flake8)

## Contributing

To improve the PR reviewer bot:

1. Edit `.github/scripts/pr_reviewer.py` for logic changes
2. Edit `.github/workflows/pr-reviewer.yml` for workflow changes
3. Add tests to `.github/tests/test_pr_reviewer.py`
4. Run `pre-commit run --files .github/scripts/pr_reviewer.py` to check code quality
5. Test thoroughly with a test PR before deploying to production

## License

This bot is part of the VLLM-Omni project and follows the same license terms.
