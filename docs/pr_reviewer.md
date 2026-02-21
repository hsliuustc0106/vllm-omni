# VLLM-Omni PR Reviewer

## Overview

The VLLM-Omni PR Reviewer is an automated code review bot powered by GLM-4.7 AI model. It helps maintain code quality by providing intelligent feedback on pull requests.

## Usage

### Triggering a Review

To trigger an automated PR review, mention the bot in a PR comment:

```
@vllm-omni-reviewer please review
```

Or include in your PR description:

```
@vllm-omni-reviewer
```

The bot will automatically review your changes and post a detailed comment.

## What Gets Reviewed

- **vLLM Architecture Compatibility**: Ensures changes align with vLLM's design patterns
- **Multi-modal Integration**: Reviews audio, vision, and text processing implementations
- **Performance Implications**: Analyzes impact on inference latency and throughput
- **Code Quality**: Checks Python best practices, type hints, and documentation
- **Security Considerations**: Identifies potential security vulnerabilities
- **Testing Coverage**: Recommends additional test cases when needed

## Review Output

The bot posts a structured review comment with:

- **Overview**: Brief summary of the PR's purpose
- **Critical Issues (Must Fix)**: Blocking issues that need to be addressed
- **Important Issues (Should Fix)**: Significant concerns that should be resolved
- **Minor Issues & Suggestions**: Small improvements and optional suggestions
- **Positive Aspects**: Highlights well-implemented features
- **Performance Considerations**: Analysis of performance impact
- **Testing Recommendations**: Suggestions for additional tests
- **Overall Assessment**: Final recommendation (Approve/Request Changes/Needs Major Work)

## Architecture

The PR Reviewer consists of:

1. **GitHub Actions Workflow** (`.github/workflows/pr-reviewer.yml`): Triggers on @mention
2. **Python Script** (`.github/scripts/pr_reviewer.py`): Fetches PR data and calls GLM-4.7 API
3. **GLM-4.7 API**: Provides intelligent code analysis

## Setup for Contributors

To set up the PR reviewer in your own fork:

1. Get a GLM API key from [https://open.bigmodel.cn/usercenter/apikeys](https://open.bigmodel.cn/usercenter/apikeys)
2. Add `GLM_API_KEY` as a repository secret in your GitHub settings
3. Copy the workflow file and script to your repository

## Troubleshooting

**Bot doesn't respond**: Ensure you're a repository member (not an external user)

**Review fails**: Check the Actions tab for error logs

**Incomplete review**: Large PRs may have truncated diffs - consider splitting into smaller PRs

## Cost Estimate

| Component | Cost |
|-----------|------|
| GitHub Actions (public repo) | Free |
| GLM API (glm-4.7) | ~0.50-5 CNY per PR (varies by size) |
| Total (20 PRs/month) | ~10-100 CNY/month (~$2-15 USD) |
