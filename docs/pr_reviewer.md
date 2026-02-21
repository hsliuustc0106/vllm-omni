# VLLM-Omni PR Reviewer Bot

## Overview

The VLLM-Omni PR Reviewer Bot is an automated code reviewer that uses the GLM-4.7 API to provide intelligent, comprehensive code reviews for pull requests in the vllm-omni repository.

## Features

- **Intelligent Code Analysis**: Leverages GLM-4.7 for understanding code context and providing meaningful feedback
- **Comprehensive Reviews**: Covers code quality, architecture, security, testing, and documentation
- **Structured Output**: Provides well-formatted reviews with clear sections and actionable suggestions
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
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  PR Reviewer Script             │
│  (.github/scripts/              │
│   pr_reviewer.py)               │
│                                 │
│  1. Fetch PR details & diff     │
│  2. Build review prompt         │
│  3. Call GLM-4.7 API            │
│  4. Post review comment         │
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

### What to Look For

When testing, verify that:
- [ ] The workflow triggers on the `@vllm-omni-reviewer` comment
- [ ] The GLM API call completes without errors
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
3. **Check API key** - Ensure `GLM_API_KEY` is configured in repository secrets

### Permission Denied Error

If you see "Permission denied" or similar:
- Only repository members/collaborators/owners can trigger reviews
- Contact a repository maintainer for access

### API Errors

If the GLM API call fails:
- Check the Actions tab for detailed error logs
- Verify the `GLM_API_KEY` secret is correctly configured
- Ensure sufficient API quota is available

## Configuration

### Required Secrets

The following secret must be configured in the repository settings:

- `GLM_API_KEY` - Your GLM (BigModel) API key for accessing the GLM-4.7 API

To add the secret:
1. Go to repository Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Name: `GLM_API_KEY`
4. Value: Your GLM API key

### Workflow Customization

The workflow can be customized in `.github/workflows/pr-reviewer.yml`:
- Change Python version
- Add additional dependencies
- Adjust timeout values
- Modify trigger conditions

## Contributing

To improve the PR reviewer bot:

1. Edit `.github/scripts/pr_reviewer.py` for logic changes
2. Edit `.github/workflows/pr-reviewer.yml` for workflow changes
3. Test thoroughly with a test PR before deploying to production

## License

This bot is part of the VLLM-Omni project and follows the same license terms.
