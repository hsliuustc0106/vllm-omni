#!/usr/bin/env python3
"""
PR Reviewer using GLM API for vllm-omni project.
"""
import os
import sys
from pathlib import Path

import httpx
from github import Github

# Configuration
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
MODEL = "glm-4.7"  # GLM-4.7 model

def get_pr_diff(repo_name: str, pr_number: int, github_token: str) -> str:
    """Fetch PR diff using GitHub API."""
    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)
    return pr.diff()

def get_pr_context(repo_name: str, pr_number: int, github_token: str) -> dict:
    """Fetch PR metadata and context."""
    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Get changed files
    files = []
    for file in pr.get_files():
        files.append({
            "filename": file.filename,
            "status": file.status,
            "additions": file.additions,
            "deletions": file.deletions,
            "patch": file.patch[:5000] if file.patch else ""  # Truncate for context
        })

    return {
        "number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "author": pr.user.login,
        "base": pr.base.ref,
        "head": pr.head.ref,
        "files": files
    }

def call_glm_api(prompt: str, api_key: str) -> str:
    """Call GLM API for code review."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": """You are an elite code reviewer specializing in the vllm-omni project - a multi-modal inference system built on top of vLLM.

Your expertise includes:
- vLLM architecture (PagedAttention, continuous batching, tensor parallelism)
- Multi-modal AI systems (audio, vision, text processing)
- PyTorch, CUDA programming, GPU optimization
- Real-time streaming inference and low-latency systems
- WebSocket protocols and async/await in Python

Review the PR changes and provide structured feedback in the following format:

## PR Review Summary

### Overview
[Brief summary of the PR's purpose]

### Critical Issues (Must Fix)
- [ ] Issue description with file:line references

### Important Issues (Should Fix)
- [ ] Issue description

### Minor Issues & Suggestions
- [ ] Minor issue or suggestion

### Positive Aspects
- [ ] Well-implemented feature

### Performance Considerations
[Analysis of performance impact]

### Testing Recommendations
[Suggestions for additional tests]

### Overall Assessment
[Your recommendation: Approve / Request Changes / Needs Major Work]"""
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 16000
    }

    with httpx.Client(timeout=120.0) as client:
        response = client.post(GLM_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]

def post_review_comment(repo_name: str, pr_number: int, body: str, github_token: str):
    """Post review as a comment on the PR."""
    gh = Github(github_token)
    repo = gh.get_repo(repo_name)
    pr = repo.get_pull(pr_number)

    # Create issue comment
    pr.create_issue_comment(body)

def main():
    api_key = os.environ.get("GLM_API_KEY")
    github_token = os.environ.get("GITHUB_TOKEN")
    pr_number_str = os.environ.get("PR_NUMBER")
    repo_name = os.environ.get("REPO")

    # Debug: print environment variables status
    print(f"DEBUG: GLM_API_KEY={'***' if api_key else 'MISSING'}")
    print(f"DEBUG: GITHUB_TOKEN={'***' if github_token else 'MISSING'}")
    print(f"DEBUG: PR_NUMBER={pr_number_str}")
    print(f"DEBUG: REPO={repo_name}")

    # Validate all required variables
    if not api_key:
        print("Error: GLM_API_KEY environment variable is missing")
        sys.exit(1)
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is missing")
        sys.exit(1)
    if not pr_number_str:
        print("Error: PR_NUMBER environment variable is missing")
        sys.exit(1)
    if not repo_name:
        print("Error: REPO environment variable is missing")
        sys.exit(1)

    try:
        pr_number = int(pr_number_str)
    except ValueError:
        print(f"Error: PR_NUMBER is not a valid integer: {pr_number_str}")
        sys.exit(1)

    print(f"Reviewing PR #{pr_number} in {repo_name}...")

    # Get PR diff and context
    print("Fetching PR details...")
    diff = get_pr_diff(repo_name, pr_number, github_token)
    context = get_pr_context(repo_name, pr_number, github_token)

    # Build prompt for GLM
    prompt = f"""Review this pull request:

**PR Title**: {context['title']}
**Author**: {context['author']}
**Branch**: {context['head']} -> {context['base']}

**Description**:
{context['body'] or 'No description provided'}

**Files Changed**:
{chr(10).join(f"- {f['filename']} ({f['status']}: +{f['additions']} -{f['deletions']})" for f in context['files'])}

**Full Diff**:
{diff[:50000]}  # Truncate if too large

Please review this PR considering vLLM architecture, multi-modal integration patterns, performance, and code quality."""

    # Call GLM API
    print("Calling GLM API for review...")
    try:
        review = call_glm_api(prompt, api_key)

        # Save to file
        output_path = Path("/tmp/review_output.md")
        output_path.write_text(review)

        # Post as comment
        print("Posting review comment...")
        post_review_comment(repo_name, pr_number, review, github_token)

        print("Review completed successfully!")

    except Exception as e:
        print(f"Error during review: {e}")
        # Post error comment
        gh = Github(github_token)
        repo = gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        pr.create_issue_comment(f" PR Reviewer encountered an error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
