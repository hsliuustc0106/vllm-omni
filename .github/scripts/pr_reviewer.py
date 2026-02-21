#!/usr/bin/env python3
"""
VLLM-Omni PR Reviewer Bot

This bot provides automated code reviews using the GLM-4.7 API.
Triggered by commenting @vllm-omni-reviewer on a pull request.
"""

import os
import sys
import re
import json
import requests
from typing import Optional, Dict, Any

# Configuration
TRIGGER_PHRASE = "@vllm-omni-reviewer"
GLM_API_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
GLM_MODEL = "glm-4.7"

def log(message: str) -> None:
    """Log a message to stderr."""
    print(f"[PR Reviewer] {message}", file=sys.stderr, flush=True)

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error."""
    value = os.environ.get(name)
    if not value:
        log(f"Error: Environment variable {name} is not set")
        sys.exit(1)
    return value

def check_trigger(comment_body: str) -> bool:
    """Check if the comment contains the trigger phrase."""
    return TRIGGER_PHRASE in comment_body

def fetch_pr_diff(repo_name: str, pr_number: int, token: str) -> Optional[str]:
    """Fetch the diff for a pull request."""
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.diff"
    }

    log(f"Fetching PR diff from {url}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        diff = response.text
        log(f"Successfully fetched diff ({len(diff)} bytes)")
        return diff
    else:
        log(f"Failed to fetch PR diff: {response.status_code}")
        log(f"Response: {response.text}")
        return None

def fetch_pr_details(repo_name: str, pr_number: int, token: str) -> Optional[Dict[str, Any]]:
    """Fetch PR details including title and description."""
    url = f"https://api.github.com/repos/{repo_name}/pulls/{pr_number}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    log(f"Fetching PR details from {url}")
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        log(f"Failed to fetch PR details: {response.status_code}")
        return None

def build_review_prompt(pr_title: str, pr_description: str, diff: str) -> str:
    """Build the prompt for the GLM-4.7 API."""
    return f"""You are an expert code reviewer for the VLLM-Omni project. Please review the following pull request:

## Pull Request Details
**Title:** {pr_title}

**Description:**
{pr_description if pr_description else "No description provided."}

## Code Changes (Diff)
{diff}

## Review Guidelines

Please provide a comprehensive code review with the following sections:

### 1. Overview
- Brief summary of the changes
- Overall assessment (positive, neutral, or concerns)

### 2. Code Quality
- Code style and consistency
- Potential bugs or edge cases
- Performance considerations
- Error handling

### 3. Architecture & Design
- Integration with existing codebase
- Design patterns and best practices
- Potential improvements

### 4. Security & Safety
- Security concerns (if any)
- Resource management
- Input validation

### 5. Testing & Documentation
- Test coverage considerations
- Documentation completeness
- Examples and usage clarity

### 6. Specific Suggestions
- Line-by-line specific feedback (use `file:line` format)
- Concrete actionable suggestions
- Code examples for improvements (if applicable)

### 7. Approval Status
- **LGTM** (Looks Good To Me) if the PR is ready to merge
- **LGTM with suggestions** if the PR is good but has minor suggestions
- **Changes requested** if significant changes are needed

## Important Notes
- Be constructive and helpful
- Focus on objective technical feedback
- Acknowledge good practices when you see them
- Prioritize critical issues over nitpicks
- If the diff is empty or minimal, acknowledge this and provide any relevant context-specific guidance

Please format your response in Markdown with clear section headers.
"""

def call_glm_api(prompt: str, api_key: str) -> Optional[str]:
    """Call the GLM-4.7 API to get code review."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": GLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9
    }

    log(f"Calling GLM-4.7 API ({GLM_MODEL})")
    response = requests.post(GLM_API_URL, headers=headers, json=payload, timeout=60)

    if response.status_code == 200:
        data = response.json()
        try:
            review = data["choices"][0]["message"]["content"]
            log(f"Successfully received review ({len(review)} chars)")
            return review
        except (KeyError, IndexError) as e:
            log(f"Failed to parse API response: {e}")
            log(f"Response: {json.dumps(data, indent=2)}")
            return None
    else:
        log(f"GLM API request failed: {response.status_code}")
        log(f"Response: {response.text}")
        return None

def post_review_comment(repo_name: str, pr_number: int, token: str, review: str) -> bool:
    """Post the review as a comment on the PR."""
    url = f"https://api.github.com/repos/{repo_name}/issues/{pr_number}/comments"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    # Format the review comment
    comment_body = f"""## 🤖 VLLM-Omni PR Review

{review}

---
*This review was generated automatically by the VLLM-Omni PR Reviewer Bot using GLM-4.7.*
"""

    payload = {
        "body": comment_body
    }

    log(f"Posting review comment to PR #{pr_number}")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 201:
        log("Successfully posted review comment")
        return True
    else:
        log(f"Failed to post comment: {response.status_code}")
        log(f"Response: {response.text}")
        return False

def main() -> int:
    """Main entry point."""
    log("VLLM-Omni PR Reviewer Bot starting...")

    # Get environment variables
    token = get_env_var("GITHUB_TOKEN")
    api_key = get_env_var("GLM_API_KEY")
    repo_name = get_env_var("REPO_NAME")
    pr_number_str = get_env_var("PR_NUMBER")
    comment_body = get_env_var("COMMENT_BODY")

    try:
        pr_number = int(pr_number_str)
    except ValueError:
        log(f"Invalid PR number: {pr_number_str}")
        return 1

    log(f"Repository: {repo_name}")
    log(f"PR Number: {pr_number}")

    # Check if the comment contains the trigger phrase
    if not check_trigger(comment_body):
        log(f"Comment does not contain trigger phrase '{TRIGGER_PHRASE}', exiting")
        return 0

    log(f"Trigger phrase detected! Starting review process...")

    # Fetch PR details
    pr_details = fetch_pr_details(repo_name, pr_number, token)
    if not pr_details:
        log("Failed to fetch PR details")
        return 1

    pr_title = pr_details.get("title", "Unknown")
    pr_description = pr_details.get("body", "")

    log(f"PR Title: {pr_title}")

    # Fetch PR diff
    diff = fetch_pr_diff(repo_name, pr_number, token)
    if diff is None:
        log("Failed to fetch PR diff")
        return 1

    if not diff:
        log("Warning: Empty diff - this might be a draft PR or no code changes")

    # Build prompt
    prompt = build_review_prompt(pr_title, pr_description, diff)

    # Call GLM API
    review = call_glm_api(prompt, api_key)
    if not review:
        log("Failed to get review from GLM API")
        return 1

    # Post review comment
    if not post_review_comment(repo_name, pr_number, token, review):
        log("Failed to post review comment")
        return 1

    log("PR review completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
