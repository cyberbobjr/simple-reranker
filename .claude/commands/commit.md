---
description: Generate a conventional commit message from git changes and create the commit
---

Analyze the current git changes and generate a conventional commit message following the Conventional Commits format.

## Instructions:

1. **Get the git diff and changed files:**
   - Run `git diff --cached --name-status` to see staged files
   - Run `git diff --cached` to get the full diff
   - If no staged changes, check unstaged with `git diff --name-status` and `git diff`

2. **Analyze the changes:**
   - Review what files were modified, added, or deleted
   - Understand the nature of the changes from the diff
   - Identify the primary purpose of the changes

3. **Generate a conventional commit message:**
   - Use format: `<type>(<scope>): <description>`
   - Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`
   - Scope is optional but useful (e.g., api, auth, config, docker, ci)
   - Description: lowercase, concise (max 72 chars), no period at end
   - Examples:
     - `feat(api): add health check endpoint`
     - `fix(auth): resolve token validation bug`
     - `docs: update installation instructions`
     - `refactor(docker): simplify build configuration`
     - `chore: bump version to v1.2.1`

4. **Present the message and create the commit:**
   - Show the generated commit message
   - Explain briefly what changes it captures
   - Ask user to confirm
   - If confirmed, run `git commit -m "<message>"`
   - If user wants to modify it, let them provide a custom message

## Important:
- If no changes are staged, inform the user to stage changes first with `git add`
- Focus on the MOST IMPORTANT change if there are multiple unrelated changes
- Suggest splitting into multiple commits if changes are too diverse
- Always ensure the message is clear, concise, and follows conventional commits format
