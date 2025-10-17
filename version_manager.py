#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version Manager - Version management for Reranking & Embedding Service
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import date
from typing import List, Tuple


def get_current_version() -> Tuple[int, int, int]:
    """Get current version from version.py"""
    try:
        with open('version.py', 'r') as f:
            content = f.read()

        # Look for __version_info__ = (1, 0, 0) line
        match = re.search(r'__version_info__\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)', content)
        if match:
            return int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            print("❌ Unable to read current version from version.py")
            sys.exit(1)
    except FileNotFoundError:
        print("❌ File version.py not found")
        sys.exit(1)


def update_version_file(major: int, minor: int, patch: int) -> None:
    """Update version.py file with new version"""
    version_string = f"{major}.{minor}.{patch}"
    version_info = f"({major}, {minor}, {patch})"

    content = f'''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version information for Reranking & Embedding Service
"""

__version__ = "{version_string}"
__version_info__ = {version_info}

def get_version():
    """Get the current version string."""
    return __version__

def get_version_info():
    """Get version as tuple (major, minor, patch)."""
    return __version_info__'''

    with open('version.py', 'w') as f:
        f.write(content)

    print(f"✅ Version updated to {version_string}")


def bump_version(bump_type: str) -> Tuple[int, int, int]:
    """Bump version according to type (major, minor, patch)"""
    major, minor, patch = get_current_version()

    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
    elif bump_type == "minor":
        minor += 1
        patch = 0
    elif bump_type == "patch":
        patch += 1
    else:
        print(f"❌ Invalid version type: {bump_type}")
        sys.exit(1)

    return major, minor, patch


def print_current_version():
    """Display current version"""
    major, minor, patch = get_current_version()
    version_string = f"{major}.{minor}.{patch}"
    print(f"📋 Current version: v{version_string}")
    return version_string


def get_latest_tag() -> str:
    """Get the latest git tag"""
    try:
        result = subprocess.run(
            ['git', 'describe', '--tags', '--abbrev=0'],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None


def get_commits_since_tag(tag: str = None) -> List[str]:
    """Get commit messages since the last tag (or all commits if no tag)"""
    try:
        if tag:
            cmd = ['git', 'log', f'{tag}..HEAD', '--pretty=format:%s']
        else:
            cmd = ['git', 'log', '--pretty=format:%s']

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commits = result.stdout.strip().split('\n')
        return [c for c in commits if c]
    except Exception as e:
        print(f"⚠️  Unable to get git commits: {e}")
        return []


def categorize_commits(commits: List[str]) -> dict:
    """Categorize commits by type (feat, fix, chore, etc.)"""
    categories = {
        'Added': [],
        'Changed': [],
        'Fixed': [],
        'Deprecated': [],
        'Removed': [],
        'Security': []
    }

    for commit in commits:
        # Parse conventional commits format
        if commit.startswith('feat:') or commit.startswith('feat('):
            msg = re.sub(r'^feat(\([^)]+\))?:\s*', '', commit)
            categories['Added'].append(msg)
        elif commit.startswith('fix:') or commit.startswith('fix('):
            msg = re.sub(r'^fix(\([^)]+\))?:\s*', '', commit)
            categories['Fixed'].append(msg)
        elif commit.startswith('refactor:') or commit.startswith('refactor('):
            msg = re.sub(r'^refactor(\([^)]+\))?:\s*', '', commit)
            categories['Changed'].append(msg)
        elif commit.startswith('perf:') or commit.startswith('perf('):
            msg = re.sub(r'^perf(\([^)]+\))?:\s*', '', commit)
            categories['Changed'].append(msg)
        elif commit.startswith('docs:') or commit.startswith('docs('):
            msg = re.sub(r'^docs(\([^)]+\))?:\s*', '', commit)
            categories['Changed'].append(msg)
        elif commit.startswith('style:') or commit.startswith('style('):
            msg = re.sub(r'^style(\([^)]+\))?:\s*', '', commit)
            categories['Changed'].append(msg)
        elif commit.startswith('chore:') or commit.startswith('chore('):
            # Skip chore commits unless they're important
            if 'deprecat' in commit.lower():
                msg = re.sub(r'^chore(\([^)]+\))?:\s*', '', commit)
                categories['Deprecated'].append(msg)
        elif commit.startswith('security:') or 'security' in commit.lower():
            msg = re.sub(r'^security(\([^)]+\))?:\s*', '', commit)
            categories['Security'].append(msg)
        elif not commit.startswith('chore'):
            # Uncategorized commits go to Changed
            categories['Changed'].append(commit)

    # Remove empty categories
    return {k: v for k, v in categories.items() if v}


def generate_changelog_entry(version: str, commits: List[str]) -> str:
    """Generate a changelog entry from commits"""
    today = date.today().strftime("%Y-%m-%d")
    categories = categorize_commits(commits)

    if not categories:
        print("⚠️  No commits found to generate changelog")
        return None

    entry = f"## [{version}] - {today}\n\n"

    for category, items in categories.items():
        entry += f"### {category}\n"
        for item in items:
            entry += f"- {item}\n"
        entry += "\n"

    return entry


def update_changelog(version: str) -> bool:
    """Update CHANGELOG.md with new version entry"""
    changelog_path = "CHANGELOG.md"

    # Get commits since last tag
    last_tag = get_latest_tag()
    print(f"📋 Last tag: {last_tag if last_tag else 'none (first release)'}")

    commits = get_commits_since_tag(last_tag)
    if not commits:
        print("⚠️  No new commits found since last tag")
        return False

    print(f"📝 Found {len(commits)} commits since {last_tag if last_tag else 'repository creation'}")

    # Generate changelog entry
    new_entry = generate_changelog_entry(version, commits)
    if not new_entry:
        return False

    # Read existing changelog or create new one
    if os.path.exists(changelog_path):
        with open(changelog_path, 'r') as f:
            content = f.read()
    else:
        content = """# Changelog

All notable changes to the Reranking & Embedding Service will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

"""

    # Insert new entry after the header
    lines = content.split('\n')
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('## '):
            header_end = i
            break

    if header_end == 0:
        # No existing versions, append at end
        new_content = content.rstrip() + '\n\n' + new_entry
    else:
        # Insert before first version
        new_content = '\n'.join(lines[:header_end]) + '\n' + new_entry + '\n'.join(lines[header_end:])

    # Write updated changelog
    with open(changelog_path, 'w') as f:
        f.write(new_content)

    print(f"✅ CHANGELOG.md updated with version {version}")
    print(f"\n📄 Generated changelog entry:")
    print(f"─" * 50)
    print(new_entry)
    print(f"─" * 50)

    return True


def print_git_commands(version_string: str, commit_message: str = None, changelog_updated: bool = False):
    """Display git commands to execute"""
    if not commit_message:
        commit_message = f"chore: bump version to v{version_string}"

    tag_name = f"v{version_string}"

    print(f"\n🔧 Git commands to execute:")
    print(f"─" * 50)
    print(f"# 1. Add changes")
    if changelog_updated:
        print(f"git add version.py CHANGELOG.md")
    else:
        print(f"git add version.py")
    print(f"")
    print(f"# 2. Create commit")
    print(f'git commit -m "{commit_message}"')
    print(f"")
    print(f"# 3. Create tag")
    print(f'git tag -a {tag_name} -m "Release {tag_name}"')
    print(f"")
    print(f"# 4. Push to remote (with tags)")
    print(f"git push origin")
    print(f"git push origin {tag_name}")
    print(f"")
    print(f"💡 Or in a single command:")
    print(f"git push origin --follow-tags")
    print(f"─" * 50)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Version manager for Reranking & Embedding Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:
  # Display current version
  python version_manager.py current

  # Bump patch version (1.0.0 → 1.0.1)
  python version_manager.py bump patch

  # Bump minor version (1.0.1 → 1.1.0)
  python version_manager.py bump minor

  # Bump major version (1.1.0 → 2.0.0)
  python version_manager.py bump major

  # Set specific version
  python version_manager.py set 2.1.3
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Current command
    subparsers.add_parser('current', help='Display current version')

    # Bump command
    bump_parser = subparsers.add_parser('bump', help='Bump version')
    bump_parser.add_argument('type', choices=['major', 'minor', 'patch'],
                           help='Version type to bump')
    bump_parser.add_argument('--message', '-m',
                           help='Custom commit message')

    # Set command
    set_parser = subparsers.add_parser('set', help='Set specific version')
    set_parser.add_argument('version',
                          help='Version to set (format: X.Y.Z)')
    set_parser.add_argument('--message', '-m',
                          help='Custom commit message')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Check we're in the right directory
    if not os.path.exists('version.py'):
        print("❌ Error: version.py not found. Run this script from the project root.")
        sys.exit(1)

    if args.command == 'current':
        print_current_version()

    elif args.command == 'bump':
        current_version = print_current_version()
        new_major, new_minor, new_patch = bump_version(args.type)
        new_version = f"{new_major}.{new_minor}.{new_patch}"

        print(f"🔄 Update: v{current_version} → v{new_version}")

        # Confirmation
        response = input(f"\n❓ Confirm update to v{new_version}? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Operation cancelled")
            sys.exit(0)

        # Update
        update_version_file(new_major, new_minor, new_patch)
        changelog_updated = update_changelog(new_version)
        print_git_commands(new_version, args.message, changelog_updated)

    elif args.command == 'set':
        # Validate version format
        match = re.match(r'^(\d+)\.(\d+)\.(\d+)$', args.version)
        if not match:
            print(f"❌ Invalid version format: {args.version}")
            print("💡 Use format: X.Y.Z (example: 2.1.3)")
            sys.exit(1)

        new_major, new_minor, new_patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        current_version = print_current_version()

        print(f"🔄 Update: v{current_version} → v{args.version}")

        # Confirmation
        response = input(f"\n❓ Confirm update to v{args.version}? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            print("❌ Operation cancelled")
            sys.exit(0)

        # Update
        update_version_file(new_major, new_minor, new_patch)
        changelog_updated = update_changelog(args.version)
        print_git_commands(args.version, args.message, changelog_updated)


if __name__ == "__main__":
    main()