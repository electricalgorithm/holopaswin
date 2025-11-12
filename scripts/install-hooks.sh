#!/bin/bash
#
# Install git hooks from .githooks/ to .git/hooks/
# This script sets up the pre-commit hook for code quality checks

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

GITHOOKS_DIR=".githooks"
GIT_HOOKS_DIR=".git/hooks"

# Check if .git directory exists
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Warning: .git directory not found. Are you in the repository root?${NC}"
    exit 1
fi

# Check if .githooks directory exists
if [ ! -d "$GITHOOKS_DIR" ]; then
    echo -e "${YELLOW}Error: $GITHOOKS_DIR directory not found.${NC}"
    exit 1
fi

# Create .git/hooks directory if it doesn't exist
mkdir -p "$GIT_HOOKS_DIR"

# Install hooks
echo -e "${GREEN}Installing git hooks...${NC}"

for hook in "$GITHOOKS_DIR"/*; do
    if [ -f "$hook" ] && [ -x "$hook" ]; then
        hook_name=$(basename "$hook")
        target="$GIT_HOOKS_DIR/$hook_name"
        
        # Remove existing hook if it's a symlink or file
        if [ -L "$target" ] || [ -f "$target" ]; then
            echo "Removing existing $hook_name hook..."
            rm "$target"
        fi
        
        # Create symlink to the hook in .githooks
        ln -s "../../$GITHOOKS_DIR/$hook_name" "$target"
        echo -e "${GREEN}✓ Installed $hook_name${NC}"
    fi
done

echo -e "\n${GREEN}Git hooks installed successfully!${NC}"
echo "Hooks will run automatically on git commit."

