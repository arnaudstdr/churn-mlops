#!/bin/sh

git config core.hooksPath .githooks
chmod -R +x .githooks

echo "✅ Hooks Git installés via .githooks/"
