#!/bin/bash
set -e

# Install system deps
apt-get update && apt-get install -y cmake bear patchelf clang-format clangd

# Install rocprof-trace-decoder
echo "Installing rocprof-trace-decoder..."
mkdir -p ~/rocprofiler-setup
cd ~/rocprofiler-setup

wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.1/rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb
dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.1-Linux.deb

git clone https://github.com/ROCm/aqlprofile.git
cd aqlprofile
./build.sh
cd build
make install
cd ../../..
rm -rf rocprofiler-setup

# Install claude code
curl -fsSL https://claude.ai/install.sh | bash -s stable

mkdir -p ~/.claude && chmod 700 ~/.claude

cat > ~/.claude/settings.json <<EOF
{
  "\$schema": "https://json.schemastore.org/claude-code-settings.json",
  "env": {
    "ANTHROPIC_API_KEY": "dummy",
    "ANTHROPIC_BASE_URL": "https://llm-api.amd.com/Anthropic",
    "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: <LLM_API_KEY>",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "claude-haiku-4.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4.6",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4.7",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
    "ENABLE_TOOL_SEARCH": "true"
  }
}
EOF