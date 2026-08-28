# Example host plans

`fleetctl plan --write` generates `hosts/<name>/host.yml` for the box it runs
on; these are what such files look like for four common shapes, with real
values replaced by documentation ones. Copy one to `hosts/<name>/host.yml`,
edit, and `python -m fleetctl apply --dry-run --host <name>` to see what it
would do.

| file | shape | engine | for |
|---|---|---|---|
| `hub.host.yml` | Linux, systemd | none | an always-on box that routes and does not serve |
| `gpu-desktop.host.yml` | Windows, scheduled task | llama.cpp + llama-swap, Vulkan | a gaming PC that lends its idle hours |
| `mac-laptop.host.yml` | macOS, cron | llama.cpp + llama-swap, Metal | an Apple Silicon laptop with a Metal ceiling |
| `cpu-box.host.yml` | Linux, systemd | Ollama, CPU | a small always-on fallback |

Every key is explained in [../usage.md](../usage.md#the-host-plan); the
layering (shape, `fleet.yml`, `host.yml`, command line) is in
[../fleetctl.md](../fleetctl.md).
