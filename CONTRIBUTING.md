# Contributing

Thanks for looking. A few things about how this repository works will save
you time.

## This repo is a snapshot

The code here is exported from a private repository where the maintainers
run their own fleet. Every commit on `main` is one export: a redacted copy of
an allow-list of paths. That has two consequences for you:

- **Pull requests are welcome and are merged here first.** After a merge, a
  maintainer imports the change into the private side, and the next export
  brings it back out. Your commit stays in this repo's history; the export
  after it will re-land the same content as part of a `sync` commit.
- **Do not expect to see the maintainers' day-to-day commits.** History here
  is a series of `sync <date>` commits, not a development log. The design
  notes in `docs/` are the record of why things are the way they are.

## Before you open a PR

```bash
pip install -r gateway/requirements.txt -r gateway/requirements-dev.txt
ruff check .
python -m pytest gateway/tests fleetctl/tests -q
./install.sh --dry-run          # or .\install.ps1 -DryRun on Windows
```

CI runs the same on Ubuntu, macOS and Windows, and `install.sh` in a container
per package family (Debian, Ubuntu, Fedora, Arch, openSUSE). All of it runs on
GitHub-hosted runners; this repository never registers a self-hosted runner.

## What makes a change easy to merge

- **One thing per PR**, with the failure it fixes or the machine it adds
  support for stated in the description. A `fleetctl detect --json` from the
  box in question is worth more than prose.
- **Keep `fleetctl/` stdlib-only.** It runs before anything is installed; a
  dependency there is a dependency on the bootstrap.
- **`check()` must be honest.** A step that cannot tell reports `blocked`, not
  `ok` and not `missing`. `--dry-run` must change nothing, including on the
  network.
- **Platform quirks get a comment with the measurement.** Most of the odd
  lines in this codebase exist because of something a real machine did; say
  what yours did.
- **No real hosts, addresses, or accounts** in code, tests, or docs. Use
  `example.com`, `10.0.0.x`, `admin@example.com`, and generic host names.
  The export gate on the private side will refuse anything that looks like
  a real one, so a PR that carries one cannot be re-exported.

## Reporting a security issue

See [SECURITY.md](SECURITY.md). Please do not open a public issue for
anything that could expose a running gateway.

## License

By contributing you agree that your contributions are licensed under the
Apache License 2.0, the same as the rest of the project.
