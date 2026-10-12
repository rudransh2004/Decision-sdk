# Changelog

All notable changes to this project are documented in this file.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- The package ships a `py.typed` marker, so type checkers read its inline annotations.
- The `dev` extra installs `fastapi`, `httpx`, `ruff`, `mypy`, and `types-PyYAML`.
  The serving tests now run under `pip install -e ".[dev]"` instead of skipping.

### Changed

- The package version comes from `futurix_decision.__version__` alone.
  The `futurix-decision serve` API reports that version instead of a hard-coded `0.1.0`.
