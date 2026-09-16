#!/usr/bin/env bash
# Local reproduction of the CI gates before `git push` (ci.yml + the blocking
# jobs of security-audit.yml). Every command is the one CI runs; a step this
# script does not cover is a step that can only fail remotely
# (feedback_ci_local_verify_preflight_2026_09_15).
#
# usage: scripts/preflight.sh [--quick]   (--quick skips the test suites, fuzz build and semver)
set -euo pipefail
cd "$(dirname "$0")/.."

quick=0
[[ "${1:-}" == "--quick" ]] && quick=1
ALL_FEATURES='simd,media-stack,sync,physics,crypto,bincode-compat'
MSRV=1.87

step() { printf '\n\033[1;34m== %s\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

step "actionlint (workflow YAML)"
if have actionlint; then actionlint .github/workflows/*.yml; else echo "skip: actionlint not installed" >&2; fi

step "cargo fmt --check"
cargo fmt -- --check

step "clippy -D warnings (no default / docs.rs set / python bindings)"
cargo clippy --lib --all-targets --no-default-features -- -D warnings
cargo clippy --lib --all-targets --no-default-features --features "$ALL_FEATURES" -- -D warnings
cargo clippy --lib --no-default-features --features "python,$ALL_FEATURES" -- -D warnings

step "MSRV $MSRV (no default / docs.rs set)"
if rustup run "$MSRV" cargo --version >/dev/null 2>&1; then
  cargo "+$MSRV" check --lib --locked --no-default-features
  cargo "+$MSRV" check --lib --locked --no-default-features --features "$ALL_FEATURES"
else
  echo "skip: toolchain $MSRV not installed (rustup toolchain install $MSRV --profile minimal)" >&2
fi

step "feature powerset (cargo-hack, depth 2)"
if have cargo-hack; then
  cargo hack check --lib --feature-powerset --depth 2 \
    --exclude-features python,wasm,all-bridges,pyo3,numpy,wasm-bindgen,js-sys,web-sys
else echo "skip: cargo-hack not installed" >&2; fi

step "rustdoc -D warnings (docs.rs set)"
RUSTDOCFLAGS="-Dwarnings" cargo doc --lib --no-deps --no-default-features --features "$ALL_FEATURES"

step "security: audit / deny / machete / stub-guard"
if have cargo-audit; then cargo audit --deny yanked --ignore RUSTSEC-2025-0141; else echo "skip: cargo-audit not installed" >&2; fi
if have cargo-deny; then cargo deny --all-features check all; else echo "skip: cargo-deny not installed" >&2; fi
if have cargo-machete; then cargo machete; else echo "skip: cargo-machete not installed" >&2; fi
hits=$(grep -rnE 'todo!\(|unimplemented!\(|panic!\([^)]*STUB|dbg!\(' src/ --include="*.rs" || true)
[[ -z "$hits" ]] || { echo "stub / dbg residual:"; echo "$hits"; exit 1; }

if [[ $quick -eq 1 ]]; then
  echo; echo "preflight --quick: OK (test suites / fuzz build / semver skipped)"; exit 0
fi

step "tests (no default / docs.rs set / doctests)"
cargo test --lib --no-default-features
cargo test --lib --no-default-features --features "$ALL_FEATURES"
cargo test --test analytic_oracle --no-default-features
cargo test --test wire_roundtrip --no-default-features --features bincode-compat
cargo test --doc --no-default-features --features "$ALL_FEATURES"

step "fuzz targets build (nightly)"
if have cargo-fuzz && rustup run nightly cargo --version >/dev/null 2>&1; then
  (cd fuzz && cargo +nightly fuzz build)
else echo "skip: cargo-fuzz / nightly not installed" >&2; fi

step "semver-checks vs crates.io (docs.rs set)"
if have cargo-semver-checks; then
  cargo semver-checks check-release --package libasp --only-explicit-features --features "$ALL_FEATURES"
else echo "skip: cargo-semver-checks not installed" >&2; fi

echo; echo "preflight: OK"
