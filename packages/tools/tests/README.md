# Babylon.js Test Infrastructure

## Overview

This package contains the Playwright-based test suites for Babylon.js:

- **Visualization tests** — pixel-comparison screenshots for WebGL2 and WebGPU
- **Interaction tests** — user interaction scenarios (click, drag, etc.)
- **Performance tests** — rendering benchmark measurements
- **Integration tests** — end-to-end integration checks

## Running Tests Locally

### BabylonNative visualization tests

Build BabylonNative's Win32 `Playground` target with the Playwright bridge,
then point the native Playwright project at the executable:

```powershell
$env:BABYLON_NATIVE_PLAYGROUND = "C:\path\to\BabylonNative\Build\Apps\Playground\Release\Playground.exe"
# Optional identity for granular excludedEngines matching:
$env:BABYLON_NATIVE_GRAPHICS_API = "d3d11"   # d3d11 | d3d12 | vk | gl | gles | metal | ...
$env:BABYLON_NATIVE_PLATFORM = "win32"       # win32 | uwp | android | ios | osx | unix | ...
npx playwright test --config playwright.config.ts --project native
```

Native uses the **same** goldens as browser visualization tests under
`test/visualization/ReferenceImages/` — native output is expected to match web.

The native project defaults to zero test retries (`--retries` can override this).
Native scene-construction errors fail immediately rather than evaluating the same
scene repeatedly; transient asset-request retries are unchanged. Browser projects
retain their existing retry behavior. Failed native tests terminate their host
before Playwright replaces the worker, while successful tests reuse the engine.
Native startup is lazy per worker inside `beforeEach`, so a host startup crash
fails the current test instead of skipping the rest of a `beforeAll` group.
Shutdown waits at most five seconds before forcing termination, then allows five
seconds to observe exit; shutdown failures are reported, not silently ignored.

For timing comparisons, keep the catalog, executable, assets, worker count and
goldens fixed, and retain a structured report:

```powershell
$env:PLAYWRIGHT_JSON_OUTPUT_NAME = "$PWD\native-results.json"
npx playwright test --config playwright.config.ts --project native --workers 4 --retries 0 --update-snapshots=none --trace off --reporter=line,json
```

The report includes engine initialization, snippet fetch, scene preparation,
rendering, PNG capture and comparison steps. Count actual result statuses;
unexecuted cases are not passes. Native and legacy validation timings are not
directly comparable when their catalogs, exclusions or reference images differ.

#### `excludedEngines` for native (combinable, OR)

| Tag | Meaning |
|---|---|
| `native` | all native backends/platforms |
| `native-d3d11` / `native-d3d12` / `native-vk` / `native-gl` / `native-gles` / `native-metal` | one graphics API |
| `native-win32` / `native-uwp` / `native-android` / `native-ios` / `native-osx` / `native-unix` | one platform |
| `native-win32-d3d11` (etc.) | compound platform+API |

Examples in `config.json`:

```json
"excludedEngines": ["native"]
"excludedEngines": ["native-d3d11", "native-vk"]
"excludedEngines": ["webgpu", "native-gles"]
"excludedEngines": ["native-android-gles", "native-ios-metal"]
```

### Visualization tests (local browser)

```bash
# All WebGL2 tests using local Chrome
npm run test:visualization -- --project=webgl2

# A single test
npm run test:visualization -- --project=webgl2 --grep "Particle system"

# WebGPU tests
npm run test:visualization -- --project=webgpu
```

### Visualization tests (BrowserStack)

Tests can run on BrowserStack infrastructure using Playwright's
`connectOptions.wsEndpoint` — a direct CDP WebSocket connection to a remote
BrowserStack browser session. Both `ci-monorepo` and `ci-browser-testing`
use this approach.

**Prerequisites:**

1. Add your credentials to `.env` at the repo root:
    ```
    BROWSERSTACK_USERNAME=your_username
    BROWSERSTACK_ACCESS_KEY=your_access_key
    ```
2. Run `npm install`

**Commands:**

```bash
# WebGL2 on BrowserStack (Chrome/macOS by default)
CDN_BASE_URL="https://cdn.babylonjs.com" BSTACK_TEST_TYPE=webgl2 \
  npx playwright test --config ./playwright.browserstack.config.ts

# Single test
CDN_BASE_URL="https://cdn.babylonjs.com" BSTACK_TEST_TYPE=webgl2 \
  npx playwright test --config ./playwright.browserstack.config.ts \
  --grep "Particle system"

# WebGPU on BrowserStack
CDN_BASE_URL="https://cdn.babylonjs.com" BSTACK_TEST_TYPE=webgpu \
  npx playwright test --config ./playwright.browserstack.config.ts

# Override browser (e.g. Firefox)
CDN_BASE_URL="https://cdn.babylonjs.com" BSTACK_TEST_TYPE=webgl2 \
  BSTACK_BROWSER=playwright-firefox BSTACK_OS="OS X" BSTACK_OS_VERSION=Sonoma \
  npx playwright test --config ./playwright.browserstack.config.ts

# Test against a local dev server (requires BrowserStack Local tunnel running separately)
CDN_BASE_URL="http://localhost:1337" BSTACK_TEST_TYPE=webgl2 \
  npx playwright test --config ./playwright.browserstack.config.ts
```

### Environment Variables

| Variable                        | Description                                                                         | Default                         |
| ------------------------------- | ----------------------------------------------------------------------------------- | ------------------------------- |
| `BSTACK_TEST_TYPE`              | Test suite to run: `webgl2`, `webgpu`, `performance`, `interaction`, `es6vis`       | `webgl2`                        |
| `CDN_BASE_URL`                  | Base URL where Babylon.js snapshot is served                                        | —                               |
| `CIWORKERS`                     | Number of parallel Playwright workers / BrowserStack sessions                       | `5` (BrowserStack), `4` (local) |
| `BSTACK_BROWSER`                | Override BrowserStack browser (e.g. `playwright-firefox`, `playwright-webkit`)      | `chrome`                        |
| `BSTACK_OS`                     | Override BrowserStack OS                                                            | `OS X`                          |
| `BSTACK_OS_VERSION`             | Override BrowserStack OS version                                                    | `Sonoma`                        |
| `BSTACK_BUILD_NAME`             | Override build name on BrowserStack dashboard                                       | Auto-generated from test type   |
| `BROWSERSTACK_LOCAL`            | Enable BrowserStack Local tunnel startup for local dev server suites                | —                               |
| `BROWSERSTACK_LOCAL_IDENTIFIER` | Unique BrowserStack Local tunnel identifier for local-tunnel suites such as ES6 vis | —                               |

| `EXCLUDE_REGEX_ARRAY` | Comma-separated regex patterns to exclude tests | — |
| `TIMEOUT` | Per-test timeout in milliseconds | Playwright default |
| `SCREENSHOT_THRESHOLD` | Pixel comparison threshold (0-1) | `0.05` |
| `SCREENSHOT_MAX_PIXEL` | Maximum allowed pixel difference | `5` |

### BrowserStack Dashboard

- **Automate sessions**: https://automate.browserstack.com
- **Test Observability** (per-test results): https://observability.browserstack.com

## Configuration Files

| File                                          | Purpose                                                      |
| --------------------------------------------- | ------------------------------------------------------------ |
| `playwright.config.ts`                        | Main config for local and legacy CDP-based BrowserStack runs |
| `playwright.browserstack.config.ts`           | Config for BrowserStack CDP runs (all pipelines)             |
| `playwright.devhost.config.ts`                | Config for dev host tests (lottie, etc.)                     |
| `packages/tools/tests/playwright.utils.ts`    | Shared project definitions and browser setup                 |
| `packages/tools/tests/browserstack.config.ts` | Legacy CDP endpoint configuration                            |

---

## Performance Tests

Performance tests measure per-frame rendering time by comparing two versions of the
engine (baseline vs candidate) on the same BrowserStack machine. Each test loads a
visualization scene twice — once with each version — and runs multiple interleaved
passes so that any VM-level variance affects both equally.

### How It Works

1. **Warmup** — adaptive warmup until frame times stabilize (last 2 values within 5%).
2. **Interleaved passes** — alternates between baseline and candidate pages, rendering
   `framesToRender` frames per pass. This cancels out thermal throttling and VM noise.
3. **Statistical analysis** — paired t-test on the interleaved samples. Results are
   classified as passed (within threshold), failed (statistically significant regression),
   or inconclusive (too noisy to tell).
4. **Confirmation** — failed results trigger additional passes to reduce false positives.
5. **Retry** — inconclusive results trigger additional sampling when time allows.

### Running Performance Tests Locally (BrowserStack)

**Prerequisites:** BrowserStack credentials in `.env` (see above).

```bash
# Source credentials
export $(grep -v '^#' .env | xargs)

# Run a single test (WebGL2)
BSTACK_TEST_TYPE=performance \
  VISUALIZATION_PERF=true \
  CDN_VERSION=9.0.0 CDN_VERSION_B=9.3.3 \
  npx playwright test \
    --config ./playwright.browserstack.config.ts \
    packages/tools/tests/test/performance/visualization.test.ts \
    -g "Shadows (webgl2)"

# Run all visualization perf tests (WebGL2 only)
BSTACK_TEST_TYPE=performance \
  VISUALIZATION_PERF=true VISUALIZATION_PERF_ALL=true \
  CDN_VERSION=9.0.0 CDN_VERSION_B=9.3.3 \
  npx playwright test \
    --config ./playwright.browserstack.config.ts \
    packages/tools/tests/test/performance/visualization.test.ts \
    -g "webgl2"

# Run all engines
BSTACK_TEST_TYPE=performance \
  VISUALIZATION_PERF=true VISUALIZATION_PERF_ALL=true \
  CDN_VERSION=9.0.0 CDN_VERSION_B=9.3.3 \
  npx playwright test \
    --config ./playwright.browserstack.config.ts \
    packages/tools/tests/test/performance/visualization.test.ts
```

### Performance-Specific Environment Variables

| Variable                 | Description                                                          | Default             |
| ------------------------ | -------------------------------------------------------------------- | ------------------- |
| `VISUALIZATION_PERF`     | Set to `true` to enable performance test suite                       | — (tests skip)      |
| `VISUALIZATION_PERF_ALL` | Set to `true` to run ALL visualization tests as perf tests           | `false`             |
| `CDN_VERSION`            | Baseline CDN version (e.g. `9.0.0`)                                  | — (required)        |
| `CDN_VERSION_B`          | Candidate CDN version (e.g. `9.3.3`). Omit to test against dev build | —                   |
| `CIWORKERS`              | Playwright parallel workers (= concurrent BrowserStack sessions)     | `5`                 |
| `BROWSERSTACK_PARALLELS` | BrowserStack max concurrent sessions per platform                    | `10`                |
| `BSTACK_BUILD_NAME`      | Custom build name on BrowserStack dashboard                          | `Performance Tests` |

### Config Flags (config.json)

| Flag                             | Description                                                                              |
| -------------------------------- | ---------------------------------------------------------------------------------------- |
| `"performanceTest": true`        | Includes this test in performance runs (used when `VISUALIZATION_PERF_ALL` is not set)   |
| `"excludeFromPerformance": true` | Permanently excludes this test from performance runs, even with `VISUALIZATION_PERF_ALL` |

### Tuning Parallelism

- **`CIWORKERS`** — how many test files Playwright runs concurrently. Each worker gets
  its own BrowserStack VM, so parallel tests don't share hardware.
- **`BROWSERSTACK_PARALLELS`** — BrowserStack account-level cap per build. Set this to
  your plan maximum; use `CIWORKERS` to dial actual concurrency per job.
- Both should satisfy `CIWORKERS ≤ BROWSERSTACK_PARALLELS ≤ plan limit`.
- For performance tests, interleaved A/B design means parallelism is safe — each
  worker's measurements are self-contained on a single VM.

### CI Integration

Performance tests run in the `PerformanceTests` job in the Azure DevOps pipeline
(`ci-monorepo.yml`). The job is **optional** — it only runs on scheduled builds and
manual triggers, not on every PR. It can be enabled for PRs by setting the pipeline
variable `RUN_PERFORMANCE_TESTS=true`.

### Interpreting Results

The custom `performanceSummaryReporter` prints a summary table at the end:

```
================================================================================
PERFORMANCE SUMMARY
--------------------------------------------------------------------------------
Tests:     42 conclusive, 3 inconclusive
Baseline:  9.0.0
Candidate: 9.3.3
--------------------------------------------------------------------------------
Average:   9.3.3 is 12.4% slower
Median:    9.3.3 is 10.1% slower
Range:     5.2% faster to 38.7% slower
--------------------------------------------------------------------------------
Worst regression:    Shadows [#abc123] (38.7% slower)
Best improvement:    PBR basic [#def456] (5.2% faster)
================================================================================
```

- **Passed**: candidate is within the accepted threshold (default 15%) or faster
- **Failed**: statistically significant regression exceeding the threshold
- **Inconclusive**: measurements too noisy for a definitive result (retried automatically)
