# v0.3.1 release validation

## Scope

This patch includes NN output linearization, three regression tests, version synchronization, and documentation. Uncommitted dynamics/CLI changes in the separate development worktree are excluded.

## Isolated wheel validation

- Platform: macOS arm64; Python 3.14.5.
- Built `autoquantum-0.3.1-py3-none-any.whl` with `python3 -m pip wheel --no-deps` and an isolated build environment.
- Created a new virtual environment without system site packages and installed the wheel with its dependencies.
- Executed validation from outside both source worktrees; confirmed imports resolved to the installed wheel in the virtual environment.
- Dependency versions observed: NumPy 2.5.3, Matplotlib 3.11.2, SciPy 1.18.1.
- `python -m unittest discover -s <release-source>/tests -v`: **12 passed**, including 5 NN tests.
- `autoquantum --version`: **AutoQuantum v0.3.1**.
- Runtime version and installed distribution metadata both equal **0.3.1**.
- `autoquantum info`: exited successfully.
- `python -m pip check`: **No broken requirements found**.

Wheel SHA-256:

```text
a392faecc98809d98846296fac9a9593d226a5fa6f40fa32e16c0ea7cf361f44
```

## Not validated or fixed by this patch

No end-to-end reaction calculation, H3 CLI run, old wavepacket path, physical LEPS threshold, flux extraction, or cross-platform/Python-version matrix was certified. `info` lists legacy capabilities; it is not a health check for those capabilities. Old NN model predictions change with the output-layer correction and require revalidation or retraining. Known limitations are listed in README.md.
