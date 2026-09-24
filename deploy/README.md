# Production launch

The two-box V4.1 deployment, as it actually runs. These files are the source of
truth for production configuration. A setting that exists only on a command
line or in someone's notes is a setting the next restart silently drops.

| File | Runs on | What |
|---|---|---|
| `run-hub.sh` | box 1 | The hub server launch: every production env var (as overridable `${VAR:-default}`), log rotation to `~/logs/v41-server.log`, `exec` of the server. A bare run is production. |
| `restart-hub.sh` | box 1 | SIGKILL the running hub, optionally install a new binary (`INSTALL=...`, previous kept as `.prev`), relaunch via `run-hub.sh`, wait for `listening on`. |
| `restart-expertd-b2.sh` | box 1 (ssh → box 2) | Restart box 2's `deepstrix-expertd`: waits for the old GTT pool to drain, writes its runtime knobs file (`~/expertd-knobs.txt`, reloaded on SIGUSR2), launches, waits for the listener. |

**Order:** box 2 first. The hub connects to `V41_REMOTE_ADDR` while loading and
panics if the daemon isn't listening.

**Building:**
- Hub: `CARGO_TARGET_DIR=target-v41 nix develop -c cargo build --release -p deepstrix-server --features v41`
- Box 2: rsync `crates/` and `Cargo.*`, then **touch the sources** before building. Box 2's clock runs ~36 h ahead, so synced mtimes look stale to cargo, which then silently keeps old rlibs.

If you build somewhere other than `target-v41`, pass the result as `INSTALL=`
to `restart-hub.sh`. Production execs `target-v41/release/deepstrix-server`, so
rebuilding in place replaces the binary under the next restart.
