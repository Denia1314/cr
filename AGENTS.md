# Project delivery workflow

The user requests that every completed robot code update be uploaded to GitHub.

- After implementing a requested robot change, run the relevant tests, commit the scoped code, tests and documentation, and push to the configured origin branch. This workflow is authorized by the user; do not ask for routine push confirmation again.
- Fetch and inspect remote changes before pushing. Preserve work from both computers, resolve overlapping changes, and rerun affected tests after merging. Never force-push or discard local/remote work.
- Keep raw battle data, screenshots, local reports, credentials, machine identity and model files out of the code repository. Replay data and validated models use the existing private training-data synchronization workflow.
- Verify the remote commit after pushing. Report the commit and whether upload succeeded; if it failed, preserve the local commit and explain the actual blocker.
- A request to inspect, diagnose or report status alone does not authorize new code changes.
