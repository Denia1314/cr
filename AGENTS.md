# Project delivery workflow

The user requests that every completed robot code update be uploaded to GitHub.

- After implementing a requested robot change, run the relevant tests, commit the scoped code, tests and documentation, and push to the configured origin branch. This workflow is authorized by the user; do not ask for routine push confirmation again.
- Fetch and inspect remote changes before pushing. Preserve work from both computers, resolve overlapping changes, and rerun affected tests after merging. Never force-push or discard local/remote work.
- Keep raw battle data, screenshots, local reports, credentials, machine identity and model files out of the code repository. Replay data and validated models use the existing private training-data synchronization workflow.
- Verify the remote commit after pushing. Report the commit and whether upload succeeded; if it failed, preserve the local commit and explain the actual blocker.
- A request to inspect, diagnose or report status alone does not authorize new code changes.

## Version and stage displays

- Every delivered update must keep the package version and implemented upgrade stages in `crbot/__init__.py` current. Derive the console heading, CLI version and startup log from this shared metadata; update the user documentation in the same commit.
- Show the implemented project stage (including D/E and future stages) separately from selectable rule profiles and trained model versions. Do not label an old rule profile as the latest overall release or rename model compatibility keys just to change the display.
- Add a stage only after its functionality is implemented. Distinguish implemented capabilities, configured/enabled behavior, actual model loading and completed battle acceptance; never infer acceptance from a stage label.
- Verify the displayed stage against the delivered changes before pushing, including the console at its minimum window size when the layout changes.
