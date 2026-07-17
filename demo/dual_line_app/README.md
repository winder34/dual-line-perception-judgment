# Dual-Line Single Image Demo

This app runs the verified v235 label-free path as an interactive single-image tool.

```text
raw image
-> Parent/Fine base prediction
-> v233 error-risk detector
-> selective reobservation
-> v234 joint correction
-> final prediction and decision trace
```

The uploaded image has no truth label. Its filename and directory are not used as
model features. Accuracy, fixed, and broken metrics are intentionally absent from
the runtime response because those require a separate truth audit.

## Run

From the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m tools.serve_dual_line_demo
```

Open `http://127.0.0.1:7860`.

After the artifacts are loaded, the terminal prints the exact local URL:

```text
[READY] http://127.0.0.1:7860
[MODE] raw image input, no truth labels
```

The browser opens automatically unless `--no_browser` is supplied. The
`127.0.0.1` address is local to the computer running the server; it is not a
public demo URL. Stop the server with `Ctrl+C`.

The server loads the ResNet18 artifacts once at startup. Each request returns:

- final Parent and Fine predictions
- base predictions and Top-3 probabilities
- KEEP, REVIEW_KEEP, or SWITCH decision
- risk score and TRAIN threshold
- base/candidate validity and transition utility
- selected reobservation view and normalized bbox
- ranked candidate evidence rows

## Current artifact scope

The public demo and supplied training artifacts target this 10-class setup:

```text
persian+cat, siamese+cat, chihuahua, german+shepherd, wolf,
fox, lion, tiger, horse, deer
```

Supporting a different class set requires regenerating the class manifest,
Parent/Fine mapping, OOF profile, and correction artifact from the new training
configuration.
