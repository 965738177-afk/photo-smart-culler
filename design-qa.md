# Design QA — 授权与订阅

## Comparison target

- Source visual truth: `/var/folders/4s/l6hxr0x516g7v6mcmwhv91g80000gn/T/codex-clipboard-45ecac98-aab6-4af3-9d1f-96aca5d58fd7.png`
- Intended state: unauthenticated desktop subscription dialog, dark theme.
- Implementation: `event_cull/license_ui.py`, rebuilt into `dist/照片智能筛选器.app`.
- Intended viewport: 680 × 750 logical px.

## Comparison status

The source reference was available. The rebuilt macOS application was launched,
but the macOS session became locked before an implementation screenshot could be
captured. A same-state, same-viewport visual comparison is therefore not yet
possible.

## Implemented design changes awaiting visual capture

- Trial is now the first and largest card: `直接领取 1 天试用` / `无需邮箱`.
- Trial copy explains online issuance, current-Mac binding, one-device limit, and
  macOS Keychain storage.
- The email and activation-code form is explicitly secondary and labelled for
  purchase or formal activation only.
- Purchase and renewal actions remain present beside that secondary form.
- Dark gray surfaces, green trial trust state, gold trial CTA, and blue payment
  CTA are defined as the visual token hierarchy.

## Findings

- [P1] Rendered layout is not yet available for comparison.
  - Evidence: the automation surface reported that macOS is locked when asked
    to capture the rebuilt application.
  - Impact: viewport overflow, text wrapping, button emphasis, and dark-theme
    contrast cannot be verified from code alone.
  - Fix: unlock the macOS session, capture the rebuilt dialog at 680 × 750,
    then compare its full view and the trial-card region to the source.

## Required fidelity surfaces

- Fonts and typography: defined in the implementation; pending visual capture.
- Spacing and layout rhythm: defined in the implementation; pending visual capture.
- Colors and tokens: defined in the implementation; pending visual capture.
- Image quality and asset fidelity: no image assets are used by this dialog.
- Copy and content: reviewed in code; pending rendered line-wrap review.

## Implementation checklist

1. Unlock the desktop session.
2. Capture the rebuilt dialog in its unauthenticated state.
3. Check trial-card hierarchy, text wrapping, controls, contrast, and overflow.
4. Update this report with visual evidence and a final pass/fail result.

final result: blocked
