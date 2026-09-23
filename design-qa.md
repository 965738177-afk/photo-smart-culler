# Design QA — AI 挑图设置

## Comparison target

- Source visual truth: `/var/folders/4s/l6hxr0x516g7v6mcmwhv91g80000gn/T/codex-clipboard-0fc6fc66-81bb-4158-8c4d-ea638532a647.png`
- User-directed change: replace the source's four horizontal preference chips with vertically stacked rows, each with a right-aligned switch.
- Implementation capture: packaged `秒选 QA 2.0.34`, reviewed in the same visual comparison pass at the workbench's full 1212 × 768 px state; the earlier settings-dialog review remains valid for that unchanged surface.
- Implementation source: `event_cull/app.py::show_picker_settings` and picker-option mapping.
- Final package: `release/照片智能筛选器 9.app` (v2.0.34).

## Viewport and normalization

- The source is a 716 × 74 px crop of the former preference strip, while the implementation review includes the complete settings dialog. Comparison is therefore scoped to the preference treatment rather than pixel-for-pixel dialog geometry.
- The source and implementation were reviewed together as a before/after change: the vertical arrangement is intentional user-approved design direction, not a mismatch.
- The normal-size dialog verifies the dense first fold; the maximized dialog verifies the lower “相似图处理” controls, including the default duplicate count.

## Comparison history

1. **Earlier finding — P1:** horizontal chip controls were visually similar to status tags. The leading checkbox glyph made the active “启用 AI 精选” state easy to misread, which can leave analysis in manual-review mode with no auto-selected pictures.
   - **Fix:** replaced chips with four full-width rows: clear text at left, one compact rounded switch at right, and quiet dividers between choices.
2. **Earlier finding — P1:** the settings form's wheel delta was reversed; the lower duplicate controls could be unreachable in a short dialog.
   - **Fix:** apply normalized wheel units directly to the form canvas. The normal dialog retains a scroll region, while a larger window shows every control without clipping.
3. **Clarification improvements:** removed the custom header close square; renamed “相似度” to “分组严格度” with a concrete strict/loose explanation; changed “每组重复图片精选张数” default and reset value to 2.
4. **Earlier finding — P1:** the three adjacent toolbar dropdowns displayed only values such as `仅 1★ / 2★ / 3★`; their different effects were impossible to infer from the compact control row.
   - **Fix:** added the persistent labels “显示照片 / 备用评级 / 精选评级” and hover explanations. The first only filters the current wall; the latter two map the internal backup/featured tiers to Lightroom's written star values.
5. **Selection adjustment:** three-star candidates now use approximately one fifth of the selected target rather than one tenth. The per-similar-group limit remains one featured frame, so the extra first-picks improve coverage rather than promote near-duplicate bursts.

## Findings

- No actionable P0, P1, or P2 differences remain in the approved settings surface.
- The only close affordances now visible are macOS's standard red/yellow/green window controls; the unwanted custom white `×` button is gone.
- “启用 AI 精选” is visibly on by default; the explicitly disabled “数量不足时用重复照片补足” switch remains distinct, so the two behaviors cannot be confused.
- The workbench toolbar now visibly distinguishes filtering (`显示照片`) from rating-output mapping (`备用评级` and `精选评级`).

## Required fidelity surfaces

- **Fonts and typography:** concise bold labels stay left-aligned and scan as settings rather than selectable chips; helper copy remains subordinate.
- **Spacing and layout rhythm:** four preference rows share the same left edge, right switch edge, and divider cadence as the quality-detection rows below.
- **Colors and tokens:** active switches use the existing copper track, inactive state uses muted graphite, and the charcoal form palette is unchanged.
- **Copy and content:** strictness now communicates behavior directly: “严格” groups only nearly identical frames; “宽松” also groups slightly different poses or compositions. The lower control visibly starts at 2.
- **Toolbar clarity:** `显示照片` controls only the photos currently shown; `备用评级` and `精选评级` make the two Lightroom-output mappings explicit without adding another dialog.
- **Image quality and asset fidelity:** no image or logo asset changed.

## Primary interactions tested

- Opened the setting dialog from the native 操作 → 挑图设置… menu in the isolated v2.0.33 package.
- Verified the normal dialog's vertical preference treatment, active/default switch states, and absence of the custom close button.
- Maximized the dialog to verify the full lower controls: “分组严格度” options `严格 / 标准 / 宽松` and “每组重复图片精选张数” value `2`.
- Opened the packaged v2.0.34 workbench and verified the labeled filter/mapping trio appears in the review toolbar without clipping its view controls.
- Automated focused regression suite: `113 passed`.

final result: passed
