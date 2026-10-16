# LIANG SHUANG / D · video narration draft

Status: VALIDATION ONLY, updated 2026-10-16 using C's formal pipeline. C code received and locally rerun; B evidence and human review pending. Target 07:50–10:35, around 2m45s. Replace the numbers with frozen final-test results before the final video. Do not rename validation results as test results.

## 0:00–0:30 / Experiment design

I am Liang Shuang, responsible for the controlled evaluation. We compare four preparation settings using the same logistic regression, fixed data split and point-five threshold. S separates the never-contacted state from elapsed days. U keeps unknown job and education values as categories instead of replacing them with training modes. Everything else stays fixed.

## 0:30–1:10 / Overall results and contrasts

These validation results use C's formal pipeline fitted on the training partition. Average precision ranges from about point-three-five-eight-five to point-three-five-nine-five. S gives a small ranking improvement under both U settings. U has mixed effects and slightly worsens Brier under both S settings. Compared with the baseline, the combined setting improves average precision by about point-zero-zero-zero-nine, but recall decreases by about point-zero-nine-five percentage points. We report all five planned contrasts, including deterioration.

## 1:10–1:35 / Reference and threshold

Accuracy needs context: predicting no for everyone already achieves eighty-eight-point-three percent. Our model accuracy is around eighty-nine-point-four percent, while recall is only about seventeen percent. Better ranking and probability scores therefore do not mean that the fixed threshold identifies most subscribers. We did not tune that threshold after seeing the results.

## 1:35–2:15 / Group costs

We also use the manual’s predefined age and information groups. Among people aged sixty or above, the combined setting detects twenty-eight of one hundred and twenty-eight positives, compared with twenty-nine for the baseline. In the original unknown-information group, recall stays at ten of forty-six, but Brier becomes slightly worse. These small changes show why an overall average cannot replace subgroup reporting.

## 2:15–2:45 / Interpretation and status

Clearer semantics remain useful even when predictive gains are small. Existing history variables may already encode similar information, but that explanation still needs B’s training evidence. We make no significance or causal claim. The dataset cannot establish customer independence or future performance. After B and C review the evidence and pipeline, we will freeze the rules, refit on training plus validation, and evaluate all four settings on the sealed test partition once.

## 中文理解提纲

四组公平对照 → C正式管线已接入复跑 → S改善很小、U方向混合 → Accuracy要与全no基线比较 → ≥60岁与unknown组存在局部代价 → 收齐B证据并完成实际复核后才做最终测试。

最终版本操作：将“current validation”改为真实最终测试阶段的描述，并从 outputs/D/test 的表重新取全部数值；改写最后一段为已实际完成的流程。先交 B 核解释，再由 D 与表格逐项校对。每位成员必须实际出镜并讲解；稿件不代表视频已完成。
