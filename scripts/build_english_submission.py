"""Build the English, single-file submission without altering frozen evidence."""
from pathlib import Path
import hashlib
import io
import json
import re
import zipfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'analysis.ipynb'
nb = nbformat.read(path, as_version=4)
nb.cells = [c for c in nb.cells if not c.metadata.get('portable_bootstrap')]
assert len(nb.cells) == 41

def tables(index):
    return re.findall(r'(?m)^\|.*(?:\n\|.*)*', nb.cells[index].source)

b = tables(21)
d = tables(39)
markdown = {
0: '''# CA6003 Bank Marketing — Group 17

**Submission filename:** `Assignment_Grp_17.ipynb`

**Research question:** How do special-value semantics and unknown-information handling affect predictive reliability?

| Role | Member | Responsibility |
| --- | --- | --- |
| A | Yu Qiushi | Data provenance, governance and frozen partitions |
| B | Chen Junyu | Training-only EDA and preparation rationale |
| C | Sun Shaojie | Semantic transformation and preprocessing pipelines |
| D | Liang Shuang | Controlled experiments and evaluation |

This Notebook contains the data audit, cleaning decisions, EDA, four controlled ML experiments, results and limitations. Submit only this `.ipynb`. Core analysis code is inline. On first execution, the short setup cell downloads the data and evidence archive from a fixed public GitHub commit and verifies its SHA-256. Internet access to raw.githubusercontent.com is required unless the verified archive is already beside the Notebook. No repository clone is needed. Run All reproduces training analyses and validation fits, and recomputes final-test metrics from frozen predictions without refitting final-test models or selecting strategies using test results.

The final test was already executed with the task owner's authorization. Personal review by all four members, video recording and course submission remain pending.

## Reproducible environment

Use Python 3.12 (verified with 3.12.12). In a dedicated environment, install the packages below before selecting its Jupyter kernel. Installation is an explicit setup step, never run automatically by this Notebook.

```bash
python -m pip install numpy==2.5.4 pandas==3.0.6 scipy==1.18.1 scikit-learn==1.8.0 matplotlib==3.11.2 nbformat==5.11.1 nbclient==0.11.0 ipykernel==7.4.0
python -m ipykernel install --user --name bank-marketing --display-name "Bank Marketing (Python 3.12)"
```

Use an existing Jupyter front end or VS Code, select this kernel and run all cells in order. The downloaded `requirements.current.txt` and `requirements.D.lock.txt` provide the full dependency lock; they are available under `ROOT` after the bootstrap cell runs. Runtime versions are printed below. Different numerical-library versions can fail the strict reference checks; use the pinned environment instead of silently relaxing tolerances.

The next cell downloads and verifies the archive, caches it as `BankMarketing_Data.zip` in the current directory, and extracts it into a new temporary directory. An existing cache must pass the same hash check. The Notebook contains no encoded dataset string or hidden data cell. Source code in the archive is retained for provenance checks, not executed; all substantive analysis and ML code is visible below.''',
3: '## 01–02 / A: Data provenance, scope and quality governance',
5: '''## 1. Provenance, license and scope

We use only legacy `bank-full.csv`: 45,211 rows, 16 original inputs and target `y`. Provenance follows the outer ZIP, nested `bank.zip`, and original CSV/documentation. The additional dataset, sampled `bank.csv`, and reformatted `data.csv` are excluded.

Source: [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing), by Moro, Rita and Cortez; [DOI 10.24432/C5K306](https://doi.org/10.24432/C5K306). License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The legacy documentation also credits Moro, Laureano and Cortez (2011), *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*. Full attribution is included in the embedded `THIRD_PARTY_NOTICES.md`.

Raw bytes remain unchanged. Row identifiers, splits and audits are project derivatives. Attribution does not imply endorsement by the authors or UCI. All required inputs are restored from this Notebook's embedded snapshot; no personal file paths or online downloads are used.''',
7: '''## 2. Read without semantic cleaning and freeze the exact split

`keep_default_na=False` preserves literal `unknown` and empty strings; actual missing values are checked separately. `row_id=np.arange(len(raw))` is the zero-based original row position, not a customer identifier or predictor. Both stratified splits use `random_state=42`; the second stratifies on `y.iloc[dev]`. The manifest is sorted by original row ID, while the JSON preserves the order returned by the split operations.''',
9: '''The following table exposes label distributions only for training. Validation/test rows show predefined coverage and sizes without label statistics at this audit stage. Open-ended profiling uses training only. Sealing is a workflow constraint: the original CSV still contains all records.''',
11: '''## 3. Dictionary and prediction-time policy

The 11 permitted inputs are `age`, `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`, `pdays`, `previous` and `poutcome`.

`duration` is post-call information. We also exclude `campaign`, `contact`, `day` and `month` under a conservative pre-contact scope. This is our study design, not a UCI or course requirement. Reviewer fields remain pending until actual personal review.''',
13: '''## 4. Training-only quality evidence

The audit functions are included directly below. We distinguish actual NaN/empty strings, literal `unknown`, and the `pdays=-1` sentinel. The sentinel means no previous contact; `pdays=0` is a real interval. Negative balances, large counts and matching records are not automatically errors. A STOP rule blocks processing only when `n_affected>0`; INVESTIGATE requests review without deleting records.

Cleaning is implemented conservatively: validate the whitelist and raw values, encode sentinel semantics in S, handle job/education unknowns in U, and learn imputation/scaling/encoding only inside the current training fit. No unsupported deletion, deduplication or winsorization is applied.''',
16: '''## 5. History consistency handoff

Training cross-counts check semantic consistency and guide B's investigation. These observations do not justify inventing contact histories, identifying why an outcome is unknown, or making causal claims. Independent training-only investigation follows.''',
18: '## 03 / B: Training-only findings and stratified evidence',
24: '## 04–05 / C: Formal preparation and D: Controlled evaluation',
37: '''## 05b / Final test: Saved predictions and complete metrics

The following code recalculates metrics from the predictions produced after the original freeze, checking hashes, row order and labels. It does not refit final-test models or use test results to develop a new strategy. This is reproducible metric replay; model fitting is reproduced on validation in the preceding section.''',
40: '''## Contributions, GenAI disclosure and remaining work

**A — Yu Qiushi:** source provenance, governance, original audit and frozen splits. **B — Chen Junyu:** training-only findings, independent count verification, stratified evidence, figures and narrative with Codex assistance. **C — Sun Shaojie:** the supplied formal S/U pipeline and transformation tests. **D — Liang Shuang:** unified evaluation and the original delivered evaluation package.

At the task owner's request, OpenAI Codex assisted C/D integration, B implementation, environment verification, authorized frozen-test execution, final Notebook assembly and English presentation materials. These additions are not represented as personal work performed or reviewed by C or D.

**GenAI disclosure:** OpenAI Codex assisted coding, debugging, analysis suggestions, charts, explanation drafts, translation and automated verification. Checks include independent raw-CSV counts, hashes and frozen partitions, transformation and leakage tests, prediction-level metric reconstruction, and execution in a fresh kernel. Automated checks do not constitute personal understanding or signatures. All members must review and be able to explain the submitted work.

Group 17 and all member names have been supplied. Remaining items are actual member review, a continuous four-person video within 12 minutes, its public link, NTULearn submission and peer assessment. Data attribution and CC BY 4.0 licensing are retained above and in the embedded source notices.'''
}
markdown[21] = '''# B / D2: Data insights and preparation rationale

Exploration uses only the fixed training partition: 27,126 rows and 3,173 positives. Counts and denominators are explicit; neither validation nor test is used for open-ended EDA. The inline investigation code corresponds to `src/b_discovery.py`, with evidence in `outputs/B/`. Codex assisted this analysis; Chen Junyu's personal review remains pending.

## Core finding: Interpret unknown values through relationships between fields

In training, `pdays>=0` and `previous>0` agree for every row. However, `poutcome=unknown` includes 22,235 never-contacted records and five previously contacted records. The latter are 5/22,240 (0.0225%) of all unknown outcomes.

''' + b[0] + '''

An unknown count alone cannot distinguish an absent previous campaign from an existing campaign with an unknown result. The cause of the five exceptions cannot be identified from these fields; they are not automatically parsing errors. Their original values and row IDs are retained in `history_exceptions.csv`.

This supports S separating contact status from the conditional interval, and limiting U to job/education rather than filling every unknown outcome. Other history fields already overlap strongly, so additional predictive gains may be limited. Continuous `previous` is not mathematically equivalent to a binary status in logistic regression. S also changes dimensionality and regularization; comparisons are not causal effects.

## Supporting finding 1: The unknown association varies across age groups

''' + b[1] + '''

The overall subscription rate is 13.52% for any job/education unknown versus 11.61% when both are known. Unknown groups have higher rates below 60, but the direction reverses at age 60 or above. That unknown subgroup contains only 97 rows and is flagged as small. This is not a claim of Simpson's paradox: not every stratum reverses the overall direction. Separate job, education and history strata, including unknown counts divided by stratum size, are reported by the code in `unknown_strata`.

Alternative explanations include customer composition, recording processes, unmeasured factors and sampling variation. These results do not establish MAR or MNAR. Mode imputation merges observable information states; keeping categories may retain useful information or increase small-sample instability. U therefore requires a controlled comparison, without promising better performance or fairness.

### Descriptive age standardization

We reweight within-age subscription rates using the same pooled training age distribution. This fits no model, changes no evaluation groups and is not causal adjustment.

''' + b[2] + '''

Weights are reported in `age_weights`. This addresses only coarse age composition; within-age differences, other confounding and the uncertainty of 97 observations remain. For any unknown, the crude gap is +1.9031 percentage points and the standardized gap is +1.4216 points. Composition affects the comparison, but the remaining gap is not an effect caused by unknown status. This descriptive analysis changes neither model training data nor D's groups.

## Supporting finding 2: Investigate numerical tails without automatic deletion

''' + b[3] + '''

`numeric_tail_examples` preserves extreme balances and previous-contact counts with original row IDs. Negative balances are semantically possible. High contact counts warrant review, but without event histories we cannot prove either error or validity for each extreme. All rows are retained and all experiments use the same StandardScaler. No deletion, clipping, logarithmic transformation or extra outlier factor is introduced. Tail sensitivity remains a limitation of the linear model.

## From observation to experiment

| Observation | Preparation decision | Verification | Interpretation limit |
| --- | --- | --- | --- |
| Sentinel status differs from interval; history fields overlap | S separates status and conditional days | ES-E0 and ESU-EU | Small gains do not negate semantics or prove redundancy |
| Unknown associations vary across strata | U keeps categories versus training-mode imputation | EU-E0 and ESU-ES | Association does not identify causality or missingness mechanism |
| Overall averages can mask group differences | Fixed age and original-information groups | Recall, FPR, Brier and denominators | Group differences are not proof of discrimination |

Validation and final-test results appear separately below. Improvements, deterioration and unchanged results are all reported. No confidence intervals or significance tests were computed. Missing customer identifiers and exact snapshot times prevent guarantees of customer independence, future generalization or causal marketing benefit.

## Figures and team review

The next three charts describe training only. The fourth, model-tradeoff chart appears with D's final results and is explicitly separated from EDA. Source tables are embedded or recomputed. C supplies the fixed S/U implementation; D evaluates the predefined groups. The team still needs to review the five exceptions, key denominators and interpretation. Automated verification is not a human signature.'''
markdown[39] = '''## 06 / Discussion and limitations

Training supports B's EDA and validation fitting; validation supports the predefined comparisons. After freezing, each of the four models was refitted on train+validation (36,168 rows) and evaluated on 9,043 fixed test rows. No hyperparameter or threshold search was performed, and the whitelist and groups remained fixed. The embedded run manifest records the environment, iterations and hashes. Computational authorization is distinct from human sign-off.

### Validation: fit on training, evaluate on validation

''' + d[0] + '''

### Final test: original fit on train+validation, evaluate on test

''' + d[1] + '''

### All five contrasts: raw score differences

''' + d[2] + '''

For the full preparation, ESU-E0 changes AP by -0.00026201, Recall by -0.0945 percentage points, F1 by -0.00131466 and Brier by +0.00000422. Lower Brier/FPR and higher AP/Recall/F1 are preferable. Deterioration and ties are retained. No significance test supports a claim of statistically significant improvement.

Assess S through both ES-E0 and ESU-EU, and U through EU-E0 and ESU-ES. Overlapping history signals offer a possible explanation for small changes, not proof of mechanism. Unknown states have age-dependent associations, without guaranteeing benefits in every group. S/U alter representation, dimensionality and regularization. Their score differences are not causal effects, and validation-to-test changes also involve different fitting sample sizes.

On this test set, S slightly improves AP and Brier with either U setting, but identifies one additional positive only when U is off. U lowers AP and Recall and raises Brier with either S setting. Retaining unknown information was a reason to test U, not a demonstrated benefit: this model does not show that benefit. This does not imply that mode imputation is universally better.

### Full-preparation contrasts within groups

''' + d[3] + '''

The preceding executable results include group denominators, positive/negative counts, confusion matrices, all four models and all five contrasts. Information groups use original values. A group is flagged as small when n<100 or either class has fewer than 20 observations. Age and information dimensions overlap and must not be added together. Descriptive group differences are not proof of discrimination or fairness.

Relative to baseline, full preparation reduces Recall from 30/112 to 29/112 for age >=60 (-0.8929 percentage points), and from 10/59 to 9/59 for original unknown information (-1.6949 points). Brier also rises in these groups. Overlap means these changes cannot be added as two distinct missed people. A small aggregate change can hide a larger group-level cost.

### No-feature references and limitations

''' + d[4] + '''

Compare accuracy with the all-no reference and consider low positive Recall. AP measures ranking and is not trapezoidal PR-AUC. Brier reflects both discrimination and calibration, not a standalone calibration test. Without customer IDs, complete snapshot timing or randomized marketing assignment, we cannot establish customer independence, future generalization, causal call effects or actual profit. Retained numerical tails may affect the linear model. GenAI assistance and pending personal review must remain disclosed.'''
for index, source in markdown.items():
    nb.cells[index].source = source.replace('embedded', 'downloaded').replace("All required inputs are restored from this Notebook's downloaded snapshot; no personal file paths or online downloads are used.", 'All required inputs come from the version-pinned archive downloaded by the setup cell; no personal file paths are required.')

replacements = {
'C / D3：手册第 6 页的正式 S/U 管线；不读取任何分区。': 'C / D3: formal S/U pipeline; no partition is loaded here.',
'白名单、原始异常检查和不学习参数的语义转换；保留 index。': 'Check raw values and whitelist; preserve the index without learning parameters.',
'真空值属于上游 STOP；U 产生的 NaN 在此检查之后出现。': 'Actual missing values trigger STOP; imputation-related NaN appears after this check.',
'完整分类器也提供编码特征名；标准 Pipeline 的末端模型无此接口。': 'Expose encoded feature names for the complete classifier pipeline.',
'每次返回未拟合的新对象，供 D 在当前 fit 分区独立训练。': 'Return a fresh, unfitted pipeline for each independent training fit.',
}
for old, new in replacements.items():
    nb.cells[26].source = nb.cells[26].source.replace(old, new)
nb.cells[1].source = re.sub(r"ROOT = Path.cwd\(\).*?(?=import sklearn)", "assert (ROOT / 'data/raw/bank-full.csv').is_file(), 'Run the bootstrap cell first'\n", nb.cells[1].source, flags=re.S)
nb.cells[6].source = re.sub(r"ROOT = Path.cwd\(\).resolve\(\).*?(?=lock =)", "assert (ROOT / 'config/dataset_lock.json').is_file(), 'Run the bootstrap cell first'\n", nb.cells[6].source, flags=re.S)

# Include only project inputs and immutable evidence, never environments or credentials.
frozen = json.loads((ROOT / 'outputs/D/freeze.json').read_text())
paths = set(frozen['fingerprint']['files']) | set(frozen['evidence_hashes'])
paths.update(json.loads((ROOT / 'config/dataset_lock.json').read_text())['artifacts'])
for directory in ['data/splits', 'outputs/D/validation', 'outputs/D/test']:
    paths.update(str(p.relative_to(ROOT)) for p in (ROOT / directory).rglob('*') if p.is_file())
paths.update(['outputs/D/freeze.json', 'outputs/D/test_completed.json',
              'data/source/provenance.json', 'notes/data_dictionary.csv', 'notes/feature_policy.csv',
              'outputs/split_summary.csv', 'outputs/format_checks.csv', 'outputs/quality_audit.csv',
              'outputs/train_history_consistency.csv', 'outputs/audit_summary.json',
              'outputs/B/figures/04_model_tradeoffs.png', 'THIRD_PARTY_NOTICES.md',
              'requirements.current.txt', 'requirements.D.lock.txt'])
buffer = io.BytesIO()
with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
    for relative in sorted(paths):
        info = zipfile.ZipInfo(relative, date_time=(2026, 10, 18, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, (ROOT / relative).read_bytes())
payload = buffer.getvalue()
digest = hashlib.sha256(payload).hexdigest()
bootstrap = '''# Download a fixed revision, then verify the exact bytes before use.
# Original evidence bytes are preserved for hash verification, not imported as executable code.
import hashlib, io, tempfile, zipfile
from pathlib import Path, PurePosixPath
from urllib.request import urlopen
DATA_URL = 'https://raw.githubusercontent.com/zjuadurey/ca6003-bank-marketing/891ab3830b4307ba3e53ee9358c8473527476528/BankMarketing_Data.zip'
_cache = Path('BankMarketing_Data.zip')
if _cache.is_file():
    _payload = _cache.read_bytes()
else:
    with urlopen(DATA_URL, timeout=60) as _response:
        _payload = _response.read()
assert hashlib.sha256(_payload).hexdigest() == DIGEST, 'Data archive hash mismatch'
if not _cache.exists():
    _cache.write_bytes(_payload)
_workspace = tempfile.TemporaryDirectory(prefix='bank_marketing_')
ROOT = Path(_workspace.name)
with zipfile.ZipFile(io.BytesIO(_payload)) as _archive:
    for _entry in _archive.infolist():
        _relative = PurePosixPath(_entry.filename)
        assert not _relative.is_absolute() and '..' not in _relative.parts
        _target = ROOT.joinpath(*_relative.parts)
        _target.parent.mkdir(parents=True, exist_ok=True)
        _target.write_bytes(_archive.read(_entry))
print('Data and evidence loaded; fixed archive SHA-256 verified.')
del _payload
'''.replace('DIGEST', repr(digest))
cell = nbformat.v4.new_code_cell(bootstrap, metadata={'portable_bootstrap': True})
cell.id = 'portable-evidence-bootstrap'
nb.cells.insert(1, cell)
for cell in nb.cells:
    assert not re.search(r'[\u3400-\u9fff]', cell.source), cell.id
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
nb.metadata['submission'] = {'language': 'English', 'single_file': True, 'group_number': 17, 'filename': 'Assignment_Grp_17.ipynb', 'archive_sha256': digest,
                             'data_access': 'version-pinned HTTPS download or verified local cache',
                             'members': {'A': 'Yu Qiushi', 'B': 'Chen Junyu', 'C': 'Sun Shaojie', 'D': 'Liang Shuang'}}
nbformat.validate(nb)
nbformat.write(nb, path)
print(f'Built {path.name}: download-based loading; verified archive contains {len(paths)} files')
