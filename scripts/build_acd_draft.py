"""Assemble a review draft from actual A/C/D material; never invent B's chapter."""
from pathlib import Path
from copy import deepcopy
import nbformat

ROOT=Path(__file__).resolve().parents[1]
md=nbformat.v4.new_markdown_cell
code=nbformat.v4.new_code_cell


def main():
    a=nbformat.read(ROOT/'notebooks/A_data_foundation.ipynb',as_version=4)
    d=nbformat.read(ROOT/'notebooks/D_evaluation.ipynb',as_version=4)
    cells=[md('# CA6003 Bank Marketing · A+C+D 整合草稿\n\n更新：2026-10-16。**B的核心发现未收到，最终测试未执行，本文件尚不是完整课程提交物。**\n\nA提供数据治理与固定划分；C提供正式S/U管线和转换测试；LIANG SHUANG（D）负责评价与本次接入。C原增量ZIP不含说明提到的analysis.ipynb，本文件由已有真实材料重建，并不声称由C原包提供。B位置明确留空。核心A审计、C转换、D评价函数已内嵌，数据与证据文件随交付包提供。'),
           deepcopy(d.cells[2]),code('from IPython.display import display')]
    contract=next(c for c in d.cells if c.cell_type=='code' and 'def load_partition(' in c.source)
    cells += [md('## 01–02 / A：数据来源、范围和治理'),deepcopy(contract)]
    audit=(ROOT/'src/audit.py').read_text(encoding='utf-8').replace('from __future__ import annotations\n','').replace('from src.data_contract import CATEGORIES, EXPECTED_COLUMNS, FEATURES, NUMERIC, ROOT','\n# Data-contract names are defined above; core audit source included here.')
    for index in range(1,13):
        cell=deepcopy(a.cells[index])
        if cell.cell_type=='code' and 'from src.audit import profile_train, audit_train' in cell.source:
            cells.append(code(audit))
            cell.source=cell.source.replace('from src.audit import profile_train, audit_train','')
        cells.append(cell)
    cells += [md('## 03 / B：核心发现与分层证据（尚未收到）\n\n**待B提供真实train-only发现、可重建交叉表/分层表和分母。**\n\n必须包括历史联系三字段分析，以及job/education unknown的总体与分层证据。A的质量计数不是B已完成的独立发现，D的分组模型结果也不能替代此章。收到后补入真实观察、替代解释、S/U处理理由和限制。当前禁止凭空补“发现”或编造B贡献。详见notes/D_MISSING_ITEMS.md。'),
              code("B_discovery_status = {'received': False, 'final_test_allowed': False}\nprint(B_discovery_status)"),
              md('## 04–05 / C正式准备管线 + D四组验证\n\n以下C源码与用户ZIP中的preprocessing.py保持相同逻辑，仅为Notebook去掉模块导入。S/U边界与无泄漏测试见C_preprocessing_tests.py。当前全部结果是验证集结果。')]
    begin=next(i for i,c in enumerate(d.cells) if c.cell_type=='markdown' and c.source.startswith('## 3. C formal'))
    cells.extend(deepcopy(d.cells[begin:]))
    cells.append(md('## 06 / 全组封版待办\n\nB补核心发现并复核解释 → 成员实际复核 → D冻结并执行一次最终测试 → 更新所有表图与结论 → C核对单一Notebook → A新内核验收 → 四人连续出镜视频与实际提交。\n\n成员贡献：A数据基础；C正式预处理与14项转换测试；LIANG SHUANG/D统一评价、接入和草稿组织；B证据未收到、不得宣称完成。GenAI辅助范围和实际人工核验状态须由成员补充确认。当前无任何最终测试指标，不能提交成完整作业。'))
    for cell in cells:
        if cell.cell_type=='code':
            cell.outputs=[];cell.execution_count=None
    nbformat.write(nbformat.v4.new_notebook(cells=cells,metadata=d.metadata),ROOT/'analysis.ipynb')
    c=nbformat.read(ROOT/'notebooks/C_preprocessing.ipynb',as_version=4)
    c.cells[0].source='# C / D3：正式准备实现与验证\n\n2026-10-16在D项目重新执行。原ZIP中的C章节来源保留；analysis.ipynb为本次另行重建的A+C+D草稿，B与最终测试待补。C原环境证据已归档，以下表来自当前D统一环境的复跑。'
    nbformat.write(c,ROOT/'notebooks/C_preprocessing.ipynb')
    print('Built A+C+D review draft; B left explicitly pending; updated C chapter status.')


if __name__=='__main__':
    main()
