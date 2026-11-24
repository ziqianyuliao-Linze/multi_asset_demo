"""1号模块：使用2号统一收益率矩阵生成截至指定日的基础市场特征。
仅依赖Python标准库，不调用模型API。输入Excel须保留原工作表“日收益率矩阵”。
"""
import json
import math
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from statistics import stdev
from xml.etree import ElementTree as ET

ASSETS = ('510300', '511010', '518880')
NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
      'p': 'http://schemas.openxmlformats.org/package/2006/relationships'}


def read_matrix(path):
    """标准库直接读取Excel，不改变原表。"""
    with zipfile.ZipFile(path) as z:
        wb = ET.fromstring(z.read('xl/workbook.xml'))
        rel = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        target_by_id = {x.attrib['Id']: x.attrib['Target'] for x in rel}
        sheet = next((s for s in wb.findall('m:sheets/m:sheet', NS)
                      if s.attrib['name'] == '日收益率矩阵'), None)
        if sheet is None:
            raise ValueError('Excel中缺少工作表：日收益率矩阵')
        rid = sheet.attrib[f'{{{NS["r"]}}}id']
        target = target_by_id[rid].lstrip('/')
        sheet_path = target if target.startswith('xl/') else 'xl/' + target
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            sr = ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in sr.findall('m:si', NS):
                strings.append(''.join(n.text or '' for n in si.findall('.//m:t', NS)))
        root = ET.fromstring(z.read(sheet_path))
        records = []
        for row in root.findall('m:sheetData/m:row', NS):
            values = {}
            for c in row.findall('m:c', NS):
                ref = c.attrib['r']
                col = ''.join(ch for ch in ref if ch.isalpha())
                v = c.find('m:v', NS)
                inline = c.find('m:is', NS)
                if v is not None:
                    values[col] = strings[int(v.text)] if c.attrib.get('t') == 's' else v.text
                elif inline is not None:
                    values[col] = ''.join(t.text or '' for t in inline.findall('.//m:t', NS))
            if row.attrib.get('r') == '1':
                if tuple(values.get(k) for k in ('B', 'C', 'D')) != ASSETS:
                    raise ValueError('列头与固定资产顺序不一致')
                continue
            if not values:
                continue
            date = str(values.get('A', ''))[:10]
            datetime.strptime(date, '%Y-%m-%d')
            nums = []
            for col in ('B', 'C', 'D'):
                val = float(values[col])
                if not math.isfinite(val):
                    raise ValueError(f'{date} 存在非有限收益率')
                nums.append(val)
            records.append((date, nums))
    dates = [d for d, _ in records]
    if dates != sorted(set(dates)):
        raise ValueError('日期重复或未升序')
    if len(records) < 250:
        raise ValueError('数据不足250个有效交易日')
    return records


def compound(returns):
    return math.prod(1 + x for x in returns) - 1


def corr(xs, ys):
    xmean = sum(xs) / len(xs)
    ymean = sum(ys) / len(ys)
    xc = [x - xmean for x in xs]
    yc = [y - ymean for y in ys]
    denom = math.sqrt(sum(x * x for x in xc) * sum(y * y for y in yc))
    return sum(x * y for x, y in zip(xc, yc)) / denom if denom > 0 else 0.0


def build_features(records, as_of_date=None):
    if as_of_date is None:
        as_of_date = records[-1][0]
    selected = [(d, v) for d, v in records if d <= as_of_date]
    if len(selected) < 250:
        raise ValueError('截至指定日期有效交易日不足250条')
    if selected[-1][0] != as_of_date:
        raise ValueError('指定日期不是统一收益率矩阵中的交易日')
    asset_series = {a: [v[i] for _, v in selected] for i, a in enumerate(ASSETS)}
    assets = {}
    for asset, series in asset_series.items():
        assets[asset] = {
            'return_20d': compound(series[-20:]),
            'return_60d': compound(series[-60:]),
            'volatility_20d_ann': stdev(series[-20:]) * math.sqrt(250),
            'volatility_60d_ann': stdev(series[-60:]) * math.sqrt(250),
            'volatility_250d_ann': stdev(series[-250:]) * math.sqrt(250),
        }
    correlations = {
        f'{ASSETS[i]}_{ASSETS[j]}': corr(asset_series[ASSETS[i]][-60:], asset_series[ASSETS[j]][-60:])
        for i in range(3) for j in range(i + 1, 3)
    }
    return {'as_of_date': as_of_date, 'assets': assets, 'correlations_60d': correlations,
            'market_data_quality': {'all_required_fields_present': True,
                                    'source': 'unified_total_return_matrix',
                                    'sample_days_to_date': len(selected)}}


def write_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    if len(sys.argv) < 2:
        print('用法: python feature_builder.py "2：统一口径日收益率矩阵(1).xlsx" [YYYY-MM-DD]')
        return
    source = Path(sys.argv[1])
    records = read_matrix(source)
    date = sys.argv[2] if len(sys.argv) > 2 else records[-1][0]
    output = build_features(records, date)
    target = Path(__file__).parent / 'market_features.json'
    write_json(output, target)
    print(f'成功：{len(records)}条收益率；截至{date}；输出 {target}')


if __name__ == '__main__':
    main()
