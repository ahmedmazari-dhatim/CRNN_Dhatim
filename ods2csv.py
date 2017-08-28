import sys,zipfile,re,os,csv
from pyquery import PyQuery as pq
from lxml.cssselect import CSSSelector

def ods2csv(filepath):

    xml = zipfile.ZipFile(filepath).read('content.xml')

    def rep_repl(match):
        return '<table:table-cell>%s' %match.group(2) * int(match.group(1))
    def repl_empt(match):
        n = int(match.group(1))
        pat = '<table:table-cell/>'
        return pat*n if (n<100) else pat

    p_repl = re.compile(r'<table:table-cell [^>]*?repeated="(\d+)[^/>]*>(.+?table-cell>)')
    p_empt = re.compile(r'<table:table-cell [^>]*?repeated="(\d+)[^>]*>')
    xml = re.sub(p_repl, rep_repl, xml)
    xml = re.sub(p_empt, repl_empt, xml)

    d = pq(xml, parser='xml')
    ns={'table': 'urn:oasis:names:tc:opendocument:xmlns:table:1.0'}
    selr = CSSSelector('table|table-row', namespaces=ns)
    selc = CSSSelector('table|table-cell', namespaces=ns)
    rowxs = pq(selr(d[0]))
    data = []
    for ir,rowx in enumerate(rowxs):
        cells = pq(selc(rowx))
        if cells.text():
            data.append([cells.eq(ic).text().encode('utf-8') for ic in range(len(cells))])

    root,ext=os.path.splitext(filepath)
    with open(''.join([root,'.csv']),'wb') as f:
        for row in data:
            dw = csv.writer(f)
            dw.writerow(row)

ods2csv(os.path.expanduser('~/foo.ods')) #example