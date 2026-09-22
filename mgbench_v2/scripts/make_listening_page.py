"""Inject items.json into listening_check_template.html -> page.html (audio/ stays next to it). Usage: make_listening_page.py [dir] [extra output dir]"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "/lustre09/project/6002780/kunfang/music_grounding/mgbench_v2_out/listening_check"
tpl = open(os.path.join(HERE, "listening_check_template.html")).read()
assert tpl.count("/*__ITEMS__*/[]") == 1
items = json.load(open(os.path.join(OUT, "items.json")))
page = tpl.replace("/*__ITEMS__*/[]", json.dumps(items, separators=(",", ":")))
for d in [OUT] + sys.argv[2:]:
    open(os.path.join(d, "page.html"), "w").write(page)
print(len(items), "items ->", len(page), "bytes")
