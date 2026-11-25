"""One-page listening check: play buttons, one-click answers, a note per row, "Copy answers".

uv run python scripts/rate_page.py spec.json OUT_DIR

spec.json: {"title", "intro", "sections": [{"key", "title", "help", "choices": [...],
  "items": [{"id", "text", "clips": [[label, wav path], ...]}]}]}
Clips are copied into OUT_DIR/audio; answers are kept in the browser until copied.
"""

import argparse
import html
import json
import shutil
from pathlib import Path

PAGE = """<!doctype html><html lang="km"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>
:root{{--bg:#fff;--fg:#16181d;--mute:#667;--line:#e3e5ea;--on:#2f5bea;--card:#f6f7f9}}
@media (prefers-color-scheme:dark){{:root{{--bg:#101217;--fg:#e8e9ec;--mute:#9aa;--line:#2a2d35;--on:#6d8dff;--card:#171a21}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif;margin:0;padding:24px 16px 120px}}
main{{max-width:860px;margin:auto}} h1{{font-size:24px;margin:0 0 6px}} h2{{margin:36px 0 4px;font-size:20px}}
.help,.intro{{color:var(--mute);margin:0 0 14px}} .row{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:10px 0}}
.km{{font-size:21px;line-height:1.9;margin:0 0 8px}} .id{{color:var(--mute);font-size:12px}}
.clips,.choices{{display:flex;flex-wrap:wrap;gap:8px;margin:6px 0}}
button{{font:inherit;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:99px;padding:6px 14px;cursor:pointer}}
button.play.on{{border-color:var(--on);color:var(--on)}} .choices button[aria-pressed=true]{{background:var(--on);border-color:var(--on);color:#fff}}
input{{font:inherit;width:100%;box-sizing:border-box;border:1px solid var(--line);background:var(--bg);color:var(--fg);border-radius:8px;padding:6px 10px}}
.bar{{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 16px;display:flex;gap:12px;align-items:center;justify-content:center}}
.bar button{{background:var(--on);color:#fff;border-color:var(--on)}}
</style></head><body><main><h1>{title}</h1><p class="intro">{intro}</p>{body}</main>
<div class="bar"><span id="count"></span><button id="copy">Copy answers</button></div>
<script>
const KEY="rate:"+document.title;let A={{}};try{{A=JSON.parse(localStorage.getItem(KEY))||{{}}}}catch(e){{}}
const save=()=>{{try{{localStorage.setItem(KEY,JSON.stringify(A))}}catch(e){{}};count()}};
const rows=[...document.querySelectorAll(".row")];
const count=()=>{{document.getElementById("count").textContent=rows.filter(r=>(A[r.dataset.k]||{{}}).c).length+" / "+rows.length+" answered"}};
let cur=null;
rows.forEach(r=>{{const k=r.dataset.k,a=A[k]||{{}};
 r.querySelectorAll(".play").forEach(b=>{{const au=new Audio(b.dataset.src);
  au.onplay=()=>b.classList.add("on");au.onpause=au.onended=()=>b.classList.remove("on");
  b.onclick=()=>{{if(cur&&cur!==au){{cur.pause();cur.currentTime=0}};cur=au;au.paused?(au.currentTime=0,au.play()):au.pause()}}}});
 r.querySelectorAll(".choices button").forEach(b=>{{b.setAttribute("aria-pressed",a.c===b.textContent);
  b.onclick=()=>{{A[k]={{...(A[k]||{{}}),c:b.textContent}};r.querySelectorAll(".choices button").forEach(x=>x.setAttribute("aria-pressed",x===b));save()}}}});
 const n=r.querySelector("input");n.value=a.n||"";n.oninput=()=>{{A[k]={{...(A[k]||{{}}),n:n.value}};save()}}}});
count();
document.getElementById("copy").onclick=()=>{{const t=rows.map(r=>{{const a=A[r.dataset.k]||{{}};
 return r.dataset.k+": "+(a.c||"-")+(a.n?" | "+a.n:"")}}).join("\\n");
 navigator.clipboard.writeText(t).then(()=>{{document.getElementById("copy").textContent="Copied!"}})}};
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    audio = args.out / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    e = html.escape
    parts = []
    for sec in spec["sections"]:
        parts.append(f"<h2>{e(sec['title'])}</h2><p class='help'>{e(sec['help'])}</p>")
        for n, it in enumerate(sec["items"], 1):
            clips = []
            for label, wav in it["clips"]:
                name = f"{sec['key']}_{it['id']}_{label}.wav".replace(" ", "_")
                shutil.copy(wav, audio / name)
                clips.append(
                    f"<button class='play' data-src='audio/{e(name)}'>▶ {e(label)}</button>"
                )
            choices = "".join(f"<button>{e(c)}</button>" for c in sec["choices"])
            parts.append(
                f"<div class='row' data-k='{e(sec['key'])} {e(it['id'])}'>"
                f"<p class='km'>{e(it['text'])}</p><span class='id'>{n}. {e(it['id'])}</span>"
                f"<div class='clips'>{''.join(clips)}</div><div class='choices'>{choices}</div>"
                "<input placeholder='Note (optional)'></div>"
            )
    page = PAGE.format(
        title=e(spec["title"]), intro=e(spec.get("intro", "")), body="".join(parts)
    )
    (args.out / "index.html").write_text(page, encoding="utf-8")
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
