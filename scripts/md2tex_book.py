#!/usr/bin/env python3
"""教材 Markdown -> ctexbook LaTeX 编译器 (xelatex 路线)。

用法:
    python scripts/md2tex_book.py              # 生成 book/build/book.tex
    cd book/build && xelatex book.tex && xelatex book.tex   # 两遍生成目录

处理: 标题/列表/表格(GFM, 数学感知分列)/代码块(BookCode)/图片/
折叠块/块引用/行内与行间公式 (含 \\tag)/链接/特殊符号; 章号与节号
交由 ctex 自动编号 (转换时剥离原文手写编号, 避免双重编号)。
长表格 (>25 行) 自动转为可断页的 longtable。
"""

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BOOK = os.path.join(ROOT, "book")
BUILD = os.path.join(BOOK, "build")

TITLE = "从势能面到波包"
SUBTITLE = "分子反应动力学的量子理论、数值验证与代码实践"
AUTHOR = "AutoQuantum 项目组"
VERSION = "v0.9.1"

PREAMBLE = rf"""
\documentclass[UTF8, a4paper, 11pt, openany]{{ctexbook}}
\usepackage[margin=2.4cm, headheight=32pt]{{geometry}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs,tabularx,array,longtable}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{fancyhdr}}
\usepackage{{xcolor}}
\usepackage[colorlinks=true,linkcolor=black!75,urlcolor=blue!60!black,bookmarksnumbered=true]{{hyperref}}

\setCJKmainfont{{Songti SC}}
\setCJKsansfont{{Heiti SC}}
\setCJKmonofont{{Songti SC}}
\setlength{{\emergencystretch}}{{3em}}

\ctexset{{
  chapter = {{format=\Huge\bfseries, name={{第,章}}, number=\arabic{{chapter}}, beforeskip=0pt, afterskip=24pt}},
  section = {{format=\Large\bfseries}},
  subsection = {{format=\large\bfseries}},
}}

% 代码块: 不使用 Verbatim (会绕过 xeCJK 导致中文缺字), 使用转义文本
\newenvironment{{BookCode}}{{%
  \begin{{quote}}\ttfamily\small\obeyspaces\setlength{{\parskip}}{{0pt}}\leftskip=1.2em}}%
  {{\end{{quote}}}}

\pagestyle{{fancy}}
\fancyhf{{}}
\fancyhead[LE,RO]{{\small 从势能面到波包}}
\fancyhead[RE,LO]{{\small\nouppercase{{\leftmark}}}}
\fancyfoot[C]{{\small \thepage}}
\renewcommand{{\headrulewidth}}{{0.4pt}}
\captionsetup{{font=small,labelformat=empty}}

\hypersetup{{
  pdftitle={{{TITLE}——{SUBTITLE}}},
  pdfauthor={{{AUTHOR}}},
  pdfsubject={{分子反应动力学的量子理论、数值验证与代码实践}}
}}

\begin{{document}}

\begin{{titlepage}}
\centering
\vspace*{{4cm}}
 {{\fontsize{{44}}{{54}}\selectfont\bfseries {TITLE}}}\\[1.0cm]
 {{\Large {SUBTITLE}}}\\[1.2cm]
 \vspace{{1.2cm}}
 {{\large {AUTHOR}}}\\[0.4em]
 {{\large {VERSION}}}
\vfill
 {{\small 分子反应动力学量子计算的教材与代码手册}}
\end{{titlepage}}

\frontmatter
\tableofcontents
\mainmatter
"""

# 正文特殊字符 -> LaTeX (仅作用于非代码/非数学区域)
SYMBOL_MAP = {
    "⚠️": "注意：", "⚠": "注意：", "✅": "是", "❌": "否",
    "✓": "是", "✗": "否",
    "→": r"$\to$", "←": r"$\leftarrow$", "↔": r"$\leftrightarrow$",
    "≈": r"$\approx$", "≠": r"$\neq$", "≤": r"$\leq$", "≥": r"$\geq$",
    "×": r"$\times$", "±": r"$\pm$", "−": "$-$", "·": r"$\cdot$",
    "√": r"$\sqrt{\ }$", "⁡": "",
}
GREEK = {
    "α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$",
    "δ": r"$\delta$", "ε": r"$\varepsilon$", "ζ": r"$\zeta$",
    "η": r"$\eta$", "θ": r"$\theta$", "κ": r"$\kappa$",
    "λ": r"$\lambda$", "μ": r"$\mu$", "ν": r"$\nu$",
    "ξ": r"$\xi$", "ρ": r"$\rho$", "σ": r"$\sigma$",
    "τ": r"$\tau$", "φ": r"$\varphi$", "χ": r"$\chi$",
    "ψ": r"$\psi$", "ω": r"$\omega$", "Γ": r"$\Gamma$",
    "Θ": r"$\Theta$", "Λ": r"$\Lambda$", "Ξ": r"$\Xi$",
    "Π": r"$\Pi$", "Φ": r"$\Phi$", "Ψ": r"$\Psi$",
    "Ω": r"$\Omega$",
}
SUBSCRIPTS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
              "₅": "5", "₆": "6", "₇": "7", "₈": "8", "₉": "9"}
MATH_SYM = {
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "≈": r"\approx$", "≠": r"\neq", "≤": r"\leq", "≥": r"\geq",
    "×": r"\times", "±": r"\pm", "−": "-", "·": r"\cdot",
    "Δ": r"\Delta", "Σ": r"\Sigma", "∇": r"\nabla", "⁡": "",
}
LATEX_SPECIALS = r"\&\%\$\#\_\{\}~^<>"


def math_safe(content: str) -> str:
    for ch, rep in MATH_SYM.items():
        content = content.replace(ch, rep.replace("$", ""))
    # 数学模式中 xeCJK 被绕过: 中文/全角标点包进 \text{}
    cjk = r"[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+"
    content = re.sub(cjk, lambda m: r"\text{" + m.group(0) + "}", content)
    return content


def escape_plain(text: str) -> str:
    out = []
    for ch in text:
        if ch == "\ufe0f":
            continue
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        elif ch in GREEK:
            out.append(GREEK[ch])
        elif ch in SUBSCRIPTS:
            out.append(r"\textsubscript{%s}" % SUBSCRIPTS[ch])
        elif ch in SYMBOL_MAP:
            out.append(SYMBOL_MAP[ch])
        elif ch in LATEX_SPECIALS:
            if ch == "~":
                out.append(r"\textasciitilde{}")
            elif ch == "^":
                out.append(r"\textasciicircum{}")
            elif ch == "<":
                out.append(r"\textless{}")
            elif ch == ">":
                out.append(r"\textgreater{}")
            elif ch == "_":
                out.append(r"\_\allowbreak{}")
            else:
                out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def code_text(raw: str) -> str:
    """代码文本 → 转义 LaTeX, 并在 / - 空格后插入断行点。"""
    t = escape_plain(raw)
    t = t.replace("/", r"/\allowbreak{}")
    t = t.replace("-", r"-\allowbreak{}")
    t = t.replace(" ", r"\ ")
    t = t.replace("--", r"\mbox{-{}-}")   # 防止 -- 变成 en dash
    t = t.replace("(", r"(\allowbreak{}")
    t = t.replace(",", r",\allowbreak{}")
    # 点号后接字母/下划线时允许断行 (API 名); 小数点不处理
    t = re.sub(r"\.(?=[A-Za-z_])", lambda m: ".\\allowbreak{}", t)
    return t


def protect_inline(text: str):
    """保护行内代码与行内数学, 返回占位文本 + 占位表。"""
    spans = []

    def stash(content: str, kind: str) -> str:
        spans.append((kind, content))
        return f"\x00{len(spans) - 1}\x00"

    def code_sub(m):
        return stash("\\texttt{%s}" % code_text(m.group(1)), "raw")

    text = re.sub(r"`([^`]+)`", code_sub, text)
    parts = text.split("$$")
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r"(?<!\$)\$([^$]+?)\$(?!\$)",
                          lambda m: stash("$" + math_safe(m.group(1)) + "$",
                                         "raw"),
                          parts[i])
    return "$$".join(parts), spans


def restore(text: str, spans) -> str:
    def repl(m):
        return spans[int(m.group(1))][1]
    return re.sub(r"\x00(\d+)\x00", repl, text)


def inline(text: str) -> str:
    text = text.strip()
    text, spans = protect_inline(text)
    text = escape_plain(text)
    # 强调处理必须在还原之前: 否则代码/数学里的 * 会被误配对
    text = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"\\emph{\1}", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)",
                  lambda m: "\\href{%s}{%s}" % (
                      m.group(2).replace(r"\allowbreak{}", ""),
                      m.group(1)),
                  text)
    text = re.sub(r"(?<![\w/])(https?://[^\s)]+)", r"\\url{\1}", text)
    return restore(text, spans)


def strip_heading_number(title: str) -> str:
    title = re.sub(r"^第\s*\d+\s*章\s*", "", title)
    title = re.sub(r"^附录\s*[A-Z]?\s*", "", title)
    title = re.sub(r"^(?:\d+|[A-Z])(?:\.\d+)*\.?\s+", "", title)
    return title.strip()


def split_table_row(line: str) -> list:
    """数学/代码感知的 GFM 分列: |psi| 内部的竖线不是列分隔。"""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    cells, buf, in_math, in_code = [], "", False, False
    i = 0
    while i < len(s):
        ch = s[i]
        if in_code:
            buf += ch
            if ch == "`":
                in_code = False
        elif in_math:
            buf += ch
            if ch == "$":
                in_math = False
        elif ch == "`":
            in_code = True
            buf += ch
        elif ch == "$":
            in_math = True
            buf += ch
        elif ch == "|":
            cells.append(buf)
            buf = ""
        else:
            buf += ch
        i += 1
    cells.append(buf)
    return [c.strip() for c in cells]


def is_table_sep(line: str) -> bool:
    return bool(re.match(r"^\|[\s:\-|]+\|$", line.strip()))


def convert_table(rows: list) -> str:
    rows = [r for r in rows if not is_table_sep(r)]
    cells = [split_table_row(r) for r in rows]
    ncol = len(cells[0])
    fixed = []
    for c in cells:  # 列数与表头不一致的行: 多余并入最后一列
        if len(c) > ncol:
            c = c[:ncol - 1] + [" | ".join(c[ncol - 1:])]
        elif len(c) < ncol:
            c = c + [""] * (ncol - len(c))
        fixed.append(c)
    cells = fixed

    def cell(c, bold=False):
        t = inline(c)
        if bold:
            t = "\\textbf{%s}" % t
        if t.startswith("["):  # 会被上一行 \\ 吞作可选参数
            t = "{" + t + "}"
        return t

    n_rows = len(cells) - 1

    def emit(header, rows_subset):
        colspec = "".join(">{\\raggedright\\arraybackslash}X"
                          for _ in range(ncol))
        out = ["\\noindent{\\small",
               "\\begin{tabularx}{\\linewidth}{%s}" % colspec,
               "\\toprule",
               " & ".join(cell(c, bold=True) for c in header) + r" \\",
               "\\midrule"]
        for row in rows_subset:
            out.append(" & ".join(cell(c) for c in row) + r" \\")
        out += ["\\bottomrule", "\\end{tabularx}}"]
        return "\n".join(out)

    def est_height(row):
        # 估算折行数: CJK ~17 字/列, Latin ~28 字符/列 (footnotesize X 列)
        import math as _m
        h = 0
        for c in row:
            cjk = sum(1 for ch in c if '\u4e00' <= ch <= '\u9fff')
            other = len(c) - cjk
            h = max(h, int(_m.ceil(cjk / 17 + other / 28)))
        return max(h, 1)

    BUDGET = 30  # 每块估算不超过 30 行 (页高约 46 行)
    total_h = sum(est_height(r) for r in cells[1:])
    if total_h > BUDGET:
        chunks, cur, cur_h = [], [], 0
        for row in cells[1:]:
            rh = est_height(row)
            if cur and cur_h + rh > BUDGET:
                chunks.append(cur)
                cur, cur_h = [], 0
            cur.append(row)
            cur_h += rh
        if cur:
            chunks.append(cur)
        parts = []
        for k, ch in enumerate(chunks):
            if k:
                parts.append("\\vspace{4pt}")
                parts.append("{\\footnotesize\\color{gray}（续表）}\\par")
            parts.append(emit(cells[0], ch))
        return "\n".join(parts)
    return emit(cells[0], cells[1:])


def convert_math_block(block: str) -> str:
    body = block.strip()
    if body.startswith("$$") and body.endswith("$$"):
        body = body[2:-2].strip()
    if r"\tag{" in body:
        return "\\begin{equation}\n%s\n\\end{equation}" % body
    return "\\[\n%s\n\\]" % body


CJK = (r"[\u4e00-\u9fff\u3001\u3002\uff0c\uff1a\uff08\uff09"
       r"\u2014\u2026\u00b7\u2018\u2019]")


def join_lines(lines: list) -> str:
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not out:
            out = ln
        elif re.search(CJK + "$", out) and re.match("^" + CJK, ln):
            out += ln
        else:
            out += " " + ln
    return out


def convert_file(path: str, first: bool) -> str:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"<div[^>]*>.*?</div>", "", text, flags=re.S)
    lines = text.split("\n")

    out = []
    i = 0
    n = len(lines)
    para_buf = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            out.append(inline(join_lines(para_buf)))
            out.append("")
            para_buf = []

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # 代码块
        if stripped.startswith("```"):
            flush_para()
            i += 1
            code_lines = []
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            out.append("\\noindent\\begin{BookCode}")
            for cl in code_lines:
                esc = code_text(cl)
                if not esc.strip():
                    out.append(r"\strut\\")
                elif esc.lstrip().startswith("["):
                    out.append("{" + esc + r"} \\")
                else:
                    out.append(esc + r" \\")
            out.append("\\end{BookCode}")
            out.append("")
            continue

        # 行间公式 $$ ... $$
        if stripped.startswith("$$"):
            flush_para()
            if stripped.endswith("$$") and len(stripped) > 4:
                out.append(convert_math_block(stripped))
                out.append("")
                i += 1
                continue
            i += 1
            block = []
            while i < n and not lines[i].strip().endswith("$$"):
                block.append(lines[i])
                i += 1
            if i < n:
                block.append(lines[i][:-2] if len(lines[i].strip()) >= 2
                             else "")
                i += 1
            out.append(convert_math_block("\n".join(block)))
            out.append("")
            continue

        # 图片 (图注允许隔一个空行)
        m_img = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$", stripped)
        if m_img:
            flush_para()
            alt, path = m_img.group(1), m_img.group(2)
            i += 1
            j = i
            if j < n and not lines[j].strip():
                j += 1
            caption = None
            if j < n and re.match(r"^\*.+\*\s*$", lines[j].strip()):
                caption = lines[j].strip()
                caption = caption[1:-1].strip()
                i = j + 1
            out.append("\\noindent\\begin{figure}[H]\\centering")
            out.append("\\includegraphics[width=0.88\\textwidth]{%s}" % path)
            if caption:
                out.append("\\captionof*{figure}{%s}" % inline(caption))
            elif alt:
                out.append("\\captionof*{figure}{%s}" % inline(alt))
            out.append("\\end{figure}")
            out.append("")
            continue

        # 标题
        m_h = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if m_h:
            flush_para()
            level = len(m_h.group(1))
            title = strip_heading_number(m_h.group(2).strip())
            if level == 1:
                base = os.path.basename(path)
                if base.startswith("00-"):
                    out.append("\\phantomsection")
                    out.append("\\addcontentsline{toc}{chapter}{%s}"
                               % inline(title))
                    out.append("\\chapter*{%s}" % inline(title))
                else:
                    out.append("\\clearpage")
                    out.append("\\chapter{%s}" % inline(title))
            elif level == 2:
                cmd = "section*" if first else "section"
                out.append("\\%s{%s}" % (cmd, inline(title)))
            elif level == 3:
                cmd = "subsection*" if first else "subsection"
                out.append("\\%s{%s}" % (cmd, inline(title)))
            else:
                out.append("\\subsubsection{%s}" % inline(title))
            out.append("")
            i += 1
            continue

        # 折叠块
        if stripped.startswith("<details>"):
            flush_para()
            i += 1
            summary = ""
            if i < n and lines[i].strip().startswith("<summary>"):
                summary = lines[i].strip()
                summary = summary[len("<summary>"):].replace(
                    "</summary>", "").strip()
                i += 1
            body = []
            while i < n and not lines[i].strip().startswith("</details>"):
                body.append(lines[i])
                i += 1
            i += 1
            out.append("\\begin{quote}\\small")
            if summary:
                out.append("\\textbf{%s}\\par" % inline(summary))
            for bline in body:
                if bline.strip():
                    out.append(inline(bline.strip()) + "\\par")
            out.append("\\end{quote}")
            out.append("")
            continue

        # 表格
        if stripped.startswith("|"):
            flush_para()
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(convert_table(rows))
            out.append("")
            continue

        # 列表 (含嵌套)
        m_li = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", line)
        if m_li:
            flush_para()
            items = []
            while i < n:
                m = re.match(r"^(\s*)([-*+]|\d+\.)\s+(.*)$", lines[i])
                if not m:
                    if (items and lines[i].strip()
                            and not lines[i].strip().startswith(
                                ("#", "|", "```", "$$", "<"))):
                        items[-1][2] += " " + lines[i].strip()
                        i += 1
                        continue
                    break
                items.append([len(m.group(1)),
                              m.group(2).endswith("."),
                              m.group(3).strip()])
                i += 1
            out.append(render_list(items))
            out.append("")
            continue

        # 块引用
        if stripped.startswith(">"):
            flush_para()
            quote = []
            while i < n and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("\\begin{quote}")
            out.append(inline(join_lines(quote)))
            out.append("\\end{quote}")
            out.append("")
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        para_buf.append(stripped)
        i += 1

    flush_para()
    return "\n".join(out)


def render_list(items) -> str:
    indents = sorted({it[0] for it in items})
    level_of = {ind: k for k, ind in enumerate(indents)}
    out = []
    stack = []
    for indent, ordered, content in items:
        lvl = level_of[indent]
        while len(stack) > lvl + 1:
            out.append("\\end{%s}" % stack.pop())
        if len(stack) == lvl + 1:
            prev_ordered = stack[-1] == "enumerate"
            if prev_ordered != ordered:
                out.append("\\end{%s}" % stack.pop())
        if len(stack) <= lvl:
            env = "enumerate" if ordered else "itemize"
            out.append("\\begin{%s}" % env)
            stack.append(env)
        out.append("\\item %s" % inline(content))
    while stack:
        out.append("\\end{%s}" % stack.pop())
    return "\n".join(out)


def main():
    os.makedirs(BUILD, exist_ok=True)
    chapters = sorted(os.listdir(os.path.join(BOOK, "chapters")))
    appendix = sorted(os.listdir(os.path.join(BOOK, "appendix")))

    tex = [PREAMBLE]
    for name in chapters:
        if not name.endswith(".md"):
            continue
        path = os.path.join(BOOK, "chapters", name)
        tex.append(convert_file(path, first=name.startswith("00-")))
    tex.append("\\appendix")
    for name in appendix:
        if not name.endswith(".md"):
            continue
        path = os.path.join(BOOK, "appendix", name)
        body = convert_file(path, first=False)
        body = re.sub(r"^\\clearpage\n", "", body, count=1)
        tex.append(body)
    tex.append("\\end{document}")

    out_path = os.path.join(BUILD, "book.tex")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"wrote {out_path} ({sum(len(t) for t in tex)} chars)")


if __name__ == "__main__":
    sys.exit(main())
