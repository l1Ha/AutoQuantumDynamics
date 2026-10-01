# AutoQuantum — 科学计算软件常用命令

.PHONY: test check benchmark book clean

# 运行全量测试 (97+ 项)
test:
	python3 -m unittest discover -s tests -v

# 本地质量闸门 (语法→导入→测试→配图)
check:
	bash scripts/check.sh

# 微基准 (波包步速/NN 每轮/扫描单点)
benchmark:
	python3 scripts/benchmark.py --quick

# 教材 PDF 构建 (需系统 TeX Live + XeTeX)
book:
	python3 scripts/make_book_figures.py
	python3 scripts/md2tex_book.py
	cd book/build && xelatex -interaction=nonstopmode book.tex
	cd book/build && xelatex -interaction=nonstopmode book.tex
	cd book/build && xelatex -interaction=nonstopmode book.tex
	@echo "PDF: book/build/book.pdf"

# 清理构建产物
clean:
	rm -rf dist build book/build book/figures/*.png **/__pycache__ .pytest_cache
	find . -name "*.pyc" -delete

# 集群操作 (需 SSH 免密)
sync:
	bash scripts/remote.sh sync

remote-test:
	bash scripts/remote.sh test

submit:
	python3 scripts/production.py --system H3_2D --pes leps \
	    --e-min 0.10 --e-max 0.30 --n-points 16 --chunks 4 \
	    --partition liquid_high --rates --output-prefix scan
