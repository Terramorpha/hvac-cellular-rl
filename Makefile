all: graph.png
	feh graph.png

graph.png: graph.dot
	fdp -Tpng graph.dot > graph.png

graph.dot: main.py
	python3 main.py
