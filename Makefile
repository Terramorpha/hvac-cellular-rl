all: graph.png
	feh graph.png

graph.png: graph.dot
	fdp -Tpng graph.dot > graph.png

graph.dot: graph.py
	python3 $^
