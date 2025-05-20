.PHONY: build run clean

model:
	cd tools && mkdir -p model && python3 convert.py

build: model
	mkdir -p build
	cd build && cmake .. && make

run:
	./build/embedai_test

clean:
	rm -rf build
	rm -rf ./tools/model
