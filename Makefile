.PHONY: build run clean

build:
	mkdir -p build
	cd build && cmake .. && make

run:
	./build/embedai_test

clean:
	rm -rf build
