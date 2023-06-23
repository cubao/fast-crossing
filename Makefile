PROJECT_SOURCE_DIR ?= $(abspath ./)
PROJECT_NAME ?= $(shell basename $(PROJECT_SOURCE_DIR))

all:
	@echo nothing special

clean:
	rm -rf build dist wheelhouse *.egg-info
force_clean:
	docker run --rm -v `pwd`:`pwd` -w `pwd` -it alpine/make make clean
.PHONY: clean force_clean

lint:
	pre-commit run -a
lint_install:
	pre-commit install

build:
	mkdir -p build && cd build && \
	cmake .. && make
.PHONY: build

docs_build:
	mkdocs build
docs_serve:
	mkdocs serve -a 0.0.0.0:8088

DOCKER_TAG_WINDOWS ?= ghcr.io/cubao/build-env-windows-x64:v0.0.1
DOCKER_TAG_LINUX ?= ghcr.io/cubao/build-env-manylinux2014-x64:v0.0.3
DOCKER_TAG_MACOS ?= ghcr.io/cubao/build-env-macos-arm64:v0.0.1

test_in_win:
	docker run --rm -w `pwd` -v `pwd`:`pwd` -v `pwd`/build/win:`pwd`/build -it $(DOCKER_TAG_WINDOWS) bash
test_in_mac:
	docker run --rm -w `pwd` -v `pwd`:`pwd` -v `pwd`/build/mac:`pwd`/build -it $(DOCKER_TAG_MACOS) bash
test_in_linux:
	docker run --rm -w `pwd` -v `pwd`:`pwd` -v `pwd`/build/linux:`pwd`/build -it $(DOCKER_TAG_LINUX) bash

DEV_CONTAINER_NAME ?= $(USER)_$(subst /,_,$(PROJECT_NAME)____$(PROJECT_SOURCE_DIR))
DEV_CONTAINER_IMAG ?= $(DOCKER_TAG_LINUX)
test_in_dev_container:
	docker ps | grep $(DEV_CONTAINER_NAME) \
		&& docker exec -it $(DEV_CONTAINER_NAME) bash \
		|| docker run --rm --name $(DEV_CONTAINER_NAME) \
			--network host --security-opt seccomp=unconfined \
			-v `pwd`:`pwd` -w `pwd` -it $(DEV_CONTAINER_IMAG) bash

PYTHON ?= python3
python_install:
	$(PYTHON) setup.py install
python_build:
	$(PYTHON) setup.py bdist_wheel
python_sdist:
	$(PYTHON) setup.py sdist
	# tar -tvf dist/fast_crossing-*.tar.gz
python_test: pytest
pytest:
	pytest tests --capture=tee-sys
.PHONY: python_install python_build python_sdist python_test pytest

# conda create -y -n py36 python=3.6
# conda create -y -n py37 python=3.7
# conda create -y -n py38 python=3.8
# conda create -y -n py39 python=3.9
# conda create -y -n py310 python=3.10
# conda env list
python_build_py36:
	PYTHON=python conda run --no-capture-output -n py36 make python_build
python_build_py37:
	PYTHON=python conda run --no-capture-output -n py37 make python_build
python_build_py38:
	PYTHON=python conda run --no-capture-output -n py38 make python_build
python_build_py39:
	PYTHON=python conda run --no-capture-output -n py39 make python_build
python_build_py310:
	PYTHON=python conda run --no-capture-output -n py310 make python_build
python_build_py311:
	PYTHON=python conda run --no-capture-output -n py311 make python_build
python_build_all: python_build_py36 python_build_py37 python_build_py38 python_build_py39 python_build_py310 python_build_py311
python_build_all_in_linux:
	docker run --rm -w `pwd` -v `pwd`:`pwd` -v `pwd`/build/linux:`pwd`/build -it $(DOCKER_TAG_LINUX) make python_build_all
	make repair_wheels && rm -rf dist/*.whl && mv wheelhouse/*.whl dist && rm -rf wheelhouse
python_build_all_in_macos: python_build_py38 python_build_py39 python_build_py310 python_build_py311
python_build_all_in_windows: python_build_all

repair_wheels:
	python -m pip install auditwheel # sudo apt install patchelf
	ls dist/* | xargs -n1 auditwheel repair --plat manylinux2014_x86_64

pypi_remote ?= pypi
upload_wheels:
	python -m pip install twine
	twine upload dist/*.whl -r $(pypi_remote)

tar.gz:
	tar -cvz --exclude .git -f ../fast_crossing.tar.gz .
	ls -alh ../fast_crossing.tar.gz

NUM_POINTS = 100000
benchmark_point_in_polygon:
	python3 benchmarks/benchmark_point_in_polygon.py generate_test_data --num=$(NUM_POINTS) -o dist/point_in_polygon
	python3 benchmarks/benchmark_point_in_polygon.py shapely \
		dist/point_in_polygon/random_num_$(NUM_POINTS)__bbox_800.00x600.00__radius_250.00__points.npy \
		dist/point_in_polygon/random_num_$(NUM_POINTS)__bbox_800.00x600.00__radius_250.00__polygon.npy \
		dist/mask_shapely.npy
	python3 benchmarks/benchmark_point_in_polygon.py matplotlib \
		dist/point_in_polygon/random_num_$(NUM_POINTS)__bbox_800.00x600.00__radius_250.00__points.npy \
		dist/point_in_polygon/random_num_$(NUM_POINTS)__bbox_800.00x600.00__radius_250.00__polygon.npy \
		dist/mask_matplotlib.npy
	python3 benchmarks/benchmark_point_in_polygon.py cubao \
		dist/point_in_polygon/random_num_$(NUM_POINTS)__bbox_800.00x600.00__radius_250.00__points.npy \
		dist/point_in_polygon/random_num_$(NUM_POINTS)__bbox_800.00x600.00__radius_250.00__polygon.npy \
		dist/mask_cubao.npy
.PHONY: benchmark_point_in_polygon

SYNC_OUTPUT_DIR ?= headers/include/cubao
sync_headers:
	cp src/densify_polyline.hpp $(SYNC_OUTPUT_DIR)
	cp src/fast_crossing.hpp $(SYNC_OUTPUT_DIR)
	cp src/flatbush.h $(SYNC_OUTPUT_DIR)
	cp src/kd_quiver.hpp $(SYNC_OUTPUT_DIR)
	cp src/nanoflann_kdtree.hpp $(SYNC_OUTPUT_DIR)
	cp src/polyline_in_polygon.hpp $(SYNC_OUTPUT_DIR)
	cp src/pybind11_fast_crossing.hpp $(SYNC_OUTPUT_DIR)
	cp src/pybind11_flatbush.hpp $(SYNC_OUTPUT_DIR)
	cp src/pybind11_nanoflann_kdtree.hpp $(SYNC_OUTPUT_DIR)
	cp src/pybind11_quiver.hpp $(SYNC_OUTPUT_DIR)
	cp src/quiver.hpp $(SYNC_OUTPUT_DIR)

# https://stackoverflow.com/a/25817631
echo-%  : ; @echo -n $($*)
Echo-%  : ; @echo $($*)
ECHO-%  : ; @echo $* = $($*)
echo-Tab: ; @echo -n '    '
