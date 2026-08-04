init:
	pip install -e .[tests]

test:
	py.test tests/ --doctest-modules omnimatch/ --doctest-glob='*.md' README.md docs/example.md

doctest:
	py.test --doctest-modules -k "not tests" omnimatch/ --doctest-glob='*.md' README.md docs/example.md

coverage:
	py.test --cov=omnimatch --cov-report html --cov-report term tests/

api-docs:
	rm -rf docs/api
	sphinx-apidoc -e -T -o docs/api omnimatch

docs: api-docs
	python -m sphinx -b html docs docs/_build/html
