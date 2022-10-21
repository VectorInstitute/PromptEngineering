.PHONY: install_prompt_dev
install_prompt_dev:

	bash install_python.sh;\
	. ~/.bashrc;\
	pyenv global 3.8.15;\
	python -m pip install --upgrade pip;\
	python -m venv env;\
	. env/bin/activate; \
	pip install -r requirements.txt;\
	pip install -e .[dev]; \
