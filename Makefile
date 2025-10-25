.PHONY: help init lab clean reset deactivate

help:
	@echo "Usage: make [target]"
	@echo "  init        - Set up virtual environment and install dependencies"
	@echo "  lab         - Start JupyterLab"
	@echo "  clean       - Remove virtual environment and unregister Jupyter kernel"
	@echo "  deactivate  - Print how to leave the current .venv"
	@echo "  reset       - Clean and reinitialize environment"

init:
	@echo "[INIT] Creating virtual environment and installing dependencies..."
	@bash install_host.sh
	@.venv/bin/python -m ipykernel install --user --name=ml4g_project1 --display-name "Python (.venv) ml4g_project1" || true

lab:
	@echo "[LAB] Starting JupyterLab..."
	@. .venv/bin/activate && exec python -m jupyterlab

# 不能在子進程裡幫你退出目前 shell 的 .venv
# 所以：如果正在 .venv，就提示你先手動 deactivate，再執行 clean。
clean:
	@echo "[CLEAN] Unregistering Jupyter kernel (if any)..."
	@jupyter kernelspec uninstall -f ml4g_project1 2>/dev/null || true
	@echo "[CLEAN] Checking active venv..."
	@if [ -n "$$VIRTUAL_ENV" ] && [ "$$VIRTUAL_ENV" = "$(abspath .venv)" ]; then \
		echo "⚠️  You are currently INSIDE .venv: $$VIRTUAL_ENV"; \
		echo "   Please run:  deactivate"; \
		echo "   Then rerun:  make clean"; \
	else \
		echo "[CLEAN] Removing .venv directory..."; \
		rm -rf .venv || true; \
		echo "✅ Cleaned."; \
	fi

# 給你一個快捷鍵提醒怎麼離開 .venv
deactivate:
	@if [ -n "$$VIRTUAL_ENV" ]; then \
		echo "You are in: $$VIRTUAL_ENV"; \
		echo "Run:  deactivate"; \
	else \
		echo "Not inside a virtualenv."; \
	fi

reset: clean init
