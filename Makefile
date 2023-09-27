launch:
	curl  http://localhost:4040/api/tunnels > credentials/tunnels.json
	python3 src/z_deployment_functions.py
	python3 src/d_api.py --model-name=$(model)


