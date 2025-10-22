# filepath: /home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/Makefile
	.PHONY: install train validate clean

	install:
	  pip install -r requirements.txt

	train:
	  python train.py

	validate:
	  python validate.py

	clean:
	  rm -rf mlruns/
	  rm -f model.pkl
