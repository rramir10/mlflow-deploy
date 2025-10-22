# filepath: /home/manuelcastiblan/academic/mlflow-deploy/mlflow-deploy/Makefile
	.PHONY: install train validate clean

	install:
	  pip install -r requirements.txt

	train:
	  python src/train.py

	validate:
	  python src/validate.py

	clean:
	  rm -rf mlruns/
	  rm -f model.pkl
