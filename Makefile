# Makefile
install:
 pip install -r requirements.txt
train:
 python src/train.py
validate:
 python src/validate.py
ci: install train validate
cd:
 @echo "Aquí iría el comando de promoción o despliegue del modelo"
 @echo "Por ejemplo: mlflow models serve ..."
