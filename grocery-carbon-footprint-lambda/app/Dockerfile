FROM public.ecr.aws/lambda/python:3.8

COPY app.py requirements.txt app.py carbon_footprint_grocery.csv ./
COPY resnet_test_model.pt /opt/ml/model

RUN python3.8 -m pip install -r requirements.txt -t .

CMD ["app.lambda_handler"]
