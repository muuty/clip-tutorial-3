FROM public.ecr.aws/lambda/python:3.10

RUN yum update -y
RUN yum install -y git

# 경로 정의
WORKDIR ${LAMBDA_TASK_ROOT}

# feature.json, main.py, requirements.txt 파일을 컨테이너의 LAMBDA_TASK_ROOT 경로로 복사
COPY features.json main.py requirements.txt ${LAMBDA_TASK_ROOT}/

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run main.py when the container launches
CMD [ "main.handler" ]
