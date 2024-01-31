FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN pip install -U accelerate
RUN pip install -U transformers
RUN pip install -U PyPDF2
RUN pip install python-docx
RUN pip uninstall sympy -y
RUN pip uninstall sympy -y
RUN pip install sympy

RUN mkdir /tmp/c

COPY ./src /src
RUN python3 /src/code.py
